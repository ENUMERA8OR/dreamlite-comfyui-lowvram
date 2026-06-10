# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0.

import argparse
import hashlib
import re
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers.utils import load_image

from dreamlite import DreamLitePipeline

warnings.filterwarnings("ignore")



class GQAQuerySlicedAttnProcessor:
    def __init__(self, query_slice_size=256):
        self.query_slice_size = max(int(query_slice_size), 1)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = attn.inner_dim
        head_dim = inner_dim // attn.heads
        kv_heads = getattr(attn, "kv_heads", attn.heads)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if kv_heads != attn.heads and attn.heads % kv_heads != 0:
            raise RuntimeError(f"Cannot use GQA slicing: query heads {attn.heads} is not divisible by kv heads {kv_heads}.")

        sdpa_kwargs = {"dropout_p": 0.0, "is_causal": getattr(attn, "is_causal", False)}
        if kv_heads != attn.heads:
            sdpa_kwargs["enable_gqa"] = True

        query_tokens = query.shape[2]
        chunks = []
        for start in range(0, query_tokens, self.query_slice_size):
            end = min(start + self.query_slice_size, query_tokens)
            query_chunk = query[:, :, start:end, :]

            mask_chunk = attention_mask
            if attention_mask is not None and attention_mask.shape[-2] == query_tokens:
                mask_chunk = attention_mask[:, :, start:end, :]

            chunks.append(F.scaled_dot_product_attention(query_chunk, key, value, attn_mask=mask_chunk, **sdpa_kwargs))

        hidden_states = torch.cat(chunks, dim=2)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


def enable_gqa_query_slicing(pipeline, query_slice_size=256):
    target = getattr(pipeline, "unet", pipeline)
    enabled = 0
    skipped = 0

    for module in target.modules():
        if not hasattr(module, "set_processor"):
            continue
        if not all(hasattr(module, name) for name in ("to_q", "to_k", "to_v", "to_out", "heads", "inner_dim")):
            skipped += 1
            continue
        if getattr(module, "added_kv_proj_dim", None) is not None or getattr(module, "to_out", None) is None:
            skipped += 1
            continue

        module.set_processor(GQAQuerySlicedAttnProcessor(query_slice_size=query_slice_size))
        enabled += 1

    return enabled, skipped



def parse_args():
    parser = argparse.ArgumentParser(description="DreamLite-base inference with CPU/GPU offload")
    parser.add_argument("--model_id", type=str, default="models/DreamLite-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--offload_mode", type=str, default="sequential_cpu_offload", choices=["none", "model_cpu_offload", "sequential_cpu_offload", "cpu"])
    parser.add_argument("--weight_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--prompt", type=str, default="a dog running on the grass")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--attention_mode", type=str, default="gqa_query_slicing", choices=["sdpa", "gqa_query_slicing", "diffusers_slicing"])
    parser.add_argument("--attention_slice_size", type=int, default=256)
    parser.add_argument("--attention_slicing", action=argparse.BooleanOptionalAction, default=None, help="Compatibility alias: true selects gqa_query_slicing, false selects sdpa.")
    return parser.parse_args()


def normalize_args(args):
    if args.attention_slicing is True:
        args.attention_mode = "gqa_query_slicing"
    elif args.attention_slicing is False:
        args.attention_mode = "sdpa"
    args.attention_slice_size = max(int(args.attention_slice_size), 1)
    return args


def build_safe_output_name(prompt: str, ext: str = ".png", max_stem_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt).strip("._")
    if not slug:
        slug = "dreamlite_output"
    slug = slug[:max_stem_len].rstrip("._")
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}{ext}"


def dtype_from_name(name: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def configure_attention(pipeline, args):
    if args.attention_mode == "gqa_query_slicing":
        enabled, skipped = enable_gqa_query_slicing(pipeline, args.attention_slice_size)
        print(f"Attention mode: gqa_query_slicing, slice_size={args.attention_slice_size}, enabled={enabled}, skipped={skipped}")
    elif args.attention_mode == "diffusers_slicing":
        pipeline.enable_attention_slicing()
        print("Attention mode: diffusers_slicing")
    else:
        print("Attention mode: sdpa")


def configure_pipeline(pipeline, args):
    cuda_requested = args.device.startswith("cuda") and args.offload_mode != "cpu"
    cuda_available = torch.cuda.is_available()

    if cuda_requested and cuda_available:
        torch.cuda.empty_cache()

    configure_attention(pipeline, args)

    if args.offload_mode == "cpu" or not cuda_requested or not cuda_available:
        if cuda_requested and not cuda_available:
            print("CUDA requested but unavailable; using CPU.")
        pipeline.to("cpu")
        print("Memory mode: cpu")
        return pipeline

    if args.offload_mode == "sequential_cpu_offload":
        pipeline.enable_sequential_cpu_offload(device=args.device)
        print(f"Memory mode: sequential_cpu_offload on {args.device}")
        return pipeline

    if args.offload_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=args.device)
        print(f"Memory mode: model_cpu_offload on {args.device}")
        return pipeline

    pipeline.to(args.device)
    print(f"Memory mode: full cuda via .to({args.device})")
    return pipeline


def main():
    args = normalize_args(parse_args())
    weight_dtype = dtype_from_name(args.weight_dtype)

    print(f"Loading diffusers pipeline from: {args.model_id}")
    print(f"dtype: {args.weight_dtype}")

    pipeline = DreamLitePipeline.from_pretrained(args.model_id, torch_dtype=weight_dtype)
    pipeline = configure_pipeline(pipeline, args)

    prompt = args.prompt
    input_image = load_image(args.image_path) if args.image_path else None
    if input_image is not None:
        width, height = input_image.size
    else:
        width, height = args.width, args.height

    print("Generating image...")
    image = pipeline(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        image=input_image,
        height=height,
        width=width,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]

    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)

    out_path = Path(args.output if args.output else build_safe_output_name(prompt))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"Image saved to {out_path}")


if __name__ == "__main__":
    main()
