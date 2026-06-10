# DreamLite-base Offload, GQA Slicing, Dtype, And Quality Session Report

Date: 2026-06-04  
Project root: `/mnt/c/users/aditya/videos/playground/llms/text_to_image`  
Primary goal: make DreamLite-base usable at `1024x1024` on a 4 GB GTX 1650 Ti WSL setup, then document the viable quality path and ComfyUI integration.

## 1. Session Summary

This session started from a working DreamLite-mobile setup and a newly working DreamLite-base install, then focused on why DreamLite-base failed at `1024x1024` on CUDA and how to make it run on constrained VRAM.

The main result:

```text
DreamLite-base can run at 1024x1024 on the 4 GB GTX 1650 Ti using:
- sequential CPU offload
- a custom GQA-aware query-token attention slicer
- float32 for stable quality
- 6-8 inference steps for strong output
```

The final best-known quality recipe from this session was:

```text
model: DreamLite-base
resolution: 1024x1024
device: cuda
memory mode: sequential_cpu_offload
attention mode: gqa_query_slicing
attention slice size: 256
dtype: float32
steps: 8 currently best among tested points
seed: 2147483647
prompt: hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statue isometric view
negative_prompt: low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers
```

The practical conclusion is that the earlier full-CUDA failures were not the final limit of the hardware. With offload and custom attention slicing, DreamLite-base became viable at 1024. `float16` remained numerically unstable, `bfloat16` worked, and `float32` became the best quality/stability path.

## 2. Important Paths

Project root:

```text
/mnt/c/users/aditya/videos/playground/llms/text_to_image
```

Earlier session report:

```text
/mnt/c/users/aditya/videos/playground/llms/text_to_image/sessions/dreamlite_session_2026-06-04.md
```

This session report:

```text
/mnt/c/users/aditya/videos/playground/llms/text_to_image/sessions/dreamlite_base_offload_gqa_session_2026-06-04.md
```

DreamLite repo:

```text
/root/dreamlite-persist/DreamLite
```

DreamLite env:

```text
/root/dreamlite-persist/env
```

DreamLite-base model symlink:

```text
/root/dreamlite-persist/DreamLite/models/DreamLite-base -> /mnt/d/AI/dreamlite-models/DreamLite-base
```

Main new offload CLI:

```text
/root/dreamlite-persist/DreamLite/infer_base_offload.py
```

Separate project-root ComfyUI base node scaffold:

```text
/mnt/c/users/aditya/videos/playground/llms/text_to_image/comfyui-dreamlite-base-node
```

ComfyUI base node files:

```text
comfyui-dreamlite-base-node/__init__.py
comfyui-dreamlite-base-node/dreamlite_base_node.py
comfyui-dreamlite-base-node/README.md
comfyui-dreamlite-base-node/CHANGELOG.md
comfyui-dreamlite-base-node/RELEASE_CHECKLIST.md
comfyui-dreamlite-base-node/.gitignore
comfyui-dreamlite-base-node/examples/dreamlite_base_t2i_workflow.json
```

Output directory:

```text
/mnt/d/AI/output
D:\AI\output
```

Cache/temp directories used:

```text
/mnt/d/AI/cache/tmp
/mnt/d/AI/cache/torch
```

## 3. Initial DreamLite-base Command

The user first asked to convert a DreamLite-mobile command to DreamLite-base.

Original mobile-style command used:

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_mobile.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-mobile --device cuda --weight_dtype float32 --num_inference_steps 4 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/horse_seed1024.png --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statute isometric view --negative low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

Converted DreamLite-base command used `infer.py`, DreamLite-base model path, and real `--negative_prompt`:

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-base --device cuda --weight_dtype float32 --num_inference_steps 4 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/base_warrior_glass_seed2147483647.png --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statue isometric view' --negative_prompt 'low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

Important correction made during session:

```text
statute -> statue
```

The user reported that DreamLite-base quality was much better than DreamLite-mobile when it ran, but it was slower and more memory constrained.

## 4. Initial ComfyUI Base Node Scaffold

A separate folder was created under the project root:

```text
comfyui-dreamlite-base-node
```

The first version included:

```text
__init__.py
dreamlite_base_node.py
README.md
CHANGELOG.md
RELEASE_CHECKLIST.md
.gitignore
examples/dreamlite_base_t2i_workflow.json
```

Initial node name:

```text
DreamLite Base Generate
```

Initial node category:

```text
DreamLite
```

Initial node supported:

```text
prompt
negative_prompt
steps
width
height
seed
guidance_scale
image_guidance_scale
device
dtype
optional image input
```

The node was built separately from the mobile node, as requested. It was intentionally not installed into `/root/comfy/ComfyUI/custom_nodes/` at first because the user asked to keep everything in a separate root-project folder.

## 5. Full CUDA Failures

### 5.1 Full CUDA float32 OOM

The first DreamLite-base full-CUDA float32 run at `1024x1024`, `4 steps` failed during generation:

```text
torch.OutOfMemoryError: CUDA out of memory
```

Important details from the error:

```text
GPU total capacity: 4.00 GiB
0 bytes free
allocated memory was reported as much larger than physical VRAM
```

The giant non-PyTorch memory number looked like a WSL/CUDA accounting bug, but the important fact was clear: full CUDA float32 exceeded the 4 GB VRAM capacity.

### 5.2 Full CUDA float16 OOM

Next test used:

```text
--weight_dtype float16
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It failed earlier, during:

```text
pipeline.to("cuda")
```

Representative error:

```text
RuntimeError: CUDA driver error: out of memory
```

Conclusion:

```text
DreamLite-base cannot be fully moved to CUDA on this 4 GB GPU, even in float16.
```

This meant resolution was not the only problem. Full model residency itself did not fit.

## 6. Offload Plan

The user asked whether the pipeline could be batched or streamed so only the needed part stays in VRAM and is flushed afterward.

The correct term identified was:

```text
CPU/GPU offload
```

Diffusers/Accelerate capabilities checked in the installed DreamLite env:

```text
diffusers 0.38.0
torch 2.11.0+cu128
accelerate 1.13.0
```

Pipeline capability check showed:

```text
enable_model_cpu_offload: True
enable_sequential_cpu_offload: True
enable_attention_slicing: True
enable_vae_slicing: False
enable_vae_tiling: False
```

Key explanation:

- `enable_model_cpu_offload()` moves larger model components between CPU and GPU.
- `enable_sequential_cpu_offload()` is more aggressive and offloads submodules/layers, using less VRAM but running slower.
- `enable_attention_slicing()` reduces attention memory but later proved incompatible with DreamLite-base GQA layers.
- VAE slicing/tiling were not exposed by DreamLite's pipeline.

## 7. New Offload Script

A separate CLI script was created so the original `infer.py` stayed unchanged:

```text
/root/dreamlite-persist/DreamLite/infer_base_offload.py
```

Initial purpose:

- Avoid `pipeline.to("cuda")` for constrained VRAM.
- Support `sequential_cpu_offload` and `model_cpu_offload`.
- Support real `negative_prompt`.
- Support `float16`, `bfloat16`, `float32`.
- Keep D-drive temp/cache usage.

Initial CLI supported:

```text
--model_id
--device
--offload_mode none|model_cpu_offload|sequential_cpu_offload|cpu
--weight_dtype float16|bfloat16|float32
--num_inference_steps
--prompt
--negative_prompt
--image_path
--width
--height
--guidance_scale
--image_guidance_scale
--seed
--output
--attention_slicing / --no-attention_slicing
```

Later it was upgraded to:

```text
--attention_mode sdpa|gqa_query_slicing|diffusers_slicing
--attention_slice_size
```

## 8. First Offload Test And Attention Slicing Failure

A `1024x1024`, `float16`, `1 step`, `sequential_cpu_offload`, attention slicing test got past the previous `.to("cuda")` OOM.

This proved:

```text
sequential CPU offload works and avoids the full-CUDA model residency failure.
```

But generation then failed in DreamLite attention:

```text
RuntimeError: Expected size for first two dimensions of batch2 tensor to be: [2, 64] but got: [2, 16]
```

The failure path:

```text
dreamlite/models/attention_processor.py
SlicedAttnProcessor
attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
torch.baddbmm(...)
```

Root cause:

```text
Diffusers-style SlicedAttnProcessor assumes query heads and key/value heads are shaped the same way.
DreamLite-base uses grouped-query attention in places:
query heads != key/value heads
```

Specifically, DreamLite's normal `AttnProcessor2_0` handles:

```python
query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
key = key.view(batch_size, -1, attn.kv_heads, head_dim).transpose(1, 2)
value = value.view(batch_size, -1, attn.kv_heads, head_dim).transpose(1, 2)
```

But the generic sliced processor used `head_to_batch_dim()` for key/value in a way that did not respect `kv_heads`, causing the head-dimension mismatch.

## 9. Custom GQA-aware Query Slicing

A custom processor was added to `infer_base_offload.py`:

```text
GQAQuerySlicedAttnProcessor
```

And a helper:

```text
enable_gqa_query_slicing(pipeline, query_slice_size=256)
```

Core design:

- Do not use generic Diffusers `SlicedAttnProcessor`.
- Keep query heads and key/value heads separate.
- Slice along the query-token axis, not the packed `batch * heads` dimension.
- Use PyTorch `scaled_dot_product_attention` with `enable_gqa=True` when `kv_heads != heads`.

Conceptual shape handling:

```text
query: [batch, query_heads, query_tokens, head_dim]
key:   [batch, kv_heads,    key_tokens,   head_dim]
value: [batch, kv_heads,    key_tokens,   head_dim]
```

Pseudo-flow:

```python
for start in range(0, query_tokens, query_slice_size):
    query_chunk = query[:, :, start:end, :]
    output_chunk = F.scaled_dot_product_attention(
        query_chunk,
        key,
        value,
        attn_mask=mask_chunk,
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=True when needed,
    )
concat chunks along query-token axis
```

Observed activation:

```text
Attention mode: gqa_query_slicing, slice_size=256, enabled=60, skipped=0
```

The processor was validated with a small unit-style shape test:

```text
query heads: 8
kv heads: 2
head dim: 16
output: (2, 7, 128)
finite: True
```

A second validation compared full-query chunk vs sliced query chunks:

```text
max_abs_diff: 5.960464477539063e-08
allclose: True
```

This established that the custom slicing was mathematically consistent for the tested GQA case.

## 10. Updated ComfyUI Base Node

The ComfyUI base node was updated to expose:

```text
memory_mode:
- sequential_cpu_offload
- model_cpu_offload
- full_cuda
- cpu

attention_mode:
- gqa_query_slicing
- sdpa
- diffusers_slicing

attention_slice_size
```

The node was simplified to call:

```text
/root/dreamlite-persist/DreamLite/infer_base_offload.py
```

directly through the DreamLite env rather than duplicating the entire pipeline code inside the node.

This makes the Comfy node easier to maintain because the offload/GQA logic now lives in one CLI file.

Current Comfy node scaffold path:

```text
/mnt/c/users/aditya/videos/playground/llms/text_to_image/comfyui-dreamlite-base-node/dreamlite_base_node.py
```

Current README default profile:

```text
memory_mode: sequential_cpu_offload
dtype: float16 initially, but session evidence later showed float32 is the quality profile
attention_mode: gqa_query_slicing
attention_slice_size: 256
resolution: 1024x1024
```

Important note: after the final float32 tests, the README/node defaults may deserve one more pass to make `float32` + 8 steps the quality preset, while keeping bfloat16 as a faster fallback.

## 11. Dtype Findings

### 11.1 Float16

`float16` with offload and GQA slicing successfully generated at `1024x1024`, `1 step`, but the image was black.

Conclusion:

```text
float16 fits, but is numerically unstable on this stack for DreamLite-base.
```

This matched earlier mobile observations where `float16` intermittently produced black images.

### 11.2 Bfloat16

`bfloat16` worked.

First `bfloat16`, `1 step` command:

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp TORCHINDUCTOR_CACHE_DIR=/mnt/d/AI/cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_base_offload.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-base --device cuda --offload_mode sequential_cpu_offload --weight_dtype bfloat16 --num_inference_steps 1 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/base_warrior_glass_1024_gqa_bf16_s1.png --attention_mode gqa_query_slicing --attention_slice_size 256 --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statue isometric view' --negative_prompt 'low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

Result:

```text
Not black.
Valid rough running humanoid silhouette.
Only 1 step, so not a quality result.
```

Pixel stats for `base_warrior_glass_1024_gqa_bf16_s1.png`:

```text
size: 1024x1024
RGB extrema: full 0-255 range
mean RGB: [59.16, 179.91, 176.14]
stddev RGB: [75.64, 58.34, 61.19]
near_black: 0.0015%
near_white: 2.8831%
```

`bfloat16`, `4 steps` output:

```text
/mnt/d/AI/output/base_warrior_glass_1024_gqa_bf16_s4.png
```

Visual assessment:

```text
Coherent running humanoid.
Glass/crystal material clear.
Turquoise translucency strong.
Platform/base present.
More glass runner statue than culturally specific ancient Indian warrior.
Some overexposure/clipping.
```

Pixel stats:

```text
size: 1024x1024
mean RGB: [51.51, 173.04, 159.35]
stddev RGB: [77.43, 62.18, 64.56]
near_black: 0.4932%
near_white: 3.8537%
any channel saturated: 15.4073%
```

Conclusion:

```text
bfloat16 is viable and much more stable than float16.
```

### 11.3 Float32

`float32` became viable once `sequential_cpu_offload` and `gqa_query_slicing` were in place.

`float32`, `1 step` output:

```text
/mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s1.png
```

Note: this file was later overwritten by a 4-step run due to a misleading output filename. The original 1-step stats were captured before overwrite:

```text
size: 1024x1024
RGB extrema: R 0-255, G 0-255, B 29-255
mean RGB: [81.47, 191.79, 193.4]
stddev RGB: [65.72, 49.22, 47.18]
near_black: 0.0%
near_white: 0.7902%
any channel saturated: 24.6675%
```

Visual assessment of original 1-step float32:

```text
Not black.
Same rough running humanoid silhouette as bfloat16.
Smoother/cleaner background than bfloat16 smoke test.
Still only 1 step, so only a stability smoke test.
```

A 4-step float32 run was accidentally saved over the same `_s1.png` file. It succeeded and looked stronger than bfloat16.

Then a 6-step float32 run was saved to:

```text
/mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s4.png
```

Note: filename says `_s4`, but the command used `--num_inference_steps 6`.

6-step float32 stats:

```text
size: 1024x1024
mean RGB: [88.21, 174.28, 169.26]
stddev RGB: [59.14, 51.87, 53.8]
near_black: 0.329%
near_white: 2.1128%
any channel saturated: 9.1646%
```

Visual assessment:

```text
Clearly best result up to that point.
Clean running humanoid silhouette.
Glass material strong.
Better anatomy than 4 steps.
More warrior-like due to headpiece/helmet, belt, ornament shapes, statue/base presentation.
Less overexposed than earlier outputs.
```

Finally an 8-step float32 run was saved to:

```text
/mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s4_8.png
```

Note: filename is messy, but command used `--num_inference_steps 8`.

8-step float32 stats:

```text
size: 1024x1024
mean RGB: [88.66, 162.06, 155.67]
stddev RGB: [59.53, 56.1, 55.48]
near_black: 0.4079%
near_white: 1.4696%
any channel saturated: 7.1551%
```

Compared with 6-step:

```text
6-step near_white: 2.1128%
8-step near_white: 1.4696%

6-step any-channel saturated: 9.1646%
8-step any-channel saturated: 7.1551%
```

Visual assessment:

```text
8 steps was better than 6.
Cleaner anatomy and silhouette.
Better profile/face shape.
More convincing headpiece/warrior ornament.
Torso, belt, skirt/cloth, and platform details were more controlled.
Glass material still read clearly.
Pose was more stable and less warped.
```

Current step-quality ordering from tested points:

```text
4 steps < 6 steps < 8 steps
```

The session ended before testing 10 or 12 steps, so 8 steps is the best tested point, not a proven Pareto frontier.

## 12. Numeric Precision Explanation

The user asked how `weight_dtype` affects the math of the pipeline.

Core explanation:

`float16`, `bfloat16`, and `float32` are numeric formats / precision types, not mathematical dimensions.

Matrix dimensions are things like:

```text
1024 x 1024
batch x tokens x channels
```

The dtype controls how each number inside those matrices/tensors is represented.

```text
float16: 16-bit floating point, less memory, faster, smaller numeric range, can overflow/underflow
bfloat16: 16-bit floating point, less memory, wider exponent range than float16, often more stable
float32: 32-bit floating point, more memory, more precision/range, usually most stable
```

The same equations run under different number systems.

Important equations discussed:

```text
y = W x + b
attention = softmax(Q K^T / sqrt(d)) V
noise_prediction = UNet(latent, text_embeddings, timestep)
next_latent = scheduler_step(latent, noise_prediction)
image = VAE_decode(latent)
```

How they connect:

1. The text encoder turns prompt tokens into embeddings using learned transformations and attention.
2. The UNet takes current noisy latent, text embeddings, and timestep.
3. Inside the UNet, many layers use `y = W x + b` and attention.
4. Cross-attention lets image latent features query text concepts.
5. The UNet predicts noise to remove.
6. The scheduler uses the current latent and predicted noise to produce the next cleaner latent.
7. After denoising steps finish, the VAE decodes the final latent into RGB pixels.

Detailed attention interaction:

```text
Q = Wq latent + bq
K = Wk text + bk
V = Wv text + bv
attention = softmax(Q K^T / sqrt(d)) V
```

So `y = W x + b` is not separate from attention; it is how Q, K, and V are produced.

For cross-attention:

```text
Q comes from image/latent features.
K and V come from prompt/text embeddings.
```

This is how prompt conditioning interacts with the current noisy image state. The current latent asks what prompt information is relevant, and the prompt/text vectors provide concepts to inject back into the image features.

Scheduler/VAE interaction:

```text
for each timestep t:
    noise_prediction_t = UNet(latent_t, text_embeddings, t)
    latent_t_minus_1 = scheduler_step(latent_t, noise_prediction_t, t)

final_latent = latent_0
image = VAE_decode(final_latent)
```

The scheduler operates in latent space. It does not directly create RGB pixels. The VAE decoder turns the final latent into an image.

## 13. PDF Created

A PDF explanation was created in the project root:

```text
/mnt/c/users/aditya/videos/playground/llms/text_to_image/dreamlite_pipeline_math_explanation.pdf
```

It explains:

- Prompt-to-embedding flow.
- `y = W x + b`.
- Attention equation.
- How text conditioning and latent image state interact.
- Scheduler updates.
- VAE decoding.
- Why dtype matters.

The PDF was generated directly as a simple PDF because no local `pandoc`, `wkhtmltopdf`, `reportlab`, `fpdf`, or `matplotlib` installation was available.

## 14. Key Commands

### 14.1 Working bfloat16 4-step command

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp TORCHINDUCTOR_CACHE_DIR=/mnt/d/AI/cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_base_offload.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-base --device cuda --offload_mode sequential_cpu_offload --weight_dtype bfloat16 --num_inference_steps 4 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/base_warrior_glass_1024_gqa_bf16_s4.png --attention_mode gqa_query_slicing --attention_slice_size 256 --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statue isometric view' --negative_prompt 'low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

### 14.2 Working float32 6-step command

Note: output filename says `_s4`, but this command used `6` steps.

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp TORCHINDUCTOR_CACHE_DIR=/mnt/d/AI/cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_base_offload.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-base --device cuda --offload_mode sequential_cpu_offload --weight_dtype float32 --num_inference_steps 6 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s4.png --attention_mode gqa_query_slicing --attention_slice_size 256 --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statue isometric view' --negative_prompt 'low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

### 14.3 Working float32 8-step command

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp TORCHINDUCTOR_CACHE_DIR=/mnt/d/AI/cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_base_offload.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-base --device cuda --offload_mode sequential_cpu_offload --weight_dtype float32 --num_inference_steps 8 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s4_8.png --attention_mode gqa_query_slicing --attention_slice_size 256 --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statue isometric view' --negative_prompt 'low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

### 14.4 Cleaner future command naming

Recommended future naming pattern:

```text
/mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s08_seed2147483647.png
/mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s10_seed2147483647.png
```

This avoids confusion like `_s4_8` or `_s4` for a 6-step render.

## 15. Image Quality Findings

### 15.1 One-step outputs

One-step outputs are stability smoke tests only.

They can show:

```text
- whether generation runs
- whether image is black
- whether prompt conditioning creates a rough silhouette
- whether seed gives usable composition
```

They should not be used to judge final quality.

Important observation:

```text
The one-step bfloat16 and float32 outputs already showed an apparent running humanoid silhouette.
```

That meant prompt conditioning, seed, offload path, and GQA slicing were working.

### 15.2 Four-step outputs

Four steps produced coherent images but were not the final frontier for base quality.

bfloat16 4-step was viable but more clipped/less controlled than later float32 results.

Float32 4-step looked stronger than bfloat16, but was later surpassed by 6 and 8 steps.

### 15.3 Six-step float32

Six steps was the first clearly strong DreamLite-base float32 result at 1024.

It improved:

```text
- anatomy
- running pose
- glass material clarity
- platform coherence
- warrior-like ornament hints
```

### 15.4 Eight-step float32

Eight steps improved further and is the best tested point.

It improved:

```text
- face/profile
- headpiece
- torso detail
- belt/skirt ornament
- platform detailing
- clipping/overexposure
- silhouette stability
```

Current quality ordering from tested points:

```text
float16: unstable / black
bfloat16 4 steps: viable but less controlled
float32 4 steps: stronger
float32 6 steps: clearly good
float32 8 steps: best tested
```

## 16. Pareto Frontier Discussion

The user correctly challenged the idea that 6 steps represented the inference-step Pareto frontier.

Conclusion:

```text
6 steps was not the Pareto frontier.
It was only the first clearly successful high-quality point.
```

A real Pareto frontier would require a controlled sweep across:

```text
steps: 4, 6, 8, 10, 12
same prompt
same seed
same dtype
same memory mode
same attention mode
same resolution
```

After testing 8 steps, the curve had not obviously flattened because 8 improved over 6.

Recommended next tests:

```text
10 steps
12 steps only if 10 still improves meaningfully
additional seeds if prompt adherence remains weaker than desired
```

## 17. Current Best Known Recipe

Best tested quality profile:

```text
DreamLite-base
1024x1024
float32
sequential_cpu_offload
gqa_query_slicing
attention_slice_size=256
8 steps
seed=2147483647
guidance_scale=3.5
image_guidance_scale=1.0
```

Best tested command:

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp TORCHINDUCTOR_CACHE_DIR=/mnt/d/AI/cache/torch PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_base_offload.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-base --device cuda --offload_mode sequential_cpu_offload --weight_dtype float32 --num_inference_steps 8 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s08_seed2147483647.png --attention_mode gqa_query_slicing --attention_slice_size 256 --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statue isometric view' --negative_prompt 'low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

Faster fallback profile:

```text
bfloat16
4-8 steps
same offload and GQA slicing
```

Avoid for this setup:

```text
float16 for DreamLite-base final renders, because it produced black image output.
full_cuda for DreamLite-base on 4 GB VRAM, because it OOMs.
diffusers_slicing for DreamLite-base, because it is incompatible with GQA and caused a shape error.
```

## 18. Remaining Improvement Ideas

1. Test `10` and possibly `12` steps with float32.
2. Use cleaner output filenames that include dtype, steps, seed, and attention mode.
3. Update ComfyUI node defaults from the older bfloat16/4-step profile to the newer float32/8-step quality profile, or add separate quality/fast presets in documentation.
4. Add JSON sidecar logging to `infer_base_offload.py` for every output image.
5. Add a seed sweep helper using `infer_base_offload.py`.
6. Test `model_cpu_offload` with float32 and GQA slicing to see whether it can buy back speed without OOM.
7. Test smaller `attention_slice_size` values only if OOM returns:

```text
256 -> 128 -> 64
```

8. Improve prompt adherence for ancient Indian warrior identity with more explicit clothing and silhouette terms.
9. Compare base vs mobile using the same improved prompt and seed sweep.
10. Install/copy the Comfy node into `/root/comfy/ComfyUI/custom_nodes/` and verify the UI node works end to end.

## 19. Technical Caveats

- The PowerShell `wsl : NativeCommandError` header appeared repeatedly even when generation completed successfully. The important signal was whether the Python process ended with a traceback and whether the output image was saved.
- File naming was inconsistent during experimentation. Some files contain step counts in names that do not match the actual command.
- One-step images should not be used for quality judgment.
- `float32` succeeded only after sequential offload and GQA slicing were implemented.
- `bfloat16` worked and was faster/loading sometimes appeared quicker, but float32 produced the best visual quality in the final tests.
- VAE slicing/tiling were unavailable in the current DreamLite pipeline API.
- Generic Diffusers attention slicing remains available as a CLI mode named `diffusers_slicing`, but it is known to fail on DreamLite-base GQA attention and should not be used for normal runs.

## 20. Final State At End Of Session

Working:

```text
DreamLite-base 1024x1024 with float32 on 4 GB GPU using sequential offload and GQA query slicing.
```

Best tested output:

```text
/mnt/d/AI/output/base_warrior_glass_1024_gqa_f32_s4_8.png
```

Best known settings:

```text
--weight_dtype float32
--num_inference_steps 8
--offload_mode sequential_cpu_offload
--attention_mode gqa_query_slicing
--attention_slice_size 256
--width 1024
--height 1024
--seed 2147483647
```

Key implementation added:

```text
/root/dreamlite-persist/DreamLite/infer_base_offload.py
```

Key node scaffold updated:

```text
/mnt/c/users/aditya/videos/playground/llms/text_to_image/comfyui-dreamlite-base-node
```

Next best action:

```text
Run 10-step float32 with the same settings and clean output name, then compare against the 8-step result.
```
