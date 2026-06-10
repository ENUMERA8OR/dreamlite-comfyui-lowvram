import os
import select
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from comfy.model_management import InterruptProcessingException
import comfy.model_management as model_management

try:
    import comfy.utils
except Exception:
    comfy = None


def _env_path(name, default):
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


BUNDLED_OFFLOAD_SCRIPT = Path(__file__).resolve().parent / "scripts" / "infer_base_offload.py"
DREAMLITE_PYTHON = _env_path("DREAMLITE_PYTHON", Path("python"))
DREAMLITE_REPO = _env_path("DREAMLITE_REPO", None)
DREAMLITE_MODEL = _env_path("DREAMLITE_BASE_MODEL", None)
DREAMLITE_TMP = _env_path("DREAMLITE_TMP", None)
DREAMLITE_OFFLOAD_SCRIPT = _env_path("DREAMLITE_OFFLOAD_SCRIPT", BUNDLED_OFFLOAD_SCRIPT)
DREAMLITE_TORCH_CACHE = _env_path("DREAMLITE_TORCH_CACHE", None)


def _command_exists(path: Path):
    return path.exists() or shutil.which(str(path)) is not None


class DreamLiteBaseGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 26, "min": 1, "max": 100, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 2147483647, "min": 0, "max": 2**31 - 1}),
                "memory_mode": (["sequential_cpu_offload", "model_cpu_offload", "full_cuda", "cpu"], {"default": "sequential_cpu_offload"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["float16", "bfloat16", "float32"], {"default": "float32"}),
                "attention_mode": (["gqa_query_slicing", "sdpa", "diffusers_slicing"], {"default": "gqa_query_slicing"}),
                "attention_slice_size": ("INT", {"default": 256, "min": 1, "max": 4096, "step": 32}),
            },
            "optional": {"image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "log")
    FUNCTION = "generate"
    CATEGORY = "DreamLite"

    def _make_temp_dir(self):
        if DREAMLITE_TMP is None:
            return Path(tempfile.mkdtemp(prefix="dreamlite_base_comfy_"))
        try:
            DREAMLITE_TMP.mkdir(parents=True, exist_ok=True)
            return Path(tempfile.mkdtemp(prefix="dreamlite_base_comfy_", dir=str(DREAMLITE_TMP)))
        except Exception:
            return Path(tempfile.mkdtemp(prefix="dreamlite_base_comfy_"))

    def _to_tensor(self, image_path: Path):
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        return torch.from_numpy(arr)[None, ...]

    def _save_input_image(self, image, out_dir: Path):
        if image is None:
            return None
        arr = image[0].detach().cpu().numpy()
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")
        image_path = out_dir / "dreamlite_base_input.png"
        img.save(image_path)
        return image_path

    def generate(self, prompt, negative_prompt, steps, width, height, seed, memory_mode, device, dtype, attention_mode, attention_slice_size, image=None):
        if not _command_exists(DREAMLITE_PYTHON):
            raise RuntimeError(f"DreamLite python not found: {DREAMLITE_PYTHON}")
        if DREAMLITE_REPO is None or not DREAMLITE_REPO.exists():
            raise RuntimeError("Set DREAMLITE_REPO to your local DreamLite repository path.")
        if DREAMLITE_MODEL is None or not DREAMLITE_MODEL.exists():
            raise RuntimeError("Set DREAMLITE_BASE_MODEL to your local DreamLite-base model path.")
        if not DREAMLITE_OFFLOAD_SCRIPT.exists():
            raise RuntimeError(f"DreamLite offload script not found: {DREAMLITE_OFFLOAD_SCRIPT}")

        out_dir = self._make_temp_dir()
        out_file = out_dir / "dreamlite_base_output.png"
        input_image_path = self._save_input_image(image, out_dir)

        cmd = [
            str(DREAMLITE_PYTHON),
            str(DREAMLITE_OFFLOAD_SCRIPT),
            "--model_id", str(DREAMLITE_MODEL),
            "--device", device,
            "--offload_mode", memory_mode,
            "--weight_dtype", dtype,
            "--num_inference_steps", str(int(steps)),
            "--width", str(int(width)),
            "--height", str(int(height)),
            "--seed", str(int(seed)),
            "--guidance_scale", "3.5",
            "--image_guidance_scale", "1.0",
            "--output", str(out_file),
            "--attention_mode", attention_mode,
            "--attention_slice_size", str(int(attention_slice_size)),
            "--prompt", prompt,
            "--negative_prompt", negative_prompt,
        ]
        if input_image_path is not None:
            cmd.extend(["--image_path", str(input_image_path)])

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{DREAMLITE_REPO}:{env.get('PYTHONPATH', '')}"
        tmp_for_env = str(out_dir.parent)
        env["TMPDIR"] = tmp_for_env
        env["TEMP"] = tmp_for_env
        env["TMP"] = tmp_for_env
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if DREAMLITE_TORCH_CACHE is not None:
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(DREAMLITE_TORCH_CACHE))

        total_units = max(int(steps), 1) + 1
        pb = comfy.utils.ProgressBar(total_units) if comfy and hasattr(comfy, "utils") else None
        proc = subprocess.Popen(cmd, cwd=str(DREAMLITE_REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)

        lines = []
        progress = 0
        last_heartbeat = time.time()
        try:
            while True:
                model_management.throw_exception_if_processing_interrupted()
                line = ""
                if proc.stdout:
                    ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                    if ready:
                        line = proc.stdout.readline()
                if line:
                    msg = line.rstrip()
                    lines.append(msg)
                    if "Image saved to" in msg and pb:
                        progress = total_units
                        pb.update_absolute(progress, total_units)
                    elif (msg.startswith("Loading diffusers pipeline") or msg.startswith("Attention mode:")) and pb:
                        progress = max(progress, 1)
                        pb.update_absolute(progress, total_units)
                    continue
                if proc.poll() is not None:
                    break
                if pb and progress < max(total_units - 1, 1):
                    now = time.time()
                    if now - last_heartbeat >= 5.0:
                        progress += 1
                        progress = min(progress, max(total_units - 1, 1))
                        pb.update_absolute(progress, total_units)
                        last_heartbeat = now
        except InterruptProcessingException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    pass
            raise

        rc = proc.wait()
        if pb and progress < total_units:
            pb.update_absolute(total_units, total_units)
        if rc != 0:
            raise RuntimeError("DreamLite-base process failed:\n" + "\n".join(lines[-100:]))
        if not out_file.exists():
            raise RuntimeError("DreamLite-base did not produce output image")

        image_tensor = self._to_tensor(out_file)
        return (image_tensor, "\n".join(lines[-60:]))


NODE_CLASS_MAPPINGS = {
    "DreamLiteBaseGenerate": DreamLiteBaseGenerate,
    "DreamLiteBaseGenerateV2": DreamLiteBaseGenerate,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamLiteBaseGenerate": "DreamLite Base Generate",
    "DreamLiteBaseGenerateV2": "DreamLite Base Generate V2",
}
