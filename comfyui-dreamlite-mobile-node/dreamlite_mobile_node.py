import json
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


DREAMLITE_PYTHON = _env_path("DREAMLITE_PYTHON", Path("python"))
DREAMLITE_REPO = _env_path("DREAMLITE_REPO", None)
DREAMLITE_MODEL = _env_path("DREAMLITE_MOBILE_MODEL", None)
DREAMLITE_TMP = _env_path("DREAMLITE_TMP", None)
DREAMLITE_TORCH_CACHE = _env_path("DREAMLITE_TORCH_CACHE", None)
DREAMLITE_MOBILE_DEVICE = os.environ.get("DREAMLITE_MOBILE_DEVICE", "cuda")
DREAMLITE_MOBILE_MEMORY_MODE = os.environ.get("DREAMLITE_MOBILE_MEMORY_MODE", "sequential_cpu_offload")


def _command_exists(path: Path):
    return path.exists() or shutil.which(str(path)) is not None


class DreamLiteMobileGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "seed": ("INT", {"default": 2147483647, "min": 0, "max": 2**31 - 1}),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "log")
    FUNCTION = "generate"
    CATEGORY = "DreamLite"

    def _to_tensor(self, image_path: Path):
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        # Comfy expects [B,H,W,C]
        return torch.from_numpy(arr)[None, ...]

    def generate(self, prompt, steps, width, height, seed, dtype):
        if not _command_exists(DREAMLITE_PYTHON):
            raise RuntimeError(f"DreamLite python not found: {DREAMLITE_PYTHON}")
        if DREAMLITE_REPO is None or not DREAMLITE_REPO.exists():
            raise RuntimeError("Set DREAMLITE_REPO to your local DreamLite repository path.")
        if DREAMLITE_MODEL is None or not DREAMLITE_MODEL.exists():
            raise RuntimeError("Set DREAMLITE_MOBILE_MODEL to your local DreamLite-mobile model path.")

        selected_device = DREAMLITE_MOBILE_DEVICE
        selected_memory_mode = DREAMLITE_MOBILE_MEMORY_MODE
        selected_dtype = dtype
        if selected_device == "cpu" or selected_memory_mode == "cpu":
            selected_device = "cpu"
            selected_memory_mode = "cpu"
        if selected_device == "cpu" and dtype in ("float16", "bfloat16"):
            selected_dtype = "float32"

        if DREAMLITE_TMP is None:
            out_dir = Path(tempfile.mkdtemp(prefix="dreamlite_mobile_comfy_"))
        else:
            try:
                DREAMLITE_TMP.mkdir(parents=True, exist_ok=True)
                out_dir = Path(tempfile.mkdtemp(prefix="dreamlite_mobile_comfy_", dir=str(DREAMLITE_TMP)))
            except Exception:
                out_dir = Path(tempfile.mkdtemp(prefix="dreamlite_mobile_comfy_"))
        out_file = out_dir / "dreamlite_output.png"

        script = r'''
import json
from pathlib import Path
import torch
from PIL import Image
from dreamlite import DreamLiteMobilePipeline

prompt = __PROMPT__
steps = __STEPS__
width = __WIDTH__
height = __HEIGHT__
seed = __SEED__
device = __DEVICE__
dtype = __DTYPE__
memory_mode = __MEMORY_MODE__
model_id = __MODEL_ID__
out_file = Path(__OUT_FILE__)

weight_dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}[dtype]

pipe = DreamLiteMobilePipeline.from_pretrained(model_id, torch_dtype=weight_dtype)
if memory_mode == "cpu":
    pipe.to("cpu")
elif memory_mode == "sequential_cpu_offload":
    pipe.enable_sequential_cpu_offload(device=device)
elif memory_mode == "model_cpu_offload":
    pipe.enable_model_cpu_offload(device=device)
else:
    pipe.to(device)
img = pipe(
    prompt=prompt,
    image=None,
    height=height,
    width=width,
    num_inference_steps=steps,
    generator=torch.Generator("cpu").manual_seed(seed),
).images[0]
if img.size != (width, height):
    img = img.resize((width, height), Image.Resampling.LANCZOS)
img.save(out_file)
print(json.dumps({"out_file": str(out_file)}))
'''

        replacements = {
            "__PROMPT__": json.dumps(prompt),
            "__STEPS__": str(int(steps)),
            "__WIDTH__": str(int(width)),
            "__HEIGHT__": str(int(height)),
            "__SEED__": str(int(seed)),
            "__DEVICE__": json.dumps(selected_device),
            "__DTYPE__": json.dumps(selected_dtype),
            "__MEMORY_MODE__": json.dumps(selected_memory_mode),
            "__MODEL_ID__": json.dumps(str(DREAMLITE_MODEL)),
            "__OUT_FILE__": json.dumps(str(out_file)),
        }
        for k, v in replacements.items():
            script = script.replace(k, v)

        total_steps = max(int(steps), 1)
        pb = comfy.utils.ProgressBar(total_steps) if comfy and hasattr(comfy, "utils") else None

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{DREAMLITE_REPO}:{env.get('PYTHONPATH', '')}"
        tmp_base = str(out_dir.parent)
        env["TMPDIR"] = tmp_base
        env["TEMP"] = tmp_base
        env["TMP"] = tmp_base
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if DREAMLITE_TORCH_CACHE is not None:
            env.setdefault("TORCHINDUCTOR_CACHE_DIR", str(DREAMLITE_TORCH_CACHE))

        proc = subprocess.Popen(
            [str(DREAMLITE_PYTHON), "-c", script],
            cwd=str(DREAMLITE_REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        lines = []
        progress = 0
        last_heartbeat = time.time()
        try:
            while True:
                # Respect Comfy interrupt/cancel.
                model_management.throw_exception_if_processing_interrupted()

                line = ""
                if proc.stdout:
                    ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                    if ready:
                        line = proc.stdout.readline()
                if line:
                    msg = line.rstrip()
                    lines.append(msg)
                    if msg.startswith("__DREAMLITE_STEP__:"):
                        try:
                            current = int(msg.split(":", 1)[1].split("/", 1)[0])
                        except Exception:
                            current = None
                        if pb and current is not None:
                            progress = max(progress, min(current, total_steps))
                            pb.update_absolute(progress, total_steps)
                            last_heartbeat = time.time()
                elif proc.poll() is not None:
                    break
                else:
                    # Heartbeat progress so UI doesn't stay stuck at 0% when subprocess is silent.
                    if pb and progress < max(total_steps - 1, 1):
                        now = time.time()
                        if now - last_heartbeat >= 2.0:
                            progress += 1
                            progress = min(progress, max(total_steps - 1, 1))
                            pb.update_absolute(progress, total_steps)
                            last_heartbeat = now
                    time.sleep(0.1)
        except InterruptProcessingException:
            # Ensure child process is actually terminated on cancel.
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
        if pb and progress < total_steps:
            pb.update_absolute(total_steps, total_steps)

        if rc != 0:
            raise RuntimeError("DreamLite process failed:\n" + "\n".join(lines[-80:]))

        if not out_file.exists():
            raise RuntimeError("DreamLite did not produce output image")

        image_tensor = self._to_tensor(out_file)
        info_lines = []
        if selected_dtype != dtype:
            info_lines.append(f"[DreamLiteNode] Requested dtype '{dtype}' adjusted to '{selected_dtype}' for '{selected_device}'.")
        info_lines.append(f"[DreamLiteNode] Mobile memory mode: {selected_memory_mode} on {selected_device}.")
        log = "\n".join(info_lines + lines[-40:])
        return (image_tensor, log)


NODE_CLASS_MAPPINGS = {
    "DreamLiteMobileGenerate": DreamLiteMobileGenerate,
    "DreamLiteMobileGenerateV2": DreamLiteMobileGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamLiteMobileGenerate": "DreamLite Mobile Generate",
    "DreamLiteMobileGenerateV2": "DreamLite Mobile Generate V2",
}
