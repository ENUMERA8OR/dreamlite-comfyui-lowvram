# ComfyUI DreamLite Mobile Node

ComfyUI custom node for local `DreamLite-mobile` inference.

## What It Adds

- `DreamLite Mobile Generate`

Inputs:

- `prompt`
- `steps`
- `width`
- `height`
- `seed`
- `dtype`

Outputs:

- `image`
- `log`

## Required Environment Variables

Set these before starting ComfyUI:

```bash
export DREAMLITE_PYTHON="/path/to/dreamlite/env/bin/python"
export DREAMLITE_REPO="/path/to/DreamLite"
export DREAMLITE_MOBILE_MODEL="/path/to/DreamLite-mobile"
```

Optional:

```bash
export DREAMLITE_TMP="/path/to/writable/tmp"
export DREAMLITE_TORCH_CACHE="/path/to/torch/cache"
export DREAMLITE_MOBILE_DEVICE="cuda"
export DREAMLITE_MOBILE_MEMORY_MODE="sequential_cpu_offload"
```

## Install

```bash
ln -s /path/to/comfyui-dreamlite-mobile-node /path/to/ComfyUI/custom_nodes/comfyui-dreamlite-mobile-node
```

Restart ComfyUI after setting the environment variables.

## Recommended Starting Settings

```text
steps: 4
resolution: 1024x1024
dtype: bfloat16 or float32
```

## Attribution

- DreamLite model/research: ByteDance / DreamLite authors.
- ComfyUI: https://github.com/comfyanonymous/ComfyUI
