# ComfyUI DreamLite Base Node

ComfyUI custom node for local `DreamLite-base` inference.

## What It Adds

- `DreamLite Base Generate`
- `gqa_query_slicing`, a grouped-query-attention-safe query-token slicer
- CPU/GPU offload options exposed in ComfyUI

## Required Environment Variables

Set these before starting ComfyUI:

```bash
export DREAMLITE_PYTHON="/path/to/dreamlite/env/bin/python"
export DREAMLITE_REPO="/path/to/DreamLite"
export DREAMLITE_BASE_MODEL="/path/to/DreamLite-base"
```

Optional:

```bash
export DREAMLITE_TMP="/path/to/writable/tmp"
export DREAMLITE_TORCH_CACHE="/path/to/torch/cache"
export DREAMLITE_OFFLOAD_SCRIPT="/path/to/infer_base_offload.py"
```

If `DREAMLITE_OFFLOAD_SCRIPT` is unset, the node uses the bundled script:

```text
scripts/infer_base_offload.py
```

## Recommended Low-VRAM Settings

```text
memory_mode: sequential_cpu_offload
dtype: float32
attention_mode: gqa_query_slicing
attention_slice_size: 256
steps: 26-30
resolution: 1024x1024
```

## Install

```bash
ln -s /path/to/comfyui-dreamlite-base-node /path/to/ComfyUI/custom_nodes/comfyui-dreamlite-base-node
```

Restart ComfyUI after setting the environment variables.
