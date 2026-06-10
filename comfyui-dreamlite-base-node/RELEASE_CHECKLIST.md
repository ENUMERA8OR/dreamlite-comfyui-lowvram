# Release Checklist

- [ ] Run one 1024x1024 smoke test with `steps=1`, `memory_mode=sequential_cpu_offload`, `dtype=float32`, and `attention_mode=gqa_query_slicing`.
- [ ] Compare `attention_mode=sdpa` versus `attention_mode=gqa_query_slicing`.
- [x] Install into a clean ComfyUI `custom_nodes` directory and confirm the node imports.
