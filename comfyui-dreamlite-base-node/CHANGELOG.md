# Changelog

## 0.3.0

- Added `gqa_query_slicing`, a DreamLite-specific attention slicing mode for grouped-query attention.
- Replaced the Comfy node's `attention_slicing` boolean with `attention_mode` and `attention_slice_size`.
- Updated the node to call `infer_base_offload.py` directly.

## 0.2.0

- Added CPU/GPU offload modes.

## 0.1.0

- Initial DreamLite-base node scaffold.
