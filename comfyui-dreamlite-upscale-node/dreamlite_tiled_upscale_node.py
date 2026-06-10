import gc

import torch

import comfy.model_management as model_management
import comfy.utils


class DreamLiteTiledUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "tile_size": ("INT", {"default": 256, "min": 64, "max": 1024, "step": 64}),
                "overlap": ("INT", {"default": 32, "min": 0, "max": 256, "step": 16}),
                "output_device": (["cpu", "intermediate"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "upscale"
    CATEGORY = "DreamLite"

    def upscale(self, upscale_model, image, tile_size, overlap, output_device):
        device = model_management.get_torch_device()
        tile = int(tile_size)
        overlap = int(overlap)
        out_device = "cpu" if output_device == "cpu" else model_management.intermediate_device()

        in_img = image.movedim(-1, -3).to(device)
        upscale_model.to(device)

        try:
            while True:
                try:
                    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                        in_img.shape[3],
                        in_img.shape[2],
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                    )
                    pbar = comfy.utils.ProgressBar(steps)
                    upscaled = comfy.utils.tiled_scale(
                        in_img,
                        lambda tile_tensor: upscale_model(tile_tensor.float()),
                        tile_x=tile,
                        tile_y=tile,
                        overlap=overlap,
                        upscale_amount=upscale_model.scale,
                        pbar=pbar,
                        output_device=out_device,
                    )
                    break
                except Exception as exc:
                    model_management.raise_non_oom(exc)
                    tile //= 2
                    if tile < 64:
                        raise exc
                    model_management.soft_empty_cache()
        finally:
            upscale_model.to("cpu")
            del in_img
            gc.collect()
            model_management.soft_empty_cache()

        upscaled = torch.clamp(upscaled.movedim(-3, -1), min=0, max=1.0)
        return (upscaled.to(model_management.intermediate_dtype()),)


NODE_CLASS_MAPPINGS = {
    "DreamLiteTiledUpscaleWithModel": DreamLiteTiledUpscaleWithModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DreamLiteTiledUpscaleWithModel": "DreamLite Tiled Upscale",
}
