# DreamLite Local Image Generation Session Report

Date: 2026-06-04  
Project root: `/mnt/c/Users/Aditya/Videos/Playground/LLMs/Text_to_Image`  
Primary constraint: avoid additional C: drive bloat wherever possible.

## 1. Session Objective

The core goal of this session was to get ByteDance/DreamLite running locally for high-quality private image generation and editing on constrained consumer hardware, then integrate it with ComfyUI while controlling disk usage.

The work evolved into several connected deliverables:

- Analyze and prepare the local WSL/conda/Python environment for DreamLite.
- Run `carlofkl/DreamLite-mobile` locally.
- Preserve C: drive space by moving heavy artifacts to D: where possible.
- Install and launch ComfyUI with a custom DreamLite node.
- Patch the custom node for progress/cancel behavior.
- Enable CUDA in WSL and replace CPU-only PyTorch with CUDA PyTorch in the DreamLite env.
- Compare DreamLite-mobile and DreamLite-base.
- Patch CLI test harnesses for reproducibility.
- Download and verify DreamLite-base weights.
- Establish a practical image generation workflow using manual PowerShell commands.
- Document prompts, seeds, dtype behavior, and quality findings.

## 2. Important Paths

Project root:

```text
/mnt/c/Users/Aditya/Videos/Playground/LLMs/Text_to_Image
```

DreamLite repo:

```text
/root/dreamlite-persist/DreamLite
```

DreamLite Python environment:

```text
/root/dreamlite-persist/env
```

DreamLite-mobile weights:

```text
/mnt/d/AI/dreamlite-models/DreamLite-mobile
/root/dreamlite-persist/DreamLite/models/DreamLite-mobile -> /mnt/d/AI/dreamlite-models/DreamLite-mobile
```

DreamLite-base weights:

```text
/mnt/d/AI/dreamlite-models/DreamLite-base
/root/dreamlite-persist/DreamLite/models/DreamLite-base -> /mnt/d/AI/dreamlite-models/DreamLite-base
```

ComfyUI:

```text
/root/comfy/ComfyUI
```

ComfyUI DreamLite custom node:

```text
/root/comfy/ComfyUI/custom_nodes/dreamlite_mobile_node.py
```

D-drive cache/output locations:

```text
/mnt/d/AI/cache/tmp
/mnt/d/AI/cache/pip
/mnt/d/AI/cache/hf
/mnt/d/AI/cache/torch
/mnt/d/AI/cache/xdg
/mnt/d/AI/output
```

Comfy dependency overlay:

```text
/mnt/d/AI/comfy-pydeps
```

Comfy launch helper:

```text
/mnt/d/AI/bin/start_comfy_dreamlite.sh
```

Publishable ComfyUI node scaffold:

```text
/mnt/c/Users/Aditya/Videos/Playground/LLMs/Text_to_Image/comfyui-dreamlite-mobile-node
```

## 3. Disk Space And C: Drive Constraint

The user raised a strict non-negotiable constraint: avoid further C: drive bloat.

The practical root cause of C: usage was WSL virtual disk growth and local AI setup artifacts:

- Python environments.
- PyTorch and CUDA wheels.
- Hugging Face model weights.
- Hugging Face/Pip caches.
- Generated images and temporary files.
- LM Studio remnants were also investigated separately and cleaned earlier.

To reduce future C: pressure:

- DreamLite weights were placed on D: under `/mnt/d/AI/dreamlite-models`.
- Temp/cache folders were pointed to D: using `TMPDIR`, `TEMP`, `TMP`, `HF_HOME`, and related paths where possible.
- Heavy Comfy dependencies were put under `/mnt/d/AI/comfy-pydeps`.
- Outputs were routed to `/mnt/d/AI/output`.

Important note: WSL itself can still grow because it is backed by a virtual disk on C:. Moving heavy files to `/mnt/d` reduces growth, but installed Linux packages/environments inside `/root` still affect the WSL disk.

## 4. DreamLite-mobile Setup

DreamLite-mobile was successfully downloaded and linked:

```text
/mnt/d/AI/dreamlite-models/DreamLite-mobile
```

It is exposed to DreamLite through:

```text
/root/dreamlite-persist/DreamLite/models/DreamLite-mobile
```

The symlink points to the D-drive model folder. The model size was about `4.8G`.

The working CLI entrypoint is:

```text
/root/dreamlite-persist/DreamLite/infer_mobile.py
```

## 5. DreamLite-base Setup

DreamLite-base was initially missing. It was downloaded from Hugging Face into D: after several interruptions/stalls.

Final verified files:

```text
/mnt/d/AI/dreamlite-models/DreamLite-base/text_encoder/model.safetensors
Size: 4.0G

/mnt/d/AI/dreamlite-models/DreamLite-base/unet/diffusion_pytorch_model.safetensors
Size: 744M
```

Final model directory:

```text
/mnt/d/AI/dreamlite-models/DreamLite-base
Size: 4.8G
```

Final symlink:

```text
/root/dreamlite-persist/DreamLite/models/DreamLite-base -> /mnt/d/AI/dreamlite-models/DreamLite-base
```

DreamLite-base was verified with a smoke test:

```text
Output: /mnt/d/AI/output/base_smoke_test.png
```

Smoke test settings:

```text
model: DreamLite-base
script: infer.py
device: cuda
dtype: float32
steps: 1
size: 512x512
seed: 42
prompt: glass statue isometric view
```

Observed timing:

```text
Initial component load: about 4m12s
1 inference step: about 38s at 512x512 float32
```

Verdict: DreamLite-base is working, but much slower than mobile on this machine.

## 6. Python Environment And CUDA

The DreamLite environment is:

```text
/root/dreamlite-persist/env
```

CUDA was initially unavailable in WSL, causing errors like:

```text
AssertionError: Torch not compiled with CUDA enabled
```

The issue had two layers:

- The WSL instance initially lacked visible `/dev/dxg`.
- The DreamLite environment had CPU-only PyTorch at one point.

After Windows/WSL checks and reboot, the user's PowerShell checks confirmed:

```text
/dev/dxg exists
nvidia-smi sees NVIDIA GeForce GTX 1650 Ti
DreamLite env torch reports CUDA available
```

The DreamLite environment ended up using CUDA PyTorch:

```text
torch 2.11.0+cu128
torchvision 0.26.0+cu128
torchaudio 2.11.0+cu128
CUDA available: True
GPU: NVIDIA GeForce GTX 1650 Ti
```

CPU-only torch was removed from the DreamLite env when CUDA torch was installed.

## 7. ComfyUI Integration

ComfyUI was installed under:

```text
/root/comfy/ComfyUI
```

A custom node was created:

```text
/root/comfy/ComfyUI/custom_nodes/dreamlite_mobile_node.py
```

The node is named:

```text
DreamLiteMobileGenerate
```

The node calls DreamLite through a subprocess using:

```text
/root/dreamlite-persist/env/bin/python
```

The node uses the DreamLite-mobile model path:

```text
/root/dreamlite-persist/DreamLite/models/DreamLite-mobile
```

Node features added during the session:

- Device/dtype handling.
- D-drive temp folder routing.
- Progress heartbeat so ComfyUI does not stay at `0%` forever.
- Cancel handling using Comfy interrupt checks and subprocess group termination.
- Output conversion to ComfyUI `IMAGE` tensor format.

Known caveat:

The node originally checked CUDA availability using Comfy's own torch environment. Since Comfy may run CPU-only while the DreamLite subprocess has CUDA, this can create a false CPU fallback. The correct long-term fix is to validate CUDA inside the DreamLite subprocess environment.

ComfyUI missing optional dependencies were seen:

```text
kornia
spandrel
torchaudio
OpenGL deps
```

These affected some Comfy extras but were not required for the DreamLite custom node workflow.

## 8. Publishable Custom Node Scaffold

A publishable scaffold was created in:

```text
/mnt/c/Users/Aditya/Videos/Playground/LLMs/Text_to_Image/comfyui-dreamlite-mobile-node
```

Files created:

```text
dreamlite_mobile_node.py
README.md
examples/dreamlite_mobile_t2i_workflow.json
CHANGELOG.md
RELEASE_CHECKLIST.md
.gitignore
```

Verdict from the session: the ComfyUI node is publishable as a useful contribution, but it should be cleaned further before public release.

Recommended remaining polish before publishing:

- Configurable model/env paths instead of hardcoded `/root/...`.
- Correct CUDA detection inside DreamLite subprocess.
- Better README installation instructions.
- Example workflows for text-to-image and image edit.
- Versioned release notes.
- Optional progress logging support.

## 9. CLI Harness Patches

Two official scripts were patched for reproducibility and safer outputs.

### `infer_mobile.py`

File:

```text
/root/dreamlite-persist/DreamLite/infer_mobile.py
```

Patches:

- Added `--output` support earlier to avoid filename-too-long failures.
- Added safe output filename generation when no output path is supplied.
- Added `--seed`.
- Replaced hardcoded seed `42` with `args.seed`.

Before the seed patch, every mobile CLI run used:

```python
torch.Generator("cpu").manual_seed(42)
```

After the patch:

```python
torch.Generator("cpu").manual_seed(args.seed)
```

### `infer.py`

File:

```text
/root/dreamlite-persist/DreamLite/infer.py
```

Patches:

- Added `--output`.
- Added safe output filename generation.
- Added `--seed`.
- Replaced hardcoded seed `42` with `args.seed`.

This made DreamLite-base reproducible from PowerShell commands.

## 10. Key Testing Workflow

The user preferred manual PowerShell ISE commands for generation.

The stable command pattern:

```powershell
wsl -d Ubuntu -- bash -lc 'cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_mobile.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-mobile --device cuda --weight_dtype float32 --num_inference_steps 4 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/output.png --prompt "prompt here"'
```

Recommended output directory:

```text
/mnt/d/AI/output
```

Windows path equivalent:

```text
D:\AI\output
```

Important PowerShell quoting lesson:

- Outer PowerShell command can use single quotes around the `bash -lc` payload.
- The prompt inside can use double quotes.
- Avoid mixing unescaped double quotes inside a double-quoted PowerShell command.

## 11. Dtype Findings

`dtype` means data type / numeric precision.

Observed options:

```text
float16
bfloat16
float32
```

Findings:

- `float16` can be faster but produced black images intermittently on this setup.
- `float32` is slower but much more stable.
- For quality reliability on the user's GTX 1650 Ti WSL setup, `float32` became the preferred testing dtype.

Important conclusion:

`float32` does not inherently lower image quality. It usually improves numerical stability. `float16` can match quality when stable, but can catastrophically fail on this stack.

## 12. Hyperparameter Findings

The user shared a UI screenshot containing:

```text
Inference Steps
Guidance Scale
Image Guidance Scale
Seed
Resolution
```

Interpretation:

- `Inference Steps`: number of denoising iterations.
- `Seed`: random starting noise; key reproducibility and composition variable.
- `Guidance Scale`: prompt adherence strength in CFG-style pipelines.
- `Image Guidance Scale`: mainly relevant for image editing / image-to-image.
- `Resolution`: output size; DreamLite-mobile quality was strongest at 1024x1024.

Important implementation detail:

In `DreamLiteMobilePipeline`, `guidance_scale` and `image_guidance_scale` are accepted in the function signature but do not appear to meaningfully control generation in the same way as the base pipeline. For mobile text-to-image, `steps`, `seed`, `resolution`, `dtype`, and prompt formulation mattered more.

## 13. Seed Findings

Seed became a decisive factor.

Why:

- At 4 inference steps, there are very few denoising corrections.
- The initial latent/noise structure has a large effect on final geometry.
- Good seeds quickly align with the prompt.
- Bad seeds can produce weak anatomy, poor composition, or artifacts even with the same prompt.

A key seed used:

```text
2147483647
```

This produced very strong glass statue results and became a "gold seed" for glass figure compositions.

Practical insight:

Prompt and workflow can be copied, but curated seeds become a real creative advantage. Seed exploration is especially powerful for low-step models like DreamLite-mobile.

## 14. Negative Prompt Caveat

DreamLite-mobile CLI does not expose a real `--negative_prompt` argument.

If the user writes this inside the prompt:

```text
--negative low quality, blurry, bad anatomy
```

or:

```text
negative_prompt low quality, blurry
```

DreamLite-mobile treats it as normal prompt text, not as a true negative prompt.

This means negative prompt terms can confuse the model. A cleaner positive prompt often works better.

For DreamLite-base, `infer.py` does include:

```text
--negative_prompt
```

But the mobile pipeline does not use it in the same way.

## 15. Major Successful Prompts And Commands

### Glass Dragon

Prompt:

```text
glass dragon statue isometric view
```

Command pattern:

```powershell
wsl -d Ubuntu -- bash -lc "mkdir -p /mnt/d/AI/output && cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_mobile.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-mobile --device cuda --weight_dtype float32 --num_inference_steps 4 --width 512 --height 512 --output /mnt/d/AI/output/glass_dragon_isometric_512_s4.png --prompt 'glass dragon statue isometric view'"
```

Later quality tests used 1024x1024.

### Ancient Indian Warrior Glass Statue

Prompt:

```text
ancient indian warrior glass texture statute isometric view
```

Command:

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_mobile.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-mobile --device cuda --weight_dtype float32 --num_inference_steps 4 --width 1024 --height 1024 --seed 42 --output /mnt/d/AI/output/warrior_seed42.png --prompt 'ancient indian warrior glass texture statute isometric view'"
```

### Running Glass Horse

Prompt:

```text
hyper-realistic photo-realistic running horse glass transparent texture statute isometric view --negative low quality, worst quality, blurry, bad anatomy, deformed
```

Command:

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_mobile.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-mobile --device cuda --weight_dtype float32 --num_inference_steps 4 --width 1024 --height 1024 --seed 1024534543534 --output /mnt/d/AI/output/horse_seed1024.png --prompt 'hyper-realistic photo-realistic running horse glass transparent texture statute isometric view --negative low quality, worst quality, blurry, bad anatomy, deformed'"
```

Note: The `--negative` text here was not a true negative prompt for mobile. It was part of the literal prompt, but the result was visually strong.

### Magnum Opus Glass Warrior

Prompt:

```text
hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statute isometric view --negative low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers
```

Command:

```powershell
wsl -d Ubuntu -- bash -lc "cd /root/dreamlite-persist/DreamLite && TMPDIR=/mnt/d/AI/cache/tmp TEMP=/mnt/d/AI/cache/tmp TMP=/mnt/d/AI/cache/tmp PYTHONPATH=/root/dreamlite-persist/DreamLite /root/dreamlite-persist/env/bin/python infer_mobile.py --model_id /root/dreamlite-persist/DreamLite/models/DreamLite-mobile --device cuda --weight_dtype float32 --num_inference_steps 4 --width 1024 --height 1024 --seed 2147483647 --output /mnt/d/AI/output/horse_seed1024.png --prompt 'hyper-realistic photo-realistic running ancient Indian warrior glass turquoise tint transparent texture statute isometric view --negative low quality, worst quality, blurry, bad anatomy, deformed, extra fingers, missing fingers'"
```

Cleaned version recommended:

```text
hyper-realistic, photo-realistic isometric view of a running ancient Indian warrior statue made of transparent turquoise glass, crisp anatomy, dynamic motion pose, clean silhouette, studio rim lighting, dark gradient background, high-detail reflections and refractions, no text, no watermark
```

## 16. Prompt Engineering Lessons

The best results came from:

- Strong material words: `transparent glass`, `crystalline`, `turquoise tint`, `refraction`, `rim lighting`.
- Strong composition words: `isometric view`, `clean silhouette`, `full body`, `diorama`.
- Limiting subject complexity when using only 4 steps.
- Using seed search instead of immediately increasing steps.

Complex multi-object action prompts are harder:

```text
ancient Indian warrior on a running horse shooting an arrow
```

This combines rider, horse, weapon, action pose, glass material, and isometric composition. At 4 steps this is much harder than a single statue subject.

Recommended approach:

- First generate a strong single-subject glass statue.
- Then add one complexity at a time.
- Use fixed prompt + seed sweeps.
- Try 8 steps only if 4 steps cannot resolve composition.

## 17. DreamLite-mobile Quality Verdict

The session showed DreamLite-mobile can produce extremely strong images at 4 steps when the workflow is controlled.

Key quality factors:

- Use `1024x1024`.
- Use `float32` for stability on this setup.
- Use `4` steps for the intended mobile regime.
- Use careful prompt wording.
- Explore seed space.
- Keep the same harness for comparisons.

Fair assessment:

DreamLite-mobile is not universally equal to Flux-family models across all prompts, but it can produce Flux-like outputs for selected subjects and strong seeds. For a 0.39B class model, the quality-per-compute is exceptional.

## 18. DreamLite-base Quality Expectation

Expected improvements over mobile:

- Better fine-detail consistency.
- Better prompt adherence on complex prompts.
- Better robustness for anatomy and multi-object scenes.
- More effective use of guidance scale and negative prompt.

Tradeoff:

DreamLite-base is much slower. The smoke test showed 1 step at 512x512 float32 took about 38 seconds after a 4+ minute load.

## 19. Model Alternatives Discussed

Other Hugging Face models considered:

```text
OFA-Sys/small-stable-diffusion-v0
segmind/SSD-1B
nicolas-dufour/miro
second-state/stable-diffusion-3.5-medium-GGUF
Amshaker/Mobile-O-0.5B
Sana 600M
PixArt-Sigma
SDXL-Lightning
Playground v2.5
PixelDiT
```

Practical ranking for this hardware at the time:

- DreamLite-mobile/base remained the most practical current path.
- `miro` was interesting because it is small and flow-based, but real footprint includes other components.
- `SSD-1B` was a stronger quality candidate but heavier.
- `SD3.5-medium-GGUF` and `Mobile-O` were not prioritized due to footprint/complexity.
- PixelDiT was interesting but likely too heavy for this setup as a daily driver.

## 20. Research And Future Direction Notes

Several conceptual directions were discussed:

- Sub-1B models reaching Flux-like quality.
- Hybrid diffusion + flow + GAN objectives.
- Sparse conditional compute / prompt-token-to-active-parameter efficiency.
- Seed search as a latent prior optimization problem.
- Chinese labs moving fast in efficient local image models.
- Unified models that combine generation, editing, and understanding.

Core thesis from the session:

Quality on constrained hardware will come from efficient architectures, distillation, better latent/token representations, sparse routing, better schedulers, and practical seed/prompt search, not only bigger parameter counts.

## 21. PowerShell History Lesson

The user wanted to recover previously entered PowerShell ISE commands.

Findings:

- `Get-History` works only for the current session.
- Classic PowerShell ISE often does not persist PSReadLine history.
- If the ISE window was closed and no transcript was active, exact old commands are likely gone.

Useful commands:

```powershell
Get-History
Get-History | Where-Object { $_.CommandLine -match "infer_mobile|DreamLite|--prompt|wsl -d Ubuntu" } | Select-Object Id,CommandLine
```

Transcript logging for future sessions:

```powershell
Start-Transcript -Path "D:\AI\output\dreamlite_powershell_log.txt" -Append
Stop-Transcript
```

Recommended future improvement:

Patch `infer_mobile.py` and `infer.py` to write a JSON sidecar next to every output image containing:

```text
prompt
seed
steps
width
height
dtype
device
model_id
output path
timestamp
```

This would permanently solve prompt/command loss.

## 22. VS Code Settings Side Quest

The user wanted VS Code theme and font setup.

Changes discussed:

- Theme: `Bearded Theme: Coffee`
- Font: `Inconsolata`

The user installed Inconsolata on Windows and confirmed it appeared in settings. VS Code theme was implemented successfully, and font was confirmed visually by the user.

## 23. Current Known State

As of this report:

- DreamLite-mobile is installed and working.
- DreamLite-base is installed, linked, and smoke-tested.
- DreamLite env has CUDA-enabled PyTorch.
- Manual PowerShell command generation works.
- Best practical mobile recipe is:

```text
DreamLite-mobile
1024x1024
4 steps
cuda
float32
seed search
D-drive output/cache paths
```

- ComfyUI integration exists but should be cleaned before publishing.
- The publishable node scaffold exists in the project root.

## 24. Recommended Next Steps

1. Add JSON sidecar logging to `infer_mobile.py` and `infer.py`.
2. Create a seed-sweep helper script that runs multiple seeds and names outputs predictably.
3. Run DreamLite-base quality tests on the same glass-statue prompts.
4. Compare mobile vs base using the same prompt, seed, resolution, dtype, and output naming.
5. Clean the ComfyUI node for public release.
6. Add ESRGAN/RealESRGAN upscaling as a post-process stage, with weights stored on D:.
