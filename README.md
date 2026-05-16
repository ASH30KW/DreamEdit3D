# DreamEdit3D

Text-driven editing of 3D assets from a single image. DreamEdit3D combines
single-image concept extraction (Break-A-Scene), multi-view diffusion
(MVDream), single-image-to-3D reconstruction (GTR), and SAM-based masking
into one Gradio pipeline.

The end-to-end app lets you:

1. Upload an image and segment it into concepts with SAM (optionally
   auto-named with GPT-4V).
2. Train per-concept textual tokens (`<asset0>`, `<asset1>`, ...).
3. Re-render the concepts under new prompts as multi-view images via
   MVDream, then lift them to a 3D mesh with GTR.
4. Optionally upscale the renders with Real-ESRGAN / SwinIR.

## Repository layout

```
dreamedit3d_app.py        Main Gradio application (entry point)
dreamedit3d.py            Core training logic (concept extraction)
dreamedit3d_grad_accu.py  Variant with gradient accumulation
gradio_app.py             Gradio UI components used by the main app
train.py / inference.py   CLI training & inference (Break-A-Scene style)
integrated_pipeline.py    CLI end-to-end pipeline
mvdream/                  MVDream multi-view diffusion
snap_gtr/                 GTR image-to-3D reconstruction
segment-anything/         SAM (vendored)
enhence_image/            Real-ESRGAN and SwinIR upscalers (vendored)
mask/                     SAM-based masking helpers
utils/                    Shared utilities (incl. GPT-4V auto-naming)
scripts/                  Helper scripts
examples/                 Example inputs
docs/                     Developer notes and migration history
```

## Installation

Requires CUDA-capable GPU (tested on 48GB VRAM) and Linux.

### Option A — conda + pip (recommended)

```bash
conda create -n DreamEdit3D python=3.10 -y
conda activate DreamEdit3D
pip install -r requirements.txt
```

`requirements.txt` is a frozen snapshot of the working environment.
PyTorch is pinned to a CUDA 12.8 build; adjust the torch line for your
CUDA version if needed.

### Option B — clone an existing conda env

If you have the original `sam-bas-gtr` env on the same machine:

```bash
conda create --clone sam-bas-gtr --name DreamEdit3D
```

## Required model weights

The repository ships **source only** — download these checkpoints
yourself and place them as shown:

| File | Location | Source |
| --- | --- | --- |
| `sd-v2.1-base-4view.pt` | `models/` | [MVDream release](https://github.com/bytedance/MVDream) |
| `sam_vit_h_4b8939.pth` | `mask/checkpoints/` and `segment-anything/checkpoint/` | [SAM ViT-H](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| `sam_vit_l_0b3195.pth` *(optional)* | `mask/checkpoints/` | [SAM ViT-L](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| `sam_vit_b_01ec64.pth` *(optional)* | `mask/checkpoints/` | [SAM ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| `full_checkpoint.pth` (GTR) | `snap_gtr/ckpts/` | [GTR release](https://github.com/snap-research/GTR) |
| `RealESRGAN_x2plus.pth`, `RealESRGAN_x4plus.pth` | `enhence_image/Real-ESRGAN/weights/` | [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) |
| SwinIR model zoo *(optional)* | `enhence_image/SwinIR/model_zoo/swinir/` | [SwinIR releases](https://github.com/JingyunLiang/SwinIR/releases) |

## Configuration

The Gradio app uses GPT-4V for automatic concept naming. Set your key
before launching:

```bash
export OPENAI_API_KEY="sk-..."
# or copy .env.example to .env and edit
```

If the key is not set, auto-naming is disabled but the rest of the
pipeline still works.

## Usage

### Launch the app

```bash
./start_app.sh
# or, equivalently:
python dreamedit3d_app.py
```

Then open the URL Gradio prints (default `http://localhost:7860`).

### CLI: train a single concept

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python dreamedit3d.py \
  --instance_data_dir examples/race-chicken \
  --num_of_assets 1 \
  --initializer_tokens head \
  --class_data_dir inputs/data_dir \
  --phase1_train_steps 400 \
  --phase2_train_steps 400 \
  --output_dir outputs/race-chicken \
  --use_8bit_adam \
  --set_grads_to_none \
  --resolution 128 \
  --size 64 \
  --num_frames 1
```

### CLI: inference

```bash
python inference.py \
  --model_path outputs/race-chicken \
  --prompt "a photo of <asset0>" \
  --output_path outputs/result.jpg
```

See `docs/` for joint multi-view training, the project-based folder
layout, and the GPT-4V auto-naming details.

## Acknowledgements

DreamEdit3D builds on, and vendors source from, these projects:

- [Break-A-Scene](https://github.com/google/break-a-scene) (Avrahami et al., SIGGRAPH Asia 2023) — concept extraction
- [MVDream](https://github.com/bytedance/MVDream) — multi-view diffusion
- [GTR / snap_gtr](https://github.com/snap-research/GTR) — image-to-3D
- [Segment Anything](https://github.com/facebookresearch/segment-anything) — masking
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [SwinIR](https://github.com/JingyunLiang/SwinIR) — upscaling

Please cite the underlying papers when using this code.

## License

Apache 2.0 — see [LICENSE](LICENSE). Vendored third-party code retains
its original license; see each subdirectory.
