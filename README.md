# DreamEdit3D: Personalization of Multi-View Diffusion Models for 3D Editing

**Accepted to ECCV 2026 🎉**

**Jinxin Ai · Matthias Nießner · Ziya Erkoç**
Technical University of Munich

[Project page](https://www.jinxinai.org/dreamedit3d/) &nbsp;·&nbsp; [Video](https://youtu.be/PHyvbyREIOw)

![DreamEdit3D teaser](assets/teaser.jpg)

Official implementation of the ECCV 2026 paper.

DreamEdit3D produces multi-view-consistent edits of 3D objects guided by
natural language. We personalize a multi-view diffusion model to preserve
input identity, then generate diverse edits by composing learned token
embeddings with editing prompts.

Pipeline: SAM-based per-view object segmentation → multi-view textual
inversion + fine-tuning of MVDream (Break-A-Scene-style training) →
multi-view image generation → GTR lifts the views to a textured 3D mesh.

The end-to-end CLI takes a 4-view segmented example folder and a text
prompt, and produces a textured 3D mesh:

1. Train a textual token (`<asset0>`) that represents the input object.
2. Sample a multi-view-consistent image under the new edit prompt.
3. Lift the multi-view image to a textured 3D mesh via GTR.

## Repository layout

```
main.py                       End-to-end CLI (the only root .py)
scripts/
├── train.py                  Multi-view textual-inversion training
└── inference.py              Multi-view image sampler from trained MVDream
utils/
└── ptp_utils.py              Image-grid + attention-store helpers used during training
mvdream/                      MVDream multi-view diffusion (vendored)
snap_gtr/                     GTR image-to-3D (git submodule of
                              ASH30KW/snap_gtr@dreamedit3d, our fork with
                              transparent/RGBA rendering support)
segment-anything/             SAM (git submodule; only needed to make masks for new objects)
examples/                     Per-example inputs + (gitignored) runtime outputs
```

## Installation

Requires CUDA-capable GPU (tested on 48GB VRAM) and Linux.
Tested on Linux with Python 3.10 and CUDA 12.8.

Clone with submodules so `snap_gtr/` (required) and `segment-anything/`
(optional, see below) are populated:

```bash
git clone --recurse-submodules https://github.com/ASH30KW/DreamEdit3D.git
# or, on an existing clone:
git submodule update --init --recursive
```

### conda + pip

```bash
conda create -n DreamEdit3D python=3.10 -y
conda activate DreamEdit3D

# 1. PyTorch first, so torch-extension builds can find it later
pip install torch==2.8.0 torchvision==0.23.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 2. Everything else
pip install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# 3. Torch-extension packages (require nvcc 12.x and torch already
#    installed; install the CUDA toolkit first if you don't have it)
conda install -c nvidia cuda-toolkit=12.8 -y
pip install --no-build-isolation diso==0.1.4
pip install --no-build-isolation \
    "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"
pip install --no-build-isolation \
    "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@V0.7.7"
```

`requirements.txt` is a frozen snapshot of the working environment.
Adjust the `+cu128` markers and `--extra-index-url` if you target a
different CUDA version.

## Required model weights

The repository ships **source only**. One checkpoint must be downloaded
by hand; the rest are fetched from HuggingFace on first run.

| Weight | How to get it |
| --- | --- |
| GTR `full_checkpoint.pth` | Download from the [GTR release](https://github.com/snap-research/snap_gtr) and place it at `snap_gtr/ckpts/full_checkpoint.pth` |
| MVDream `sd-v2.1-base-4view.pt` | Auto-downloaded from [`MVDream/MVDream`](https://huggingface.co/MVDream/MVDream) into the HF cache |
| Stable Diffusion 2.1 base | Auto-downloaded from the public mirror `sd-research/stable-diffusion-2-1-base` (Stability AI removed the original `stabilityai/stable-diffusion-2-1-base` repo). Pass a local pipeline dump via `--pretrained_model_name_or_path` if you have one |

The end-to-end CLI never loads SAM. The `segment-anything/` submodule
and its [ViT-H checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
are only needed if you want to produce `mask0.png` files for your own
objects; any other segmentation tool that emits a binary PNG works too.

## Usage

`main.py` runs the full pipeline end-to-end. The exact command verified
end-to-end on `examples/character` (RTX 3090, ~5 min, identity-preserving
smile-with-teeth edit, textured GLB + OBJ mesh):

```bash
python main.py \
  --example_dir examples/character \
  --prompt "a photo of <asset0> smile with teeth" \
  --initializer_token person
```

### Bundled examples

Four ready-to-run inputs ship under `examples/`. Each one is a 4-view
object with a SAM mask per view (`01_sam_masks/view_{1..4}/{img.jpg,mask0.png}`)
plus a `meta.json` recording the original edit:

| `--example_dir` | `--initializer_token` | `--prompt` |
| --- | --- | --- |
| `examples/character` | `person` | `"a photo of <asset0> smile with teeth"` |
| `examples/dog` | `dog` | `"a photo of <asset0> smile"` |
| `examples/sofa` | `sofa` | `"a photo of <asset0> redesigned to single seat"` |
| `examples/van` | `van` | `"a photo of <asset0> in red"` |

e.g.

```bash
python main.py --example_dir examples/van --initializer_token van \
  --prompt "a photo of <asset0> in red"
```

### Your own object

The example dir must contain `01_sam_masks/view_1/img.jpg` + `mask0.png`,
and one such pair per view (`view_1`, `view_2`, …). Images are 512×512
RGB; masks are single-channel PNGs of the same size. See any of the
bundled examples for a reference. Outputs land in the same dir:

```
examples/character/
├── 01_sam_masks/                  (input, tracked in git)
├── meta.json                      (input, tracked)
├── 02_train/mvdream_model.pth     ← Stage 1 output
├── 03_multiview_images/result.jpg ← Stage 2 output
├── 04_gtr_prepared/rgb_*.png      ← Stage 3 intermediate
└── 04_gtr_3d/mesh.{glb,obj}       ← Stage 3 final output
```

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--initializer_token` | `object` | word loosely describing the object (used to init `<asset0>`) |
| `--phase1_steps` / `--phase2_steps` | 400 / 400 | training steps per phase (use 50/50 for a quick smoke test) |
| `--resolution` / `--size` | 256 / 256 | training resolution |
| `--num_frames` | 4 | number of views (set to match your `01_sam_masks/view_*` count) |
| `--seed` | 23 | RNG seed for training + inference |
| `--pretrained_model_name_or_path` | `sd-research/stable-diffusion-2-1-base` | HF model id or local path (see [Required model weights](#required-model-weights)) |
| `--skip_train` / `--skip_inference` / `--skip_gtr` | off | re-use an earlier stage's output |

### Notes on the GTR stage

- `main.py` compiles nvdiffrast's CUDA extension for the GPU it finds
  (`torch.cuda.get_device_capability()`), so it works on Ampere, Ada
  and Hopper cards alike.
- When run inside a conda env, `main.py` applies a few small fixes so
  that JIT build can find the conda CUDA toolkit (`CUDA_HOME`, an
  `nvvm` symlink, a dangling `libcudart.so` link, the conda gcc). Each
  fix is printed as it is applied. Set `DREAMEDIT3D_NO_ENV_FIX=1` to
  skip them and manage the toolchain yourself.

## Acknowledgements

DreamEdit3D builds on, and includes source from, these projects:

- [Break-A-Scene](https://github.com/google/break-a-scene) (Avrahami et al., SIGGRAPH Asia 2023) — concept extraction
- [MVDream](https://github.com/bytedance/MVDream) — multi-view diffusion
- [GTR / snap_gtr](https://github.com/snap-research/snap_gtr) — image-to-3D (git submodule of [our fork](https://github.com/ASH30KW/snap_gtr/tree/dreamedit3d) with transparent/RGBA rendering)
- [Segment Anything](https://github.com/facebookresearch/segment-anything) — masking (git submodule)

Please cite the underlying papers when using this code.

## Citation

If you use DreamEdit3D in your work, please cite it as:

```bibtex
@inproceedings{ai2026dreamedit3d,
  title     = {DreamEdit3D: Personalization of Multi-View
               Diffusion Models for 3D Editing},
  author    = {Ai, Jinxin and Nie{\ss}ner, Matthias and Erko\c{c}, Ziya},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE). Vendored third-party code retains
its original license: `mvdream/` is MIT (ByteDance, see
[`mvdream/LICENSE`](mvdream/LICENSE)), `scripts/train.py` derives from
Break-A-Scene (Apache 2.0, Google). The `snap_gtr/` submodule is under
the **Snap Inc. Non-Commercial License**, so the 3D-lifting stage is for
non-commercial research use only.
