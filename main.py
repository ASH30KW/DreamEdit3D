#!/usr/bin/env python3
"""DreamEdit3D end-to-end CLI.

Given a 4-view segmented example folder and a text-edit prompt, runs the
three stages of the pipeline and produces a textured 3D mesh:

    scripts/train.py                  -> 02_train/mvdream_model.pth
    scripts/inference.py              -> 03_multiview_images/result.jpg
    snap_gtr/scripts/prepare_mv.py    -> 04_gtr_prepared/rgb_*.png
    snap_gtr/scripts/inference.py     -> 04_gtr_3d/mesh.{glb,obj}

Usage:
    python main.py --example_dir examples/character \
                   --prompt "a photo of <asset0> smile with teeth"

The example dir must contain `01_sam_masks/view_{1..N}/img.jpg` plus the
corresponding mask files (see examples/character/ for a reference).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent
TRAIN = REPO / "scripts" / "train.py"
INFER = REPO / "scripts" / "inference.py"
SNAP_GTR = REPO / "snap_gtr"
PREPARE_MV = SNAP_GTR / "scripts" / "prepare_mv.py"
GTR_INFER = SNAP_GTR / "scripts" / "inference.py"
GTR_CKPT = SNAP_GTR / "ckpts" / "full_checkpoint.pth"


def run(cmd, cwd=None, env_extra=None, label=""):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    if label:
        print(f"\n=== {label} ===")
    print(">>>", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        sys.exit(f"❌ Failed (exit {r.returncode}): {' '.join(str(c) for c in cmd[:3])}")


def cuda_arch_list():
    """TORCH_CUDA_ARCH_LIST for the GPU nvdiffrast will be JIT-built for."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return os.environ["TORCH_CUDA_ARCH_LIST"]
    try:
        import torch
        major, minor = torch.cuda.get_device_capability()
        return f"{major}.{minor}"
    except Exception:
        return None  # let torch's extension builder pick


def fix_gtr_env():
    """Return env overrides so nvdiffrast can JIT-compile its CUDA extension.

    Inside a conda env with the `cuda-toolkit` package this also repairs
    a few known layout quirks (each fix is printed as it is applied):
      - CUDA_HOME pointing at a non-existent path
      - targets/x86_64-linux/nvvm missing (cicc lives at <env>/nvvm)
      - lib/libcudart.so symlink dangling to an old version
    Set DREAMEDIT3D_NO_ENV_FIX=1 to skip all of this.
    """
    overrides = {}
    arch = cuda_arch_list()
    if arch:
        overrides["TORCH_CUDA_ARCH_LIST"] = arch

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if os.environ.get("DREAMEDIT3D_NO_ENV_FIX") or not conda_prefix \
            or not Path(conda_prefix).is_dir():
        return overrides  # not in a conda env (or opted out); user is on their own
    env_dir = Path(conda_prefix)
    overrides["CUDA_HOME"] = str(env_dir)

    # nvvm/cicc layout fix
    target_nvvm = env_dir / "targets" / "x86_64-linux" / "nvvm"
    if not target_nvvm.exists() and (env_dir / "nvvm" / "bin" / "cicc").exists():
        print(f"[env fix] symlink {target_nvvm} -> {env_dir / 'nvvm'}")
        target_nvvm.parent.mkdir(parents=True, exist_ok=True)
        target_nvvm.symlink_to(env_dir / "nvvm")

    # libcudart.so dangling-symlink fix
    libcudart = env_dir / "lib" / "libcudart.so"
    if libcudart.is_symlink() and not libcudart.resolve().exists():
        target = env_dir / "lib" / "libcudart.so.12"
        if target.exists():
            print(f"[env fix] repoint dangling {libcudart} -> libcudart.so.12")
            libcudart.unlink()
            libcudart.symlink_to("libcudart.so.12")

    # Make <env>/targets/x86_64-linux/include visible (carries <nv/target>)
    include_dir = env_dir / "targets" / "x86_64-linux" / "include"
    if include_dir.is_dir():
        sep = os.pathsep
        overrides["CPATH"] = str(include_dir) + sep + os.environ.get("CPATH", "")

    # Force the conda gcc (matches CUDA 12.x); avoids the gcc-11 bf16 build error
    cc = env_dir / "bin" / "x86_64-conda-linux-gnu-gcc"
    cxx = env_dir / "bin" / "x86_64-conda-linux-gnu-g++"
    if cc.exists() and cxx.exists():
        overrides["CC"] = str(cc)
        overrides["CXX"] = str(cxx)

    return overrides


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--example_dir", required=True, type=Path,
                   help="path to examples/<name>/ (must contain 01_sam_masks/)")
    p.add_argument("--prompt", required=True,
                   help='text-edit prompt with <asset0>, e.g. "a photo of <asset0> smile with teeth"')
    p.add_argument("--initializer_token", default="object",
                   help="word loosely describing the object (initialiser for <asset0>)")
    p.add_argument("--phase1_steps", type=int, default=400)
    p.add_argument("--phase2_steps", type=int, default=400)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--pretrained_model_name_or_path",
                   default="sd-research/stable-diffusion-2-1-base")
    p.add_argument("--skip_train", action="store_true",
                   help="reuse existing 02_train/")
    p.add_argument("--skip_inference", action="store_true",
                   help="reuse existing 03_multiview_images/result.jpg")
    p.add_argument("--skip_gtr", action="store_true",
                   help="don't lift to 3D")
    args = p.parse_args()

    example = args.example_dir.resolve()
    masks = example / "01_sam_masks"
    if not masks.is_dir():
        sys.exit(f"❌ Input not found: {masks}")

    train_dir = example / "02_train"
    mv_dir = example / "03_multiview_images"
    mv_img = mv_dir / "result.jpg"
    gtr_prep = example / "04_gtr_prepared"
    gtr_out = example / "04_gtr_3d"

    # 1. Train per-object token <asset0>
    if not args.skip_train:
        if train_dir.exists():
            shutil.rmtree(train_dir)
        run([
            sys.executable, str(TRAIN),
            "--pretrained_model_name_or_path", args.pretrained_model_name_or_path,
            "--instance_data_dir", str(masks),
            "--num_of_assets", "1",
            "--initializer_tokens", args.initializer_token,
            "--phase1_train_steps", str(args.phase1_steps),
            "--phase2_train_steps", str(args.phase2_steps),
            "--output_dir", str(train_dir),
            "--no_prior_preservation",
            "--use_8bit_adam",
            "--set_grads_to_none",
            "--resolution", str(args.resolution),
            "--size", str(args.size),
            "--num_frames", str(args.num_frames),
            "--mvdream_training_mode", "2d",
            "--train_batch_size", "1",
            "--seed", str(args.seed),
        ],
            label=f"Stage 1: train <asset0> on {masks.relative_to(REPO)}",
            env_extra={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # 2. Sample multi-view image from the trained model
    if not args.skip_inference:
        mv_dir.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, str(INFER),
            "--model_path", str(train_dir),
            "--prompt", args.prompt,
            "--output_path", str(mv_img),
            "--num_frames", str(args.num_frames),
            "--size", str(args.size),
            "--steps", "50",
            "--scale", "7.5",
            "--seed", str(args.seed),
        ], label=f"Stage 2: sample 4-view image with prompt={args.prompt!r}")

    # 3. Lift to a textured 3D mesh
    if not args.skip_gtr:
        if not GTR_CKPT.exists():
            sys.exit(f"❌ GTR checkpoint missing: {GTR_CKPT}")
        gtr_overrides = fix_gtr_env()
        run([
            sys.executable, "scripts/prepare_mv.py",
            "--in_dir", str(mv_img),
            "--out_dir", str(gtr_prep),
            "--elevation", "15",
            "--azimuth_list", "90,180,270,0",
        ], cwd=str(SNAP_GTR), label="Stage 3a: prepare 4 views + cameras for GTR")

        run([
            sys.executable, "scripts/inference.py",
            "--ckpt_path", str(GTR_CKPT),
            "--in_dir", str(gtr_prep),
            "--out_dir", str(gtr_out),
            "--seed", "2025",
            "--export_glb",
            "--skip_mesh_gif",
        ], cwd=str(SNAP_GTR),
            env_extra=gtr_overrides,
            label="Stage 3b: GTR mesh extraction",
        )

    print("\n✅ Done.")
    if not args.skip_inference:
        print(f"   multi-view: {mv_img}")
    if not args.skip_gtr:
        print(f"   3D mesh:    {gtr_out / 'mesh.glb'}")
        print(f"               {gtr_out / 'mesh.obj'}")


if __name__ == "__main__":
    main()
