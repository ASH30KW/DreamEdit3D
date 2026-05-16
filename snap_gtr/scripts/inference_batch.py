#!/usr/bin/env python3
"""
Batch inference script - loads model ONCE and processes multiple inputs.
Usage: python inference_batch.py --ckpt_path ... --batch_file /path/to/batch.txt

batch.txt format (one per line):
in_dir1,out_dir1
in_dir2,out_dir2
...
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from loguru import logger
import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler

from utils.io_utils import read_yaml, EasyDict
from utils.deterministic import seed_everything
from utils.render_utils import get_cameras, NVDiffRasterizerContext, np_fov_to_intrinsic, invert_transform
from utils.gif_utils import save_images_to_gif
from builders.build_system import build_system

# Import the data loading function from original inference
from inference import load_eval_data, render_mesh_to_gif


@click.command()
@click.option("--ckpt_path", type=str, required=True, help="Path to checkpoint")
@click.option("--batch_file", type=str, required=True, help="File with in_dir,out_dir pairs (one per line)")
@click.option("--seed", type=int, default=2025)
@click.option("--render_mesh_res", type=int, default=512)
@click.option("--render_nerf_res", type=int, default=1024)
@click.option("--skip_mesh_export", is_flag=True, default=False)
@click.option("--skip_mesh_gif", is_flag=True, default=False)
@click.option("--render_num_frames", type=int, default=50)
@click.option("--render_fov", type=float, default=50)
@click.option("--render_cam_distance", type=float, default=3.5)
@click.option("--render_elevation", type=float, default=20)
@click.option("--export_glb", is_flag=True, default=False)
def main(**kwargs):
    opt = EasyDict(kwargs)

    # Read batch file
    batch_file = Path(opt.batch_file)
    if not batch_file.exists():
        print(f"❌ Batch file not found: {batch_file}")
        return

    pairs = []
    with open(batch_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ',' in line:
                in_dir, out_dir = line.split(',', 1)
                pairs.append((in_dir.strip(), out_dir.strip()))

    if not pairs:
        print("❌ No valid pairs found in batch file")
        return

    print(f"{'='*60}")
    print(f"🚀 BATCH MODE: {len(pairs)} items to process")
    print(f"{'='*60}")

    # Setup
    seed_everything(opt.seed)
    device = torch.device("cuda:0")

    config_file = "configs/config_texrefine.yaml"
    job_description = read_yaml(config_file)
    config = job_description["jobs"][0]
    config = EasyDict(config)

    # Load model ONCE
    print("\n📦 Loading model (this happens ONCE)...")
    ckpt_path = Path(opt.ckpt_path)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"
    state_dict = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    model = build_system(config, device=device, world_size=1)
    model.load_state_dict(state_dict['pipeline'], strict=True)
    model.switch_eval()
    print("✅ Model loaded!\n")

    # Setup cameras (same for all)
    num_frames = opt.render_num_frames
    fov_deg = opt.render_fov
    cam_distance = opt.render_cam_distance
    elevation_val = opt.render_elevation

    azimuth_deg = torch.from_numpy(np.linspace(0, 360, num=num_frames, endpoint=False, dtype=np.float32))
    elevation_deg = torch.tensor([elevation_val] * num_frames).float()

    cameras_render = get_cameras(
        azimuth_deg, elevation_deg,
        width=opt.render_mesh_res, height=opt.render_mesh_res,
        fov=fov_deg, camera_distance=cam_distance,
    )

    K = np_fov_to_intrinsic(fov_deg, opt.render_nerf_res, opt.render_nerf_res)
    camera_intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)
    camera_intrinsics = torch.from_numpy(camera_intrinsics).to(device).unsqueeze(0).expand(num_frames, -1)

    trans_cv2blender = torch.eye(4)
    trans_cv2blender[1, 1] = -1
    trans_cv2blender[2, 2] = -1
    trans_blender2cv = trans_cv2blender.T
    w2blender_cam = cameras_render['w2c']
    w2cv_cam = trans_blender2cv[None] @ w2blender_cam
    camera_poses = invert_transform(w2cv_cam)

    dr_ctx = NVDiffRasterizerContext(device=device)

    # Process each item
    success_count = 0
    for idx, (in_dir, out_dir) in enumerate(pairs):
        print(f"\n{'='*60}")
        print(f"[{idx+1}/{len(pairs)}] Processing: {Path(in_dir).name}")
        print(f"{'='*60}")

        try:
            # Load data for this item
            data_batch = load_eval_data(in_dir)

            data_batch["poses"] = camera_poses
            data_batch["intrinsics"] = camera_intrinsics
            data_batch["imgs"] = torch.zeros((num_frames, opt.render_nerf_res, opt.render_nerf_res, 3), dtype=torch.float32)
            data_batch["depths"] = torch.zeros((num_frames, opt.render_nerf_res, opt.render_nerf_res, 1), dtype=torch.float32)
            data_batch["mvps"] = cameras_render['mvp_mtx']
            data_batch["m2vs"] = cameras_render['w2c']

            param_groups = model.get_param_groups()
            from engine.optimizers import Optimizers
            optimizers = Optimizers(config["optimizers"], param_groups)

            for key, value in data_batch.items():
                if type(value) is torch.Tensor:
                    data_batch[key] = value.unsqueeze(0).cuda(device, non_blocking=True)

            Path(out_dir).mkdir(exist_ok=True, parents=True)

            # Refine texture
            grad_scaler = GradScaler(enabled=True, init_scale=2048)
            grad_scaler.load_state_dict(state_dict["scalers"])
            model.load_state_dict(state_dict['pipeline'], strict=True)

            code, loss = model.refine_texture(data_batch, optimizers, grad_scaler, iters=50, learn_code=True)
            print(f"  Texture refined, loss: {loss:.4f}")

            # NeRF rendering
            outputs = model.inference_with_code(data_batch, code)[0]
            rgbs = outputs['rgb'][0]
            rgbs = (rgbs.cpu().numpy() * 255.0).astype(np.uint8)[:, ::-1, :, :]

            nerf_gif_file = f"{out_dir}/nerf.gif"
            save_images_to_gif(rgbs, output_file=nerf_gif_file)
            print(f"  ✅ Saved nerf.gif")

            # Mesh extraction
            if not opt.skip_mesh_export:
                mesh_file_obj = f"{out_dir}/mesh.obj"
                mesh_list = model.extract_geometry(data_batch, resolution=512, level=10, code=code)
                mesh_list[0].export(mesh_file_obj)
                print(f"  ✅ Saved mesh.obj")

                if opt.export_glb:
                    mesh_file_glb = f"{out_dir}/mesh.glb"
                    mesh_list[0].export(mesh_file_glb)
                    print(f"  ✅ Saved mesh.glb")

                if not opt.skip_mesh_gif:
                    mesh_gif_file = f"{out_dir}/mesh.gif"
                    render_mesh_to_gif(mesh_gif_file, mesh_file_obj, cameras_render, dr_ctx, device=device)
                    print(f"  ✅ Saved mesh.gif")

            success_count += 1
            print(f"  ✅ Done!")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"📊 BATCH COMPLETE: {success_count}/{len(pairs)} succeeded")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
