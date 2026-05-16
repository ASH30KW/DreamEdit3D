#!/usr/bin/env python3
"""
Unified renderer for consistent color rendering throughout the pipeline.
Uses nvdiffrast with UNLIT/flat rendering (no lighting) for consistent colors.

This ensures the same colors from:
- Data preprocessing (multi-view generation)
- Training visualization
- Evaluation renders
- Benchmark GIFs

Usage:
    from utils.unified_renderer import UnifiedRenderer

    renderer = UnifiedRenderer(device="cuda")
    images = renderer.render_views(mesh_path, num_views=6)
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

# Add snap_gtr to path for nvdiffrast utilities
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "snap_gtr"))

try:
    from utils.render_utils import get_cameras, NVDiffRasterizerContext
    NVDIFFRAST_AVAILABLE = True
except ImportError:
    NVDIFFRAST_AVAILABLE = False
    print("Warning: nvdiffrast not available. Install with: pip install nvdiffrast")


class UnifiedRenderer:
    """
    Unified renderer using nvdiffrast for consistent UNLIT color rendering.

    Key features:
    - NO lighting effects (flat/emission-style rendering)
    - Directly renders vertex colors or sampled texture colors
    - Consistent output across all pipeline stages
    """

    def __init__(self, device="cuda"):
        if not NVDIFFRAST_AVAILABLE:
            raise RuntimeError("nvdiffrast is required for UnifiedRenderer")

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.ctx = NVDiffRasterizerContext(self.device)

    def load_mesh(self, mesh_path: str, normalize: bool = True) -> trimesh.Trimesh:
        """
        Load mesh from file and ensure it has vertex colors.

        Args:
            mesh_path: Path to mesh file (.glb, .obj, .ply, etc.)
            normalize: If True, center and scale mesh to [-1, 1]

        Returns:
            trimesh.Trimesh with vertex colors
        """
        # Load mesh (force='mesh' concatenates Scene geometries)
        mesh = trimesh.load(mesh_path, force='mesh')

        # Ensure vertex colors exist
        mesh = self._ensure_vertex_colors(mesh)

        # Normalize mesh position and scale
        if normalize:
            mesh.vertices -= mesh.centroid
            scale = 2.0 / np.max(mesh.extents)
            mesh.vertices *= scale

        return mesh

    def _ensure_vertex_colors(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Ensure mesh has proper per-vertex RGBA colors."""
        n = len(mesh.vertices)

        # Try to convert texture visuals to vertex colors
        if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
            try:
                material = getattr(mesh.visual, 'material', None)
                texture_image = None
                base_color = None

                if material is not None:
                    # Try different texture sources (PBR vs Simple materials)
                    if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                        texture_image = material.baseColorTexture
                    elif hasattr(material, 'image') and material.image is not None:
                        texture_image = material.image
                    # Get base color factor (solid color fallback)
                    if hasattr(material, 'baseColorFactor') and material.baseColorFactor is not None:
                        base_color = np.array(material.baseColorFactor, dtype=np.uint8)

                # Sample texture using UV coordinates
                if texture_image is not None and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                    uv = np.array(mesh.visual.uv)
                    vertex_colors = trimesh.visual.uv_to_color(uv, texture_image)
                    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
                # Use solid base color
                elif base_color is not None:
                    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=np.tile(base_color, (n, 1)))
                else:
                    mesh.visual = mesh.visual.to_color()
            except Exception as e:
                print(f"Warning: texture conversion failed ({e}), using default color")

        # Verify vertex colors shape
        vc = getattr(mesh.visual, 'vertex_colors', None)
        if vc is None or np.asarray(vc).shape != (n, 4):
            # Default gray color
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.full((n, 4), [180, 180, 180, 255], dtype=np.uint8)
            )

        return mesh

    def get_camera_poses(
        self,
        num_views: int,
        elevation_deg: float = 20.0,
        azimuth_start: float = 0.0,
        azimuth_span: float = 360.0,
        camera_distance: float = 3.5,
        fov_deg: float = 50.0,
        resolution: int = 512,
    ) -> dict:
        """
        Generate camera poses for multi-view rendering.

        Args:
            num_views: Number of camera views
            elevation_deg: Camera elevation angle in degrees
            azimuth_start: Starting azimuth angle in degrees
            azimuth_span: Total azimuth span in degrees
            camera_distance: Distance from origin
            fov_deg: Field of view in degrees
            resolution: Image resolution (square)

        Returns:
            Camera dictionary with mvp matrices and metadata
        """
        azimuths = np.linspace(
            azimuth_start, azimuth_start + azimuth_span,
            num=num_views, endpoint=False, dtype=np.float32
        )
        elevations = np.full(num_views, elevation_deg, dtype=np.float32)

        cameras = get_cameras(
            azimuth_deg=torch.from_numpy(azimuths),
            elevation_deg=torch.from_numpy(elevations),
            width=resolution,
            height=resolution,
            fov=fov_deg,
            camera_distance=camera_distance,
        )

        return cameras

    def render_mesh(
        self,
        mesh: trimesh.Trimesh,
        cameras: dict,
        bg_color: tuple = (1.0, 1.0, 1.0),
        transparent: bool = False,
    ) -> np.ndarray:
        """
        Render mesh from multiple camera views with UNLIT coloring.

        Args:
            mesh: Trimesh object with vertex colors
            cameras: Camera dictionary from get_camera_poses()
            bg_color: Background color as RGB tuple (0-1 range)
            transparent: If True, return RGBA images with alpha channel

        Returns:
            Array of rendered images, shape (num_views, H, W, 3 or 4), uint8
        """
        # Prepare mesh data
        vertices = np.array(mesh.vertices, dtype=np.float32)
        triangles = np.array(mesh.faces, dtype=np.int32)
        vertex_colors = np.array(mesh.visual.vertex_colors, dtype=np.float32) / 255.0

        v = torch.from_numpy(vertices).contiguous().to(self.device)
        f = torch.from_numpy(triangles).contiguous().to(self.device)
        vc = torch.from_numpy(vertex_colors).contiguous().to(self.device)

        mvp_mtx = cameras["mvp_mtx"].to(self.device)
        h, w = cameras["height"], cameras["width"]
        num_views = mvp_mtx.shape[0]

        bg = np.array(bg_color)
        images = []

        for i in range(num_views):
            # Transform vertices to clip space
            v_clip = self.ctx.vertex_transform(v, mvp_mtx[i:i+1])

            # Rasterize
            rast, rast_db = self.ctx.rasterize(v_clip, f, (h, w))

            # Interpolate vertex colors (UNLIT - no shading!)
            out, _ = self.ctx.interpolate(vc, rast, f)

            # Convert to numpy and composite with background
            rgbs = out.cpu().numpy()[0, ::-1, :, :]  # Flip Y axis
            alpha = rgbs[..., 3:4]
            rgb = rgbs[..., :3]

            if transparent:
                # Return RGBA with alpha channel
                rgba = np.concatenate([rgb, alpha], axis=-1)
                rgba = (rgba * 255.0).clip(0, 255).astype(np.uint8)
                images.append(rgba)
            else:
                # Alpha composite with background (UNLIT compositing)
                composited = rgb * alpha + bg * (1 - alpha)
                composited = (composited * 255.0).clip(0, 255).astype(np.uint8)
                images.append(composited)

        return np.stack(images, axis=0)

    def render_views(
        self,
        mesh_path: str,
        output_dir: str = None,
        num_views: int = 6,
        resolution: int = 512,
        camera_distance: float = 3.5,
        fov_deg: float = 50.0,
        elevation_deg: float = 20.0,
        azimuth_start: float = 0.0,
        azimuth_span: float = 360.0,
        bg_color: tuple = (1.0, 1.0, 1.0),
        save_images: bool = True,
        transparent: bool = False,
    ) -> np.ndarray:
        """
        Complete pipeline: load mesh, generate cameras, render views.

        Args:
            mesh_path: Path to mesh file
            output_dir: Directory to save images (optional)
            num_views: Number of views to render
            resolution: Image resolution
            camera_distance: Camera distance from origin
            fov_deg: Field of view in degrees
            elevation_deg: Camera elevation in degrees
            transparent: If True, return RGBA images with alpha channel
            azimuth_start: Starting azimuth in degrees
            azimuth_span: Total azimuth span in degrees
            bg_color: Background color RGB (0-1 range)
            save_images: Whether to save images to disk

        Returns:
            Array of rendered images, shape (num_views, H, W, 3), uint8
        """
        # Load and normalize mesh
        mesh = self.load_mesh(mesh_path, normalize=True)
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

        # Generate camera poses
        cameras = self.get_camera_poses(
            num_views=num_views,
            elevation_deg=elevation_deg,
            azimuth_start=azimuth_start,
            azimuth_span=azimuth_span,
            camera_distance=camera_distance,
            fov_deg=fov_deg,
            resolution=resolution,
        )

        # Render (UNLIT)
        images = self.render_mesh(mesh, cameras, bg_color=bg_color, transparent=transparent)
        print(f"Rendered {num_views} views at {resolution}x{resolution}")

        # Save images
        if save_images and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, img in enumerate(images):
                path = os.path.join(output_dir, f"rgb_{i+1:03d}.png")
                Image.fromarray(img).save(path)
            print(f"Saved images to {output_dir}")

        return images

    def render_to_gif(
        self,
        mesh_path: str,
        output_path: str,
        num_frames: int = 50,
        fps: int = 15,
        resolution: int = 512,
        camera_distance: float = 3.5,
        fov_deg: float = 50.0,
        elevation_deg: float = 20.0,
        bg_color: tuple = (1.0, 1.0, 1.0),
        transparent: bool = False,
    ):
        """
        Render mesh to rotating GIF.

        Args:
            mesh_path: Path to mesh file
            output_path: Output GIF path
            num_frames: Number of frames in GIF
            fps: Frames per second
            resolution: Image resolution
            camera_distance: Camera distance
            fov_deg: Field of view
            elevation_deg: Camera elevation
            bg_color: Background color RGB (0-1)
            transparent: If True, create GIF with transparent background
        """
        import imageio.v2 as imageio

        images = self.render_views(
            mesh_path=mesh_path,
            output_dir=None,
            num_views=num_frames,
            resolution=resolution,
            camera_distance=camera_distance,
            fov_deg=fov_deg,
            elevation_deg=elevation_deg,
            azimuth_start=0,
            azimuth_span=360,
            bg_color=bg_color,
            save_images=False,
            transparent=transparent,
        )

        # Save as GIF
        duration = 1.0 / fps
        if transparent:
            # Save with transparency (GIF format supports transparency)
            imageio.mimsave(output_path, images, duration=duration, loop=0, disposal=2)
        else:
            imageio.mimsave(output_path, images, duration=duration, loop=0)
        print(f"Saved GIF: {output_path} ({num_frames} frames, {fps} fps, transparent={transparent})")


def main():
    """CLI interface for unified renderer."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified UNLIT renderer for consistent colors")
    parser.add_argument("--mesh", type=str, required=True, help="Path to mesh file")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for images")
    parser.add_argument("--gif", type=str, default=None, help="Output GIF path (renders rotating view)")
    parser.add_argument("--num_views", type=int, default=6, help="Number of views")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--camera_distance", type=float, default=3.5, help="Camera distance")
    parser.add_argument("--fov_deg", type=float, default=50, help="Field of view in degrees")
    parser.add_argument("--elevation_deg", type=float, default=20, help="Camera elevation in degrees")
    parser.add_argument("--azimuth_start", type=float, default=0, help="Starting azimuth in degrees")
    parser.add_argument("--bg_color", type=str, default="white", choices=["white", "gray", "black"],
                        help="Background color")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")

    args = parser.parse_args()

    # Parse background color
    bg_colors = {"white": (1.0, 1.0, 1.0), "gray": (0.5, 0.5, 0.5), "black": (0.0, 0.0, 0.0)}
    bg_color = bg_colors.get(args.bg_color, (1.0, 1.0, 1.0))

    renderer = UnifiedRenderer()

    if args.gif:
        # Render GIF
        renderer.render_to_gif(
            mesh_path=args.mesh,
            output_path=args.gif,
            num_frames=args.num_views,
            fps=args.fps,
            resolution=args.resolution,
            camera_distance=args.camera_distance,
            fov_deg=args.fov_deg,
            elevation_deg=args.elevation_deg,
            bg_color=bg_color,
        )
    else:
        # Render multi-view images
        output_dir = args.output_dir or "renders"
        renderer.render_views(
            mesh_path=args.mesh,
            output_dir=output_dir,
            num_views=args.num_views,
            resolution=args.resolution,
            camera_distance=args.camera_distance,
            fov_deg=args.fov_deg,
            elevation_deg=args.elevation_deg,
            azimuth_start=args.azimuth_start,
            bg_color=bg_color,
        )


if __name__ == "__main__":
    main()
