"""
Example script to convert zero123++ image grid to input format of LRM.

For other MV generation approaches, modify the camera parameters (FOV, radius, elevation, and azimuth) accordingly.
"""
from PIL import Image, ImageOps
import rembg
import click
from loguru import logger
import math
import numpy as np
from pathlib import Path


def pad_and_remove_background(image: Image.Image, border_size: int, rembg_session) -> Image.Image:
    # overflow object need first padding before run segmentation
    padded_image = ImageOps.expand(image, border=border_size, fill="white")

    foreground_image = rembg.remove(padded_image, session=rembg_session)

    width, height = foreground_image.size
    crop_box = (border_size, border_size, width - border_size, height - border_size)
    cropped_image = foreground_image.crop(crop_box)

    return cropped_image


def fov_to_intrinsic(fov_degree, width, height):
    # Convert FOV from degrees to radians
    fov_radian = math.radians(fov_degree)

    # Calculate the focal length
    # Assuming the same focal length for both x and y axes
    f = width / (2 * math.tan(fov_radian / 2))

    # Calculate the center of the image (principal point)
    cx = width / 2
    cy = height / 2

    # Constructing the intrinsic matrix
    intrinsic_matrix = np.array([[f, 0, cx],
                                 [0, f, cy],
                                 [0, 0, 1]])
    return intrinsic_matrix


def get_cam_pose(theta, phi, radius):
    """
    Args:
        theta: angle between up axis
        phi: angle between right axis
    Note: return in world coordinate. Y is up, x is right. +z is forward.
    """
    theta, phi = np.radians(theta), np.radians(phi)
    y = radius * np.cos(theta)
    x = radius * np.sin(theta) * np.cos(phi)
    z = radius * np.sin(theta) * np.sin(phi)
    return np.array([x, y, z])


def get_c2w_opencv(eye, center, up=np.array([0.0, -1.0, 0.0])):
    # eye position in world coordinate
    # center position in world coordinate
    # up direction in world coordinate
    # return transformation from camera to world
    forward = (center - eye)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    new_up = np.cross(forward, right)
    new_up /= np.linalg.norm(new_up)

    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack((right, new_up, forward))
    c2w[:3, 3] = eye

    return c2w


def save_camera(out_cam_file, K, pose, size=(1000, 1000)):
    """
    Export camera extrinsic and intrinsic matrix to file
    Args:
        K: 3x3 intrinsic matrix
        pose: world2 camera transformation matrix
        size: image size
    """
    Path(out_cam_file).parent.mkdir(exist_ok=True, parents=True)
    with open(out_cam_file, "w") as fout:
        # write extrinsic
        fout.write("extrinsic\n")
        for rind in range(3):
            for cind in range(4):
                fout.write(f'{pose[rind][cind]:f} ')
            fout.write('\n')
        fout.write('0 0 0 1\n\n')

        # intrinsic
        height, width = size
        fout.write("intrinsic fx, fy, cx, cy, height, width \n")
        fout.write(f'{K[0][0]:f} {K[1][1]:f} {K[0][2]:f} {K[1][2]:f} {height} {width}')


def read_camera(file: str):
    """Return cam_ext and cam_intrinsic"""
    assert Path(file).is_file(), f"Cannot find {file}"
    f = open(file)
    text = f.readlines()
    cam_ext = np.array([[float(y) for y in x.split()] for x in text[1:5]])
    fx, fy, cx, cy, height, width = np.array([float(y) for y in text[7].split()])
    f.close()
    cam_int = np.eye(3)
    cam_int[0, 0], cam_int[1, 1], cam_int[0, 2], cam_int[1, 2] = fx, fy, cx, cy
    return cam_ext, cam_int, height, width


def prepare_cameras(camera_dir, fov, img_size, theta_list, phi_list, radius_list):
    """"""
    width, height = img_size
    K = fov_to_intrinsic(fov, width, height)

    num_camera = len(phi_list)
    for i in range(num_camera):
        theta, phi, radius = theta_list[i], phi_list[i], radius_list[i]
        cam_pos = get_cam_pose(theta, phi, radius)
        c2w = get_c2w_opencv(cam_pos, np.array([0.0, 0.0, 0.0]))
        pose = np.linalg.inv(c2w)
        out_cam_file = f"{camera_dir}/cam_{i:03d}.txt"
        save_camera(out_cam_file, K, pose, size=img_size)
        # print(cam_pos, out_cam_file)


@click.command()
@click.option("--in_dir", type=str, help="Path to input image file or directory with view subdirectories")
@click.option("--out_dir", type=str, help="Path to output directory")
@click.option("--azimuth_start", type=float, default=0, help="Starting azimuth angle offset in degrees (default: 0)")
@click.option("--cam_radius", type=float, default=None, help="Camera radius (distance from object). If not set, uses default based on view count")
@click.option("--fov", type=float, default=None, help="Field of view in degrees. If not set, uses default based on view count")
@click.option("--elevation", type=float, default=None, help="Camera elevation in degrees (0=horizontal, 90=top-down). Internally converted to theta = 90 - elevation")
@click.option("--elevation_list", type=str, default=None, help="Comma-separated list of per-view elevations (overrides --elevation)")
@click.option("--azimuth_list", type=str, default=None, help="Comma-separated list of per-view azimuths (overrides --azimuth_start)")
@click.option("--padding", type=int, default=0, help="Padding in pixels to add around each view (fixes boundary blur). 0=disabled")
def main(
    in_dir: str,
    out_dir: str,
    azimuth_start: float,
    cam_radius: float,
    fov: float,
    elevation: float,
    elevation_list: str,
    azimuth_list: str,
    padding: int,
):
    img_path_to_np01 = lambda img_file: np.array(Image.open(img_file)).astype('float') / 255.
    np01_to_pil = lambda x: Image.fromarray((x * 255.).astype(np.uint8))
    rembg_session = rembg.new_session()

    Path(out_dir).mkdir(exist_ok=True, parents=True)

    in_path = Path(in_dir)

    # Check if input is a directory with view subdirectories (race-chicken-mv format)
    if in_path.is_dir():
        view_dirs = sorted([d for d in in_path.iterdir() if d.is_dir() and d.name.startswith('view_')])
        if view_dirs:
            # race-chicken-mv format with 4 views
            logger.info(f"Detected multi-view directory format with {len(view_dirs)} views")
            num_views = len(view_dirs)
            img_list = []

            for view_dir in view_dirs:
                img_file = view_dir / "img.jpg"
                mask_file = view_dir / "mask0.png"

                if img_file.exists():
                    img = img_path_to_np01(str(img_file))

                    # If mask exists, apply it to extract only masked area
                    if mask_file.exists():
                        mask = img_path_to_np01(str(mask_file))
                        # Handle different mask formats (RGB or grayscale)
                        if len(mask.shape) == 3:
                            mask = mask[:, :, 0]  # Take first channel if RGB
                        # Apply mask: keep masked area, set background to white
                        img = img * mask[:, :, None] + (1 - mask[:, :, None])

                    img_list.append(img)
                else:
                    raise FileNotFoundError(f"Image not found in {view_dir}")

            # Camera params for 4-view chicken format
            # Assuming views are: front, right, back, left
            default_theta = 90  # horizontal views (elevation = 0)
            phi_list = [azimuth_start + a for a in [0, 90, 180, 270]]
            default_cam_radius = 3
            default_fov = 60
            # Use user-provided values if set, otherwise use defaults
            # elevation = 90 - theta, so theta = 90 - elevation
            actual_theta = (90 - elevation) if elevation is not None else default_theta
            actual_cam_radius = cam_radius if cam_radius is not None else default_cam_radius
            actual_fov = fov if fov is not None else default_fov
            theta_list = [actual_theta] * num_views
            radius_list = [actual_cam_radius] * num_views
            width, height = 512, 512
        else:
            raise ValueError(f"Directory {in_dir} does not contain view subdirectories")
    else:
        # Image file - intelligent grid detection
        img = img_path_to_np01(in_dir)
        h, w = img.shape[:2]
        aspect_ratio = w / h

        logger.info(f"Input image size: {w}×{h}, aspect ratio: {aspect_ratio:.2f}")

        # Try to detect grid layout intelligently
        def detect_grid_layout(width, height):
            """Detect number of rows and columns in the grid"""
            aspect = width / height

            # Common grid patterns (rows, cols, expected_aspect_ratio)
            # expected_aspect_ratio = cols / rows (for square cells)
            possible_grids = [
                (1, 6, 6.0),       # 1×6 horizontal strip
                (1, 5, 5.0),       # 1×5 horizontal strip
                (1, 4, 4.0),       # 1×4 horizontal strip
                (1, 3, 3.0),       # 1×3 horizontal strip
                (2, 3, 1.5),       # 2×3 grid (landscape)
                (3, 2, 0.667),     # 3×2 grid (portrait) - for 640×960
                (2, 2, 1.0),       # 2×2 grid
                (3, 3, 1.0),       # 3×3 grid
                (3, 1, 0.333),     # 3×1 vertical strip
                (4, 2, 0.5),       # 4×2 grid (portrait)
                (2, 4, 2.0),       # 2×4 grid (landscape)
                (6, 1, 0.167),     # 6×1 vertical strip
            ]

            # Find best match
            best_match = None
            min_diff = float('inf')

            for rows, cols, expected_aspect in possible_grids:
                diff = abs(aspect - expected_aspect)
                if diff < min_diff:
                    min_diff = diff
                    best_match = (rows, cols)

            return best_match

        rows, cols = detect_grid_layout(w, h)
        num_views = rows * cols

        logger.info(f"🔍 Auto-detected grid layout: {rows}×{cols} = {num_views} views")

        # Split image into grid
        img_list = []
        row_height = h // rows
        col_width = w // cols

        for row in range(rows):
            for col in range(cols):
                y_start = row * row_height
                y_end = (row + 1) * row_height
                x_start = col * col_width
                x_end = (col + 1) * col_width

                view = img[y_start:y_end, x_start:x_end]
                img_list.append(view)

        # Generate camera parameters based on number of views
        if num_views == 4:
            # 4-view: front, right, back, left
            logger.info("Using 4-view camera setup (front, right, back, left)")
            default_theta = 75  # elevation = 15 degrees
            phi_list = [azimuth_start + a for a in [0, -90, -180, -270]]
            default_cam_radius = 4
            default_fov = 30

        elif num_views == 6:
            # 6-view: Zero123++ setup
            logger.info("Using 6-view Zero123++ camera setup")
            default_theta = 75  # elevation = 15 degrees
            phi_list = [azimuth_start + a for a in [0, -60, -120, -180, -240, -300]]
            default_cam_radius = 4
            default_fov = 30

        else:
            # Generate evenly distributed views around the object
            logger.info(f"Generating {num_views} evenly distributed views")
            default_theta = 90  # elevation = 0 (horizontal)
            phi_list = [azimuth_start + 360 * i / num_views for i in range(num_views)]
            default_cam_radius = 4
            default_fov = 30

        # Use user-provided values if set, otherwise use defaults
        # elevation = 90 - theta, so theta = 90 - elevation
        actual_theta = (90 - elevation) if elevation is not None else default_theta
        actual_cam_radius = cam_radius if cam_radius is not None else default_cam_radius
        actual_fov = fov if fov is not None else default_fov
        theta_list = [actual_theta] * num_views
        radius_list = [actual_cam_radius] * num_views

        # Override with per-view lists if provided
        if elevation_list:
            elev_values = [float(e.strip()) for e in elevation_list.replace('，', ',').split(',') if e.strip()]
            # Cycle through if fewer values than views
            theta_list = [(90 - elev_values[i % len(elev_values)]) for i in range(num_views)]
            logger.info(f"Using per-view elevations: {elev_values} -> theta_list: {theta_list}")

        if azimuth_list:
            azim_values = [float(a.strip()) for a in azimuth_list.replace('，', ',').split(',') if a.strip()]
            # Convert azimuth to phi: phi = 90 - azimuth (cycle if needed)
            phi_list = [(90 - azim_values[i % len(azim_values)]) for i in range(num_views)]
            logger.info(f"Using per-view azimuths: {azim_values} -> phi_list: {phi_list}")

        width, height = 512, 512

    # Create subdirectory for separated views
    separated_dir = Path(out_dir).parent / "separated_views"
    separated_dir.mkdir(exist_ok=True, parents=True)

    if padding > 0:
        logger.info(f"Applying {padding}px padding to each view")

    logger.info(f"Processing {len(img_list)} views...")

    for i in range(len(img_list)):
        img = img_list[i]
        pil_img = np01_to_pil(img)

        # Save original separated view (before background removal)
        original_view_file = f"{separated_dir}/view_{i:03d}_original.png"
        pil_img.save(original_view_file)

        # Apply padding if specified (extends boundary with background color)
        if padding > 0:
            # Detect background color from corners (average of corner pixels)
            img_array = np.array(pil_img)
            corner_pixels = [
                img_array[0, 0],           # top-left
                img_array[0, -1],          # top-right
                img_array[-1, 0],          # bottom-left
                img_array[-1, -1]          # bottom-right
            ]
            bg_color = tuple(np.mean(corner_pixels, axis=0).astype(np.uint8))

            # Expand image with detected background color
            pil_img = ImageOps.expand(pil_img, border=padding, fill=bg_color)
            logger.info(f"  View {i}: Added {padding}px padding with bg color {bg_color}")

        # Process view (remove background and resize)
        border_size = int(pil_img.size[0] * 0.15)
        view = pad_and_remove_background(pil_img, border_size, rembg_session)
        view = view.resize((width, height))

        # Save processed view for GTR
        out_file = f"{out_dir}/rgb_{i:03d}.png"
        view.save(out_file)

        # Also save processed view in separated directory
        separated_view_file = f"{separated_dir}/view_{i:03d}_processed.png"
        view.save(separated_view_file)

    prepare_cameras(out_dir, actual_fov, (width, height), theta_list, phi_list, radius_list)

    logger.info(f"✅ Saved {len(img_list)} processed views to: {out_dir}")
    logger.info(f"✅ Saved {len(img_list)} separated views to: {separated_dir}")
    logger.info(f"   - Original views: view_XXX_original.png")
    logger.info(f"   - Processed views: view_XXX_processed.png")


if __name__ == "__main__":
    main()