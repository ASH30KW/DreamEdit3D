import gradio as gr
import subprocess
import os
from pathlib import Path
import shutil
import tempfile

# Set paths
CKPT_PATH = "ckpts/full_checkpoint.pth"
TEMP_DIR = "temp_gradio"
EXAMPLES_DIR = "examples"

def prepare_multiview(input_file_or_dir, selected_example=None):
    """Run prepare_mv.py script"""
    try:
        # Create temp directory for outputs
        Path(TEMP_DIR).mkdir(exist_ok=True)

        # Determine input source
        if selected_example:
            temp_input = selected_example
        elif input_file_or_dir:
            # Handle uploaded file
            if hasattr(input_file_or_dir, 'name'):
                temp_input = input_file_or_dir.name
            else:
                temp_input = input_file_or_dir
        else:
            return None, "Please select an example or upload an image"

        # Create unique output directory
        out_dir = Path(TEMP_DIR) / "prepared_mv"
        out_dir.mkdir(exist_ok=True, parents=True)

        # Run prepare script
        cmd = [
            "python", "scripts/prepare_mv.py",
            "--in_dir", str(temp_input),
            "--out_dir", str(out_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Get prepared images for preview
        prepared_images = sorted(list(out_dir.glob("rgb_*.png")))

        if not prepared_images:
            return None, f"Error: No images generated\n{result.stderr}"

        return str(out_dir), f"✓ Prepared {len(prepared_images)} views\n{result.stdout}"

    except subprocess.CalledProcessError as e:
        return None, f"Error running prepare_mv.py:\n{e.stderr}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_inference(prepared_dir, checkpoint_path=CKPT_PATH):
    """Run inference.py script"""
    try:
        if not prepared_dir:
            return None, None, "Please run preparation first"

        if not Path(checkpoint_path).exists():
            return None, None, f"Checkpoint not found: {checkpoint_path}"

        # Create output directory
        out_dir = Path(TEMP_DIR) / "inference_output"
        out_dir.mkdir(exist_ok=True, parents=True)

        # Run inference script
        cmd = [
            "python", "scripts/inference.py",
            "--ckpt_path", checkpoint_path,
            "--in_dir", prepared_dir,
            "--out_dir", str(out_dir),
            "--seed", "2025"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Get output files
        mesh_file = out_dir / "mesh.obj"
        mesh_gif = out_dir / "mesh.gif"
        nerf_gif = out_dir / "nerf.gif"

        if not mesh_file.exists():
            return None, None, f"Error: Mesh not generated\n{result.stderr}"

        return (
            str(mesh_file) if mesh_file.exists() else None,
            str(mesh_gif) if mesh_gif.exists() else None,
            str(nerf_gif) if nerf_gif.exists() else None,
            f"✓ Inference completed\n{result.stdout[-500:]}"  # Last 500 chars
        )

    except subprocess.CalledProcessError as e:
        return None, None, None, f"Error running inference.py:\n{e.stderr}"
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"


def full_pipeline(input_file_or_dir, checkpoint_path, selected_example=None):
    """Run both prepare and inference"""
    # Step 1: Prepare
    prepared_dir, prep_log = prepare_multiview(input_file_or_dir, selected_example)

    if not prepared_dir:
        return None, None, None, prep_log

    # Step 2: Inference
    mesh_file, mesh_gif, nerf_gif, inf_log = run_inference(prepared_dir, checkpoint_path)

    combined_log = f"=== PREPARATION ===\n{prep_log}\n\n=== INFERENCE ===\n{inf_log}"

    return mesh_file, mesh_gif, nerf_gif, combined_log


# Get example images and directories
def get_example_images():
    """Get list of example images from examples directory"""
    example_files = []
    if Path(EXAMPLES_DIR).exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            example_files.extend(sorted(Path(EXAMPLES_DIR).glob(ext)))
    return [str(f) for f in example_files]

def get_prepared_directories():
    """Get list of prepared multi-view directories"""
    prepared_dirs = []
    examples_path = Path(EXAMPLES_DIR)

    # Check generated_multiview directories
    generated_mv_path = examples_path / "generated_multiview"
    if generated_mv_path.exists():
        for subdir in generated_mv_path.iterdir():
            if subdir.is_dir():
                # Check if it has rgb_*.png and cam_*.txt files
                rgb_files = list(subdir.glob("rgb_*.png"))
                cam_files = list(subdir.glob("cam_*.txt"))
                if rgb_files and cam_files:
                    prepared_dirs.append(str(subdir))

    # Check other prepared directories (like race-chicken-mv)
    for subdir in examples_path.iterdir():
        if subdir.is_dir() and subdir.name.endswith('-mv'):
            # Check if it has view_* subdirectories or rgb_*.png files
            view_dirs = list(subdir.glob("view_*"))
            rgb_files = list(subdir.glob("rgb_*.png"))
            if view_dirs or rgb_files:
                prepared_dirs.append(str(subdir))

    return sorted(prepared_dirs)

# Create Gradio interface
with gr.Blocks(title="DreamEdit3D + GTR Pipeline") as demo:
    gr.Markdown("# DreamEdit3D + GTR: Multi-view to 3D Mesh")
    gr.Markdown("Upload an image grid or multi-view directory to generate a 3D mesh")

    with gr.Tab("Full Pipeline"):
        with gr.Row():
            with gr.Column():
                input_image = gr.File(label="Upload Image", file_types=["image"])

                # Add example image selector
                example_images = get_example_images()
                if example_images:
                    example_dropdown = gr.Dropdown(
                        choices=example_images,
                        label="Or select from examples",
                        value=None
                    )

                checkpoint = gr.Textbox(value=CKPT_PATH, label="Checkpoint Path")
                run_btn = gr.Button("Run Full Pipeline", variant="primary")

            with gr.Column():
                output_mesh = gr.File(label="Output Mesh (.obj)")
                output_mesh_gif = gr.Image(label="Mesh Rendering (GIF)")
                output_nerf_gif = gr.Image(label="NeRF Rendering (GIF)")

        logs = gr.Textbox(label="Logs", lines=10)

        run_btn.click(
            full_pipeline,
            inputs=[input_image, checkpoint, example_dropdown],
            outputs=[output_mesh, output_mesh_gif, output_nerf_gif, logs]
        )

    with gr.Tab("Step-by-Step"):
        gr.Markdown("### Step 1: Prepare Multi-view")
        with gr.Row():
            with gr.Column():
                prep_input = gr.File(label="Upload Image", file_types=["image"])

                # Add example selector for step-by-step
                if example_images:
                    prep_example_dropdown = gr.Dropdown(
                        choices=example_images,
                        label="Or select from examples",
                        value=None
                    )

                prep_btn = gr.Button("Prepare Multi-view")
            with gr.Column():
                prep_output_dir = gr.Textbox(label="Prepared Directory", interactive=False)
                prep_logs = gr.Textbox(label="Preparation Logs", lines=5)

        prep_btn.click(
            prepare_multiview,
            inputs=[prep_input, prep_example_dropdown],
            outputs=[prep_output_dir, prep_logs]
        )

        gr.Markdown("### Step 2: Run Inference")
        with gr.Row():
            with gr.Column():
                inf_prepared_dir = gr.Textbox(label="Prepared Directory (from Step 1)")

                # Add prepared directory selector
                prepared_dirs = get_prepared_directories()
                if prepared_dirs:
                    prepared_dir_dropdown = gr.Dropdown(
                        choices=prepared_dirs,
                        label="Or select prepared example directory",
                        value=None
                    )

                    def load_prepared_dir(dir_path):
                        if dir_path:
                            return dir_path
                        return None

                    prepared_dir_dropdown.change(
                        load_prepared_dir,
                        inputs=[prepared_dir_dropdown],
                        outputs=[inf_prepared_dir]
                    )

                inf_checkpoint = gr.Textbox(value=CKPT_PATH, label="Checkpoint Path")
                inf_btn = gr.Button("Run Inference")
            with gr.Column():
                inf_mesh = gr.File(label="Output Mesh (.obj)")
                inf_mesh_gif = gr.Image(label="Mesh Rendering (GIF)")
                inf_nerf_gif = gr.Image(label="NeRF Rendering (GIF)")

        inf_logs = gr.Textbox(label="Inference Logs", lines=5)

        inf_btn.click(
            run_inference,
            inputs=[inf_prepared_dir, inf_checkpoint],
            outputs=[inf_mesh, inf_mesh_gif, inf_nerf_gif, inf_logs]
        )

    gr.Markdown("""
    ## Usage
    1. **Full Pipeline**: Upload an image and click "Run Full Pipeline" to execute both steps
    2. **Step-by-Step**:
        - First prepare multi-view images from input
        - Then run inference to generate 3D mesh

    ## Supported Inputs
    - Zero123++ grid format (3x2 views)
    - 1x4 horizontal grid (4 views)
    - Directory with view subdirectories (view_0, view_1, etc.)
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
