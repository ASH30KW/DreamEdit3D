import gradio as gr
import os
import subprocess
import sys
import shutil
from pathlib import Path
import tempfile
import zipfile
from PIL import Image
import json
import time

# Fix MKL threading issue - add these environment variables before any numpy/torch imports
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# Alternative: os.environ['MKL_NUM_THREADS'] = '1'

class DreamEdit3DApp:
    def __init__(self):
        # Use absolute paths based on the location of this file
        self.base_dir = Path(__file__).resolve().parent

        # These are now set per-project, not created globally
        self.temp_dir = None
        self.models_dir = None

        # Initialize examples by scanning the examples directory
        self.examples = self.load_examples()
        
    def load_examples(self):
        """Load examples from the examples directory"""
        examples = {}
        examples_dir = self.base_dir / "examples"

        if not examples_dir.exists():
            return examples

        for example_dir in examples_dir.iterdir():
            if example_dir.is_dir():
                # Look for main image and masks
                main_img = None
                masks = []

                # Common image extensions
                img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

                # Check if it's a multi-view structure (has view_* subdirectories)
                view_dirs = sorted([d for d in example_dir.iterdir() if d.is_dir() and d.name.startswith('view_')])

                if view_dirs:
                    # Multi-view structure: use first view (view_1)
                    first_view = view_dirs[0]
                    for file in first_view.iterdir():
                        if file.suffix.lower() in img_extensions:
                            if 'mask' in file.name.lower():
                                masks.append(file)
                            elif main_img is None:
                                main_img = file
                else:
                    # Flat structure: look directly in example_dir
                    for file in example_dir.iterdir():
                        if file.suffix.lower() in img_extensions:
                            if 'mask' in file.name.lower():
                                masks.append(file)
                            elif main_img is None:  # First non-mask image becomes main
                                main_img = file

                if main_img and masks:
                    examples[example_dir.name] = {
                        'main_image': str(main_img),
                        'masks': [str(m) for m in sorted(masks)],
                        'num_concepts': len(masks)
                    }

        return examples
    
    def prepare_training_data(self, main_image, mask_files, concept_names, output_dir=None):
        """Prepare training data from uploaded files"""
        if main_image is None:
            return "Error: Please upload a main image"

        if not mask_files:
            return "Error: Please upload mask files"

        # Create unique training directory
        timestamp = str(int(time.time()))
        if output_dir:
            training_dir = Path(output_dir) / f"training_{timestamp}"
        else:
            training_dir = self.temp_dir / f"training_{timestamp}"
        training_dir.mkdir(exist_ok=True, parents=True)
        
        # Save main image as img.jpg
        main_img = Image.open(main_image)
        main_img.save(training_dir / "img.jpg")
        
        # Save mask files
        for i, mask_file in enumerate(mask_files):
            if mask_file is not None:
                mask_img = Image.open(mask_file)
                mask_img.save(training_dir / f"mask{i}.png")
        
        return str(training_dir)
    
    def train_model(self, main_image, mask_files, concept_names, phase1_steps, phase2_steps, temp_dir=None, models_dir=None, multiview_data_dir=None, mvdream_training_mode="2d", progress=gr.Progress()):
        """Train the DreamEdit3D model"""
        try:
            progress(0, desc="Preparing training data...")

            # ---- Strong typing for steps (fix #1)
            try:
                phase1_steps = int(phase1_steps)
                phase2_steps = int(phase2_steps)
            except Exception:
                return "❌ Phase steps must be integers.", "", ""

            # Use existing multi-view directory if provided, otherwise prepare training data
            if multiview_data_dir:
                training_dir = multiview_data_dir
                # Count masks in first view for multi-view data
                view_1_dir = Path(training_dir) / "view_1"
                if view_1_dir.exists():
                    # Only count numbered masks (mask0.png, mask1.png, etc), not labeled ones (mask_body.png)
                    saved_masks = sorted([f for f in view_1_dir.glob("mask*.png") if f.stem.replace("mask", "").isdigit()])

                    # Verify all 4 views have the same number of masks
                    for view_idx in range(1, 5):
                        view_dir = Path(training_dir) / f"view_{view_idx}"
                        if not view_dir.exists():
                            return f"❌ Missing view_{view_idx} directory in multi-view data.", "", ""
                        view_masks = sorted([f for f in view_dir.glob("mask*.png") if f.stem.replace("mask", "").isdigit()])
                        if len(view_masks) != len(saved_masks):
                            return f"❌ View {view_idx} has {len(view_masks)} masks, but view 1 has {len(saved_masks)}. All views must have the same number of masks.", "", ""
                else:
                    return "❌ Invalid multi-view data directory structure.", "", ""
            else:
                # Prepare training data with custom temp directory
                training_dir = self.prepare_training_data(main_image, mask_files, concept_names, temp_dir)
                if isinstance(training_dir, str) and training_dir.startswith("Error"):
                    return training_dir, "", ""

                # Count masks actually saved (fix #5)
                saved_masks = sorted(Path(training_dir).glob("mask*.png"))

            num_concepts = len(saved_masks)
            if num_concepts == 0:
                return "❌ No valid mask files were saved. Please upload at least one mask.", "", ""

            # Parse initializer tokens (fix #2)
            initializer_tokens = []
            if concept_names:
                initializer_tokens = [t.strip() for t in concept_names.split(",") if t.strip()]
            # Require exact match or omit the flag
            use_init_tokens = len(initializer_tokens) == num_concepts

            # Create output directory (fix #3) - use custom models_dir if provided
            timestamp = str(int(time.time()))
            if models_dir:
                output_dir = Path(models_dir) / f"model_{timestamp}"
            else:
                output_dir = self.models_dir / f"model_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # # Class data dir
            # class_data_dir = self.temp_dir / "class_data"
            # class_data_dir.mkdir(parents=True, exist_ok=True)

            # progress(0.1, desc="Starting training...")

            # # Build training command
            # cmd = [
            #     sys.executable, "train.py",
            #     "--instance_data_dir", str(training_dir),
            #     "--num_of_assets", str(num_concepts),
            #     "--class_data_dir", str(class_data_dir),
            #     "--phase1_train_steps", str(phase1_steps),
            #     "--phase2_train_steps", str(phase2_steps),
            #     "--output_dir", str(output_dir),
            #     "--use_8bit_adam",
            #     "--set_grads_to_none"
            # ]
            # if use_init_tokens:
            #     cmd.extend(["--initializer_tokens"] + initializer_tokens)


            # Require initializer tokens to match number of masks
            initializer_tokens = [t.strip() for t in concept_names.split(",") if t.strip()] if concept_names else []
            if len(initializer_tokens) != num_concepts:
                return (
                    f"❌ You provided {len(initializer_tokens)} initializer tokens but uploaded {num_concepts} masks. "
                    f"Provide exactly one token per mask (comma-separated).", "", ""
                )

            # Build training command using dreamedit3d.py with memory optimization
            # Use absolute paths for all directories
            dreamedit3d_script = self.base_dir / "dreamedit3d.py"

            # Detect if training_dir has multi-view structure
            training_path = Path(training_dir)
            view_dirs = sorted([d for d in training_path.iterdir() if d.is_dir() and d.name.startswith('view_')])
            is_multiview = len(view_dirs) >= 4

            cmd = [
                sys.executable, str(dreamedit3d_script),
                "--instance_data_dir", str(training_path.resolve()),
                "--output_dir", str(Path(output_dir).resolve()),
                "--num_of_assets", str(num_concepts),
                "--initializer_tokens", *initializer_tokens,
                "--concept_names", *initializer_tokens,  # Pass concept names for labeled mask lookup
                "--num_frames", "4" if is_multiview else "1",  # 4 views for multi-view, 1 for single
                "--phase1_train_steps", str(phase1_steps),
                "--phase2_train_steps", str(phase2_steps),
                "--train_batch_size", "1",  # Batch size 1 for memory efficiency
                "--learning_rate", "2e-6",
                "--initial_learning_rate", "5e-4",
                "--resolution", "256",
                "--size", "256",
                "--mixed_precision", "fp16",  # Use fp16 to reduce memory
                "--mvdream_training_mode", mvdream_training_mode,  # Use selected training mode
                "--img_log_steps", "100",
                "--use_8bit_adam",
                "--set_grads_to_none",
                "--model_name", "sd-v2.1-base-4view",
                "--camera_elev", "15",
                "--camera_azim", "90",
                "--camera_azim_span", "360",
                "--use_camera", "1",
                "--seed", "23",
                "--no_prior_preservation",  # Disable prior preservation to save memory
            ]

            # Only add view_mode and view_index for single-view training
            if not is_multiview:
                cmd.extend(["--view_mode", "single", "--view_index", "0"])



            progress(0.2, desc="Running training process...")

            # Log file + merge stderr into stdout (fix #4)
            log_path = output_dir / "train.log"
            
            # Set up environment with MKL fixes and memory optimization
            env = os.environ.copy()
            env['MKL_SERVICE_FORCE_INTEL'] = '1'
            env['MKL_THREADING_LAYER'] = 'GNU'
            env['OMP_NUM_THREADS'] = '1'  # Additional safety measure
            env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Additional compatibility
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Reduce memory fragmentation
            
            with open(log_path, "w", encoding="utf-8") as logf:
                logf.write("COMMAND:\n" + " ".join(cmd) + "\n")
                logf.write("ENVIRONMENT FIXES:\n")
                logf.write("MKL_SERVICE_FORCE_INTEL=1\n")
                logf.write("MKL_THREADING_LAYER=GNU\n")
                logf.write("OMP_NUM_THREADS=1\n\n")
                logf.flush()

                # process = subprocess.Popen(
                #     cmd,
                #     stdout=subprocess.PIPE,
                #     stderr=subprocess.STDOUT,  # merge
                #     text=True,
                #     bufsize=1,
                #     universal_newlines=True,
                #     env=env  # Pass the fixed environment
                # )
                repo_root = Path(__file__).resolve().parent  # ensure relative paths match CLI

                # Add PYTHONPATH to ensure proper module loading
                env['PYTHONPATH'] = str(repo_root)

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # merge
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env,
                    cwd=str(repo_root)
                )


                output_lines = []
                total_steps = phase1_steps + phase2_steps
                current_step = 0
                last_progress_update = 0

                # Stream output live, update progress
                import re
                for line in iter(process.stdout.readline, ''):
                    line = line.rstrip("\n")
                    # Keep only last 100 lines to prevent memory issues
                    output_lines.append(line)
                    if len(output_lines) > 100:
                        output_lines.pop(0)
                    logf.write(line + "\n")
                    logf.flush()  # Ensure logs are written immediately

                    # Parse actual step from progress bars like "Steps: 1%|█| 5/800 [00:01<04:05, 3.23it/s]"
                    step_match = re.search(r'Steps:.*?\|\s*(\d+)/(\d+)\s*\[', line)
                    if step_match:
                        try:
                            current_step = int(step_match.group(1))
                            total = int(step_match.group(2))
                            # Update progress only every 10 steps to reduce browser load
                            if current_step - last_progress_update >= 10 or current_step == total:
                                progress_val = min(0.2 + 0.7 * (current_step / max(total, 1)), 0.9)
                                progress(progress_val, desc=f"Training step {current_step}/{total}")
                                last_progress_update = current_step
                        except Exception:
                            pass

                process.stdout.close()
                return_code = process.wait()

            progress(1.0, desc="Training finished.")

            # Prepare a short tail of logs for the UI
            tail = "\n".join(output_lines[-20:]) if output_lines else "(no output)"
            if return_code == 0:
                # Save model info
                model_info = {
                    "model_path": str(output_dir),
                    "num_concepts": num_concepts,
                    "concept_names": initializer_tokens if use_init_tokens else [],
                    "timestamp": timestamp
                }
                with open(output_dir / "model_info.json", "w") as f:
                    json.dump(model_info, f, indent=2)

                success_msg = (
                    f"✅ Training completed successfully!\n"
                    f"Model saved to: {output_dir}\n"
                    f"Tokens: {', '.join([f'<asset{i}>' for i in range(num_concepts)])}\n"
                    f"Log: {log_path}"
                )
                return success_msg, tail, str(output_dir)
            else:
                error_msg = (
                    f"❌ Training failed with return code {return_code}\n"
                    f"See full log at: {log_path}"
                )
                return error_msg, tail, str(log_path)

        except Exception as e:
            return f"❌ Error during training: {str(e)}", "", ""

    
    def get_available_models(self):
        """Get list of trained models from legacy trained_models directory"""
        models = []
        # Only check if the legacy directory exists
        models_dir = self.base_dir / "trained_models"
        if models_dir.exists():
            for model_dir in models_dir.glob("model_*"):
                if model_dir.is_dir():
                    info_file = model_dir / "model_info.json"
                    if info_file.exists():
                        with open(info_file) as f:
                            info = json.load(f)
                        display_name = f"{model_dir.name} ({info.get('timestamp', 'unknown')})"
                        models.append((display_name, str(model_dir)))
                    else:
                        models.append((model_dir.name, str(model_dir)))
        return models
    
    def generate_image(self, model_path, prompt, output_filename,
                       negative_prompt="blurry, low quality, bad anatomy, distorted, ugly, noisy, artifacts, poorly rendered",
                       positive_prompt="high quality, detailed, sharp focus, professional lighting, good contrast, studio lighting, well-lit, crisp details",
                       steps=50, scale=7.5, camera_elev=15, camera_azim=90,
                       output_dir=None, size=256, num_frames=4, camera_azim_span=360,
                       elevation_list=None,
                       num_generations=1, seed=-1,
                       progress=gr.Progress()):
        """Generate image using trained model with customizable parameters"""
        try:
            if not model_path:
                return "❌ Please select a trained model", None

            if not prompt.strip():
                return "❌ Please enter a prompt", None

            progress(0.1, desc="Preparing generation...")

            # Use custom output directory if provided, otherwise use default
            if output_dir:
                generated_dir = Path(output_dir)
            else:
                generated_dir = self.base_dir / "generated_multiview"
            generated_dir.mkdir(exist_ok=True, parents=True)

            # Create output filename with timestamp
            import time
            timestamp = int(time.time())
            if output_filename:
                filename = f"{output_filename}_{timestamp}.jpg"
            else:
                filename = f"generated_multiview_{timestamp}.jpg"

            output_path = generated_dir / filename

            # Build inference command using absolute path
            inference_script = self.base_dir / "inference.py"
            cmd = [
                sys.executable, str(inference_script),
                "--model_path", model_path,
                "--prompt", prompt,
                "--output_path", str(output_path),
                "--size", str(size),
                "--num_frames", str(num_frames),
                "--steps", str(int(steps)),
                "--scale", str(float(scale)),
                "--camera_elev", str(int(camera_elev)),
                "--camera_azim", str(int(camera_azim)),
                "--camera_azim_span", str(camera_azim_span),
                "--negative_prompt", negative_prompt,
                "--positive_prompt", positive_prompt
            ]

            # Add elevation_list if provided
            if elevation_list:
                cmd.extend(["--elevation_list", elevation_list])

            # Add num_generations if more than 1
            if num_generations > 1:
                cmd.extend(["--num_generations", str(int(num_generations))])

            # Add seed if specified
            if seed >= 0:
                cmd.extend(["--seed", str(int(seed))])

            progress(0.3, desc=f"Generating {num_generations} image(s)...")
            
            # Set up environment with MKL fixes for inference too
            env = os.environ.copy()
            env['MKL_SERVICE_FORCE_INTEL'] = '1'
            env['MKL_THREADING_LAYER'] = 'GNU'
            env['OMP_NUM_THREADS'] = '1'
            env['PYTHONPATH'] = str(self.base_dir)

            # Run inference
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,  # Pass the fixed environment
                cwd=str(self.base_dir)  # Set working directory to project root
            )
            
            stdout, stderr = process.communicate()

            progress(1.0, desc="Generation completed!")

            if process.returncode == 0 and output_path.exists():
                if num_generations > 1:
                    return f"✅ Generated {num_generations} images successfully!\nSaved to: {output_path.parent}", str(output_path)
                else:
                    return f"✅ Image generated successfully!\nSaved to: {output_path}", str(output_path)
            else:
                error_msg = f"❌ Generation failed.\nStdout: {stdout}\nStderr: {stderr}"
                return error_msg, None
                
        except Exception as e:
            return f"❌ Error during generation: {str(e)}", None
    
    def refresh_models(self):
        """Refresh the list of available models"""
        return gr.Dropdown(choices=self.get_available_models())
    
    def load_example(self, example_name):
        """Load an example dataset"""
        if example_name not in self.examples:
            return None, [], "", []
        
        example = self.examples[example_name]
        main_image = example['main_image']
        mask_files = example['masks']
        
        # Generate concept names based on number of masks
        num_concepts = len(mask_files)
        concept_names = ", ".join([f"concept{i}" for i in range(num_concepts)])
        
        return main_image, mask_files, concept_names, mask_files

def create_app():
    app = DreamEdit3DApp()

    with gr.Blocks(title="DreamEdit3D: 3D-Aware Multi-Concept Extraction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎨 DreamEdit3D: 3D-Aware Multi-Concept Extraction from Single Images

        Extract multiple concepts from a single image with 3D consistency and generate multi-view images using MVDream.

        ## How to use:
        1. **Training Tab**: Upload your image and masks to train a new model
        2. **Generation Tab**: Use trained models to generate multi-view images with your extracted concepts

        ⚠️ **MKL Threading Fix Applied**: Environment variables set to resolve Intel MKL compatibility issues.
        """)
        
        with gr.Tabs():
            # Training Tab
            with gr.TabItem("🚀 Training", id="training"):
                gr.Markdown("### Step 1: Prepare Your Data")
                
                # Add example selector if examples are available
                if app.examples:
                    with gr.Row():
                        example_dropdown = gr.Dropdown(
                            label="Load Example (optional)",
                            choices=list(app.examples.keys()),
                            value=None
                        )
                        load_example_btn = gr.Button("Load Example")
                
                with gr.Row():
                    with gr.Column():
                        main_image = gr.Image(
                            label="Main Image",
                            type="filepath",
                            height=300
                        )
                        
                        concept_names = gr.Textbox(
                            label="Concept Names (optional)",
                            placeholder="e.g., creature, bowl, stone",
                            info="Comma-separated names for better initialization"
                        )
                    
                    with gr.Column():
                        mask_files = gr.File(
                            label="Mask Files",
                            file_count="multiple",
                            file_types=["image"],
                            height=300
                        )
                
                gr.Markdown("### Step 2: Training Parameters")
                with gr.Row():
                    phase1_steps = gr.Number(
                        label="Phase 1 Steps",
                        value=400,
                        minimum=100,
                        maximum=2000,
                        step=50
                    )
                    phase2_steps = gr.Number(
                        label="Phase 2 Steps", 
                        value=400,
                        minimum=100,
                        maximum=2000,
                        step=50
                    )
                
                train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
                
                with gr.Row():
                    training_status = gr.Textbox(label="Training Status", max_lines=5)
                    training_output = gr.Textbox(label="Training Output", max_lines=10)
                
                trained_model_path = gr.Textbox(label="Trained Model Path", visible=False)
                
                # Connect example loading if examples exist
                if app.examples:
                    load_example_btn.click(
                        fn=app.load_example,
                        inputs=[example_dropdown],
                        outputs=[main_image, mask_files, concept_names, gr.State()]
                    )
                
            # Generation Tab
            with gr.TabItem("🎨 Generation", id="generation"):
                gr.Markdown("### Generate Images with Your Trained Models")
                
                with gr.Row():
                    with gr.Column():
                        model_dropdown = gr.Dropdown(
                            label="Select Trained Model",
                            choices=app.get_available_models(),
                            interactive=True
                        )
                        
                        refresh_btn = gr.Button("🔄 Refresh Models")
                        
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="e.g., a photo of <asset0> at the beach",
                            info="Use <asset0>, <asset1>, etc. to reference your extracted concepts"
                        )
                        
                        output_filename = gr.Textbox(
                            label="Output Filename (optional)",
                            placeholder="generated_image"
                        )

                        generate_btn = gr.Button("🎨 Generate Image", variant="primary")

                        # Advanced Parameters
                        with gr.Accordion("⚙️ Advanced Parameters", open=False):
                            negative_prompt = gr.Textbox(
                                label="Negative Prompt",
                                value="blurry, low quality, bad anatomy, distorted, ugly, noisy, artifacts, poorly rendered",
                                placeholder="Features to avoid in generation",
                                info="Helps improve image quality by avoiding unwanted features"
                            )
                            with gr.Row():
                                gen_steps = gr.Slider(
                                    label="Diffusion Steps",
                                    minimum=20,
                                    maximum=100,
                                    value=50,
                                    step=5,
                                    info="More steps = better quality but slower"
                                )
                                cfg_scale = gr.Slider(
                                    label="CFG Scale",
                                    minimum=3.0,
                                    maximum=15.0,
                                    value=7.5,
                                    step=0.5,
                                    info="Higher = stronger prompt adherence"
                                )
                            with gr.Row():
                                camera_elev = gr.Slider(
                                    label="Camera Elevation",
                                    minimum=-30,
                                    maximum=60,
                                    value=15,
                                    step=5
                                )
                                camera_azim = gr.Slider(
                                    label="Camera Azimuth Start",
                                    minimum=0,
                                    maximum=360,
                                    value=90,
                                    step=15
                                )

                    with gr.Column():
                        generated_image = gr.Image(label="Generated Image", height=400)
                        generation_status = gr.Textbox(label="Generation Status")
        
        # Event handlers
        train_btn.click(
            fn=app.train_model,
            inputs=[main_image, mask_files, concept_names, phase1_steps, phase2_steps],
            outputs=[training_status, training_output, trained_model_path]
        )
        
        refresh_btn.click(
            fn=app.refresh_models,
            outputs=[model_dropdown]
        )
        
        generate_btn.click(
            fn=app.generate_image,
            inputs=[model_dropdown, prompt_input, output_filename,
                    negative_prompt, gen_steps, cfg_scale, camera_elev, camera_azim],
            outputs=[generation_status, generated_image]
        )
        
        # Examples
        with gr.Accordion("💡 Examples & Usage", open=False):
            examples_text = "- No examples found in the examples/ directory"
            if app.examples:
                examples_text = "\n".join([f"- **{name}**: {data['num_concepts']} concepts" for name, data in app.examples.items()])

            gr.Markdown(f"""
            ### Available Examples:
            {examples_text}

            ### Training Example:
            1. **Use Example**: Select from available examples in the dropdown
            2. **Upload Custom**: Upload your own image and corresponding masks
            3. Optionally provide concept names like "creature, bowl, stone"
            4. Click "Start Training" and wait for completion

            ### Generation Examples (Multi-View Output):
            - `"a photo of <asset0> at the beach"` → Generates 4 views (front, right, back, left)
            - `"an oil painting of <asset1> and <asset2>"`
            - `"<asset0> in a fantasy landscape"`
            - `"a sketch of <asset1> on white background"`

            ### File Structure for Custom Upload:
            - Main image: Single image containing multiple concepts
            - Mask files: One mask per concept (mask0.png, mask1.png, etc.)
            - Each mask should highlight one specific concept in white, background in black

            ### 3D-Aware Generation:
            DreamEdit3D uses MVDream to generate 4 multi-view images with 3D consistency,
            suitable for 3D reconstruction pipelines.

            ### MKL Threading Fix Applied:
            - `MKL_SERVICE_FORCE_INTEL=1`
            - `MKL_THREADING_LAYER=GNU`
            - `OMP_NUM_THREADS=1`
            """)
    
    return demo

if __name__ == "__main__":
    # Check if required files exist
    required_files = ["dreamedit3d.py", "inference.py"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("Please make sure you're running this in the DreamEdit3D repository directory.")
        sys.exit(1)
    
    demo = create_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,  # Let Gradio find an available port
        share=False,
        inbrowser=True,
        debug=True
    )