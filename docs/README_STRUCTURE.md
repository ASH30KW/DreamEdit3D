# DreamEdit3D App Structure

## File Organization

### Main Application
- **dreamedit3d_app.py** - Main application at root level (NEW)
  - All paths are relative to DreamEdit3D root directory
  - Unified entry point for the complete pipeline

### Original Files
- **mask/main.py** - Original file (kept for backward compatibility)
  - Can be deprecated in favor of dreamedit3d_app.py

### Data Directories

#### Input Data
- **mask/race-chicken-mv/** - Default multi-view images for testing
- **examples/** - Example images and prepared data for GTR
- **generated_multiview/** - Output from DreamEdit3D training

#### Output Data
- **mask/masked_output/** - SAM segmentation results (organized by timestamp)
  - Each folder contains view_1/, view_2/, etc. with img.jpg and mask0.png
- **temp_workspace/** - Temporary files during DreamEdit3D training
- **trained_models/** - Trained DreamEdit3D model checkpoints
- **temp_gtr_gradio/** - Temporary files for GTR pipeline
  - prepared_mv/ - Preprocessed multi-view data
  - inference_output/ - Generated 3D meshes and renders

#### Model Checkpoints
- **snap_gtr/ckpts/** - GTR model checkpoints
- **checkpoints/** - SAM model checkpoints

## Path Strategy

All paths in `dreamedit3d_app.py` are relative to the DreamEdit3D root:
- `base_dir = Path(__file__).parent` always points to DreamEdit3D root
- Examples: `"mask/race-chicken-mv"`, `"snap_gtr/ckpts/full_checkpoint.pth"`
- Benefits: Portable, no hardcoded absolute paths, easier to deploy

## Running the Application

```bash
cd /home/ai/gr/DreamEdit3D
python dreamedit3d_app.py
```

Access at: http://localhost:7860
