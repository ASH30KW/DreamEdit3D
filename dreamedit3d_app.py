import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
import torch
import matplotlib.pyplot as plt
import io
import base64
import os
import glob
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import time
import trimesh
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import DreamEdit3DApp from same directory
from gradio_app import DreamEdit3DApp

# Import GPT-4V object detector for automatic concept naming
from utils.gpt_object_detector import detect_concept_names_batch

# SAM imports
try:
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: segment_anything not installed. Install with: pip install segment-anything")
    SAM_AVAILABLE = False

# Evaluation imports
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "evaluation", "metrics"))
    from clip_metrics import CLIPEditEvaluator
    CLIP_EVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CLIP evaluation not available: {e}")
    CLIP_EVAL_AVAILABLE = False

try:
    from gpt4v_eval import evaluate_3d_generation
    GPTEVAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GPT-4V evaluation not available: {e}")
    GPTEVAL_AVAILABLE = False

try:
    from clip_iqa import CLIPIQAEvaluator, DEFAULT_PROMPT_PAIRS
    CLIP_IQA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CLIP-IQA evaluation not available: {e}")
    CLIP_IQA_AVAILABLE = False


# ===== Evaluation System =====

class EvaluationManager:
    """Manages evaluation of 3D models using CLIP, CLIP-IQA, and GPT-4V metrics"""

    def __init__(self):
        self.clip_evaluator = None
        self.clip_iqa_evaluator = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def init_clip_iqa_evaluator(self):
        """Initialize CLIP-IQA evaluator (lazy loading)"""
        if self.clip_iqa_evaluator is None and CLIP_IQA_AVAILABLE:
            try:
                self.clip_iqa_evaluator = CLIPIQAEvaluator(device=self.device)
                return True, "CLIP-IQA evaluator initialized"
            except Exception as e:
                return False, f"Failed to initialize CLIP-IQA: {e}"
        return self.clip_iqa_evaluator is not None, "CLIP-IQA evaluator ready" if self.clip_iqa_evaluator else "CLIP-IQA not available"

    def run_clip_iqa_evaluation(self, renders_dir, prompt_names=None):
        """Run CLIP-IQA no-reference image quality assessment"""
        if not CLIP_IQA_AVAILABLE:
            return None, "CLIP-IQA evaluation not available"

        success, msg = self.init_clip_iqa_evaluator()
        if not success:
            return None, msg

        try:
            if prompt_names is None:
                prompt_names = ["quality", "sharpness", "aesthetic", "render_quality"]

            per_image = self.clip_iqa_evaluator.score_directory(
                renders_dir,
                prompt_names=prompt_names,
                verbose=False
            )
            aggregated = self.clip_iqa_evaluator.aggregate_scores(per_image)

            results = {
                "per_image": per_image,
                "aggregated": aggregated
            }
            return results, "CLIP-IQA evaluation complete"
        except Exception as e:
            return None, f"CLIP-IQA evaluation error: {e}"

    def format_clip_iqa_results(self, results):
        """Format CLIP-IQA results for display"""
        if results is None:
            return "No results"

        lines = ["## CLIP-IQA Quality Assessment\n"]
        lines.append("_No-reference image quality scores (0-1, higher is better)_\n")

        aggregated = results.get("aggregated", {})

        # Group by quality dimension
        dimensions = set()
        for key in aggregated.keys():
            if key.endswith("_mean"):
                dimensions.add(key.replace("_mean", ""))

        lines.append("### Aggregated Scores\n")
        for dim in sorted(dimensions):
            mean_val = aggregated.get(f"{dim}_mean", 0)
            std_val = aggregated.get(f"{dim}_std", 0)
            # Create a visual bar
            bar_filled = int(mean_val * 20)
            bar = '█' * bar_filled + '░' * (20 - bar_filled)
            lines.append(f"- **{dim}**: {mean_val:.3f} ± {std_val:.3f} `{bar}`")

        # Show per-image scores in a collapsible section hint
        per_image = results.get("per_image", {})
        if per_image:
            lines.append(f"\n_Scored {len(per_image)} images_")

        return "\n".join(lines)

    def init_clip_evaluator(self):
        """Initialize CLIP evaluator (lazy loading)"""
        if self.clip_evaluator is None and CLIP_EVAL_AVAILABLE:
            try:
                self.clip_evaluator = CLIPEditEvaluator(device=self.device)
                return True, "CLIP evaluator initialized"
            except Exception as e:
                return False, f"Failed to initialize CLIP: {e}"
        return self.clip_evaluator is not None, "CLIP evaluator ready" if self.clip_evaluator else "CLIP not available"

    def render_mesh_for_eval(self, mesh_path, output_dir, num_views=70, progress_callback=None,
                             do_normalize=True, object_scale=1.0):
        """Render mesh to multiple views for evaluation"""
        try:
            project_root = os.path.dirname(__file__)
            render_script = os.path.join(project_root, "evaluation", "rendering", "render_views.py")

            if not os.path.exists(render_script):
                return False, f"Render script not found: {render_script}"

            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                sys.executable, render_script,
                "--mesh_path", mesh_path,
                "--output_dir", output_dir,
                "--num_views", str(num_views),
                "--resolution", "512",
                "--object_scale", str(float(object_scale)),
            ]
            if not do_normalize:
                cmd.append("--no_normalize")

            if progress_callback:
                progress_callback(f"Rendering {num_views} views...")

            # Run from project root so path resolution works correctly
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=project_root)

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                return False, f"Render failed: {error_msg}"

            # Check output
            rgb_files = list(Path(output_dir).glob("rgb_*.png"))
            if len(rgb_files) == 0:
                return False, "No images rendered"

            return True, f"Rendered {len(rgb_files)} views to {output_dir}"

        except subprocess.TimeoutExpired:
            return False, "Rendering timed out (>10 min)"
        except Exception as e:
            return False, f"Render error: {e}"

    def run_clip_evaluation(self, input_renders_dir, edit_renders_dir, text_input, text_edit,
                           text_edited_word=None, text_generic=None):
        """Run CLIP-based evaluation"""
        if not CLIP_EVAL_AVAILABLE:
            return None, "CLIP evaluation not available"

        success, msg = self.init_clip_evaluator()
        if not success:
            return None, msg

        try:
            results = self.clip_evaluator.evaluate_edit(
                input_renders_dir=input_renders_dir,
                edit_renders_dir=edit_renders_dir,
                text_input=text_input,
                text_edit=text_edit,
                text_edited_word=text_edited_word,
                text_generic=text_generic
            )
            return results, "CLIP evaluation complete"
        except Exception as e:
            return None, f"CLIP evaluation error: {e}"

    def run_gpteval(self, renders_dir, prompt_text, output_file):
        """Run GPT-4V based evaluation"""
        if not GPTEVAL_AVAILABLE:
            return None, "GPT-4V evaluation not available"

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None, "OPENAI_API_KEY not set in environment"

        try:
            results = evaluate_3d_generation(
                api_key=api_key,
                renders_dir=renders_dir,
                prompt_text=prompt_text,
                output_file=output_file
            )
            return results, "GPT-4V evaluation complete"
        except Exception as e:
            return None, f"GPT-4V evaluation error: {e}"

    def get_renders_from_mesh(self, mesh_path, project_path=None):
        """Check if renders exist, return renders directory"""
        # Primary location: 05_evaluation/renders/
        if project_path:
            eval_render_dir = Path(project_path) / "05_evaluation" / "renders"
            if eval_render_dir.exists():
                rgb_files = list(eval_render_dir.glob("rgb_*.png"))
                if len(rgb_files) > 0:
                    return str(eval_render_dir), len(rgb_files)

        # Fallback: check near mesh file (legacy)
        mesh_dir = Path(mesh_path).parent
        possible_render_dirs = [
            mesh_dir / "renders",
            mesh_dir / "eval_renders",
            mesh_dir.parent / "renders"
        ]

        for render_dir in possible_render_dirs:
            if render_dir.exists():
                rgb_files = list(render_dir.glob("rgb_*.png"))
                if len(rgb_files) > 0:
                    return str(render_dir), len(rgb_files)

        return None, 0

    def format_clip_results(self, results):
        """Format CLIP results for display"""
        if results is None:
            return "No results"

        lines = ["## CLIP Evaluation Results\n"]

        # CLIP Score (image-text similarity)
        if "CLIP_score_input" in results or "CLIP_score_edit" in results:
            lines.append("### CLIP Score (Higher is better)")
            if "CLIP_score_input" in results:
                val = results["CLIP_score_input"]
                lines.append(f"- **Input vs Original Text**: {val:.2f}")
            if "CLIP_score_edit" in results:
                val = results["CLIP_score_edit"]
                lines.append(f"- **Edited vs Edited Text**: {val:.2f}")
            lines.append("")

        lines.append("### Directional Metrics (Higher is better)")

        for key in ["CLIP_dir", "CLIP_dir-cos", "CLIP_dir-avg", "CLIP_dir-avg-cos"]:
            if key in results:
                val = results[key]
                lines.append(f"- **{key}**: {val:.4f} (x100: {val*100:.2f})")

        if "CLIP_diff-edit" in results or "CLIP_diff-noedit" in results:
            lines.append("\n### Difference Metrics (Lower is better)")
            for key in ["CLIP_diff-edit", "CLIP_diff-noedit"]:
                if key in results:
                    val = results[key]
                    lines.append(f"- **{key}**: {val:.4f} (x100: {val*100:.2f})")

        return "\n".join(lines)

    def format_gpteval_results(self, results):
        """Format GPT-4V results for display"""
        if results is None:
            return "No results"

        if "raw_response" in results:
            return f"## GPT-4V Response\n\n{results['raw_response']}"

        lines = ["## GPT-4V Evaluation Results\n"]

        dimensions = [
            ("text_3d_alignment", "Text-3D Alignment"),
            ("visual_quality", "Visual Quality"),
            ("3d_consistency", "3D Consistency"),
            ("completeness", "Completeness"),
            ("overall_quality", "Overall Quality")
        ]

        for key, label in dimensions:
            if key in results:
                score = results[key].get("score", "N/A")
                explanation = results[key].get("explanation", "")
                lines.append(f"### {label}: {score}/10")
                lines.append(f"_{explanation}_\n")

        if "summary" in results:
            lines.append("### Summary")
            if "strengths" in results["summary"]:
                lines.append("**Strengths:**")
                for s in results["summary"]["strengths"]:
                    lines.append(f"- {s}")
            if "weaknesses" in results["summary"]:
                lines.append("\n**Weaknesses:**")
                for w in results["summary"]["weaknesses"]:
                    lines.append(f"- {w}")

        return "\n".join(lines)

    def collect_all_gpteval_results(self, project_path):
        """Collect all GPT-4V evaluation results from a project"""
        eval_base = Path(project_path) / "05_evaluation"
        if not eval_base.exists():
            return []

        results = []
        for eval_dir in eval_base.iterdir():
            if eval_dir.is_dir():
                gpteval_file = eval_dir / "gpteval_results.json"
                summary_file = eval_dir / "eval_summary.json"

                if gpteval_file.exists():
                    import json
                    with open(gpteval_file, 'r') as f:
                        gpt_results = json.load(f)

                    # Get prompt from summary if available
                    prompt = ""
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            summary = json.load(f)
                            prompt = summary.get("prompts", {}).get("gpt_prompt", "")

                    results.append({
                        "mesh_name": eval_dir.name,
                        "prompt": prompt,
                        "results": gpt_results
                    })

        return results

    def calculate_average_scores(self, all_results):
        """Calculate average scores from all GPT-4V results"""
        dimensions = ["text_3d_alignment", "visual_quality", "3d_consistency", "completeness", "overall_quality"]
        totals = {d: [] for d in dimensions}

        for entry in all_results:
            results = entry.get("results", {})
            for dim in dimensions:
                if dim in results and isinstance(results[dim], dict):
                    score = results[dim].get("score")
                    if isinstance(score, (int, float)):
                        totals[dim].append(score)

        averages = {}
        for dim in dimensions:
            if totals[dim]:
                averages[dim] = sum(totals[dim]) / len(totals[dim])
            else:
                averages[dim] = None

        return averages

    def generate_gpteval_report(self, project_path):
        """Generate a PDF report of all GPT-4V evaluations"""
        try:
            from fpdf import FPDF
            from fpdf.enums import XPos, YPos
        except ImportError:
            return None, "fpdf2 not installed. Run: pip install fpdf2"

        import json
        from datetime import datetime

        # Collect all results
        all_results = self.collect_all_gpteval_results(project_path)
        if not all_results:
            return None, "No GPT-4V evaluation results found"

        # Calculate averages
        averages = self.calculate_average_scores(all_results)

        # Get project name
        project_name = Path(project_path).name

        # Create PDF with better margins
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)

        dimension_labels = {
            "text_3d_alignment": "Text-3D Alignment",
            "visual_quality": "Visual Quality",
            "3d_consistency": "3D Consistency",
            "completeness": "Completeness",
            "overall_quality": "Overall Quality"
        }

        # Title page
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.cell(0, 15, "GPT-4V Evaluation Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Project: {project_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.cell(0, 8, f"Total Meshes Evaluated: {len(all_results)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        pdf.ln(10)

        # Summary section
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Average Scores", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_draw_color(100, 100, 100)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        for dim, label in dimension_labels.items():
            avg = averages.get(dim)
            if avg is not None:
                pdf.cell(80, 7, f"{label}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, f"{avg:.1f}/10", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", "", 11)
            else:
                pdf.cell(80, 7, f"{label}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
                pdf.cell(0, 7, "N/A", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.ln(10)

        # Individual results
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Individual Results", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.line(15, pdf.get_y(), 195, pdf.get_y())
        pdf.ln(8)

        for i, entry in enumerate(all_results, 1):
            mesh_name = entry.get("mesh_name", "Unknown")
            prompt = entry.get("prompt", "")
            results = entry.get("results", {})

            # Check if we need a new page (leave room for at least one result)
            if pdf.get_y() > 200:
                pdf.add_page()

            # Mesh header with background
            pdf.set_fill_color(240, 240, 240)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 8, f"{i}. {mesh_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

            # Prompt
            if prompt:
                pdf.set_font("Helvetica", "I", 9)
                pdf.set_text_color(80, 80, 80)
                pdf.cell(0, 6, f"Prompt: {prompt}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_text_color(0, 0, 0)

            pdf.ln(2)

            # Scores and explanations
            for dim, label in dimension_labels.items():
                if dim in results and isinstance(results[dim], dict):
                    score = results[dim].get("score", "N/A")
                    explanation = results[dim].get("explanation", "")

                    # Score line
                    pdf.set_font("Helvetica", "B", 10)
                    pdf.cell(50, 6, f"{label}:", new_x=XPos.RIGHT, new_y=YPos.TOP)
                    pdf.set_font("Helvetica", "", 10)
                    pdf.cell(0, 6, f"{score}/10", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

                    # Explanation (with proper wrapping)
                    if explanation:
                        pdf.set_font("Helvetica", "I", 8)
                        pdf.set_text_color(60, 60, 60)
                        # Use multi_cell for proper text wrapping
                        pdf.set_x(20)  # Indent
                        pdf.multi_cell(170, 4, explanation)
                        pdf.set_text_color(0, 0, 0)
                        pdf.ln(1)

            pdf.ln(8)

        # Save PDF
        eval_base = Path(project_path) / "05_evaluation"
        eval_base.mkdir(parents=True, exist_ok=True)
        pdf_path = eval_base / "gpteval_report.pdf"
        pdf.output(str(pdf_path))

        # Save summary JSON
        summary_data = {
            "generated_at": datetime.now().isoformat(),
            "project": project_name,
            "total_meshes": len(all_results),
            "average_scores": averages,
            "individual_results": [
                {
                    "mesh_name": e["mesh_name"],
                    "prompt": e["prompt"],
                    "scores": {
                        dim: e["results"].get(dim, {}).get("score")
                        for dim in dimension_labels.keys()
                        if dim in e["results"]
                    }
                }
                for e in all_results
            ]
        }

        summary_path = eval_base / "gpteval_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)

        return str(pdf_path), f"Report generated: {pdf_path}"

    def format_average_scores(self, project_path):
        """Format average scores for GUI display"""
        all_results = self.collect_all_gpteval_results(project_path)
        if not all_results:
            return "No GPT-4V results found"

        averages = self.calculate_average_scores(all_results)

        dimension_labels = {
            "text_3d_alignment": "Text-3D",
            "visual_quality": "Visual",
            "3d_consistency": "3D Consist",
            "completeness": "Complete",
            "overall_quality": "Overall"
        }

        parts = []
        for dim, label in dimension_labels.items():
            avg = averages.get(dim)
            if avg is not None:
                parts.append(f"{label}: {avg:.1f}")

        if not parts:
            return "No scores available"

        return f"**Average Scores ({len(all_results)} meshes):** " + " | ".join(parts)


# Global evaluation manager
eval_manager = EvaluationManager()


class SAMMaskGenerator:
    def __init__(self):
        self.predictor = None
        self.mask_generator = None
        self.current_image = None
        self.points = []
        self.labels = []  # 1 for positive, 0 for negative
        
    def load_sam_model(self, model_type='vit_h'):
        """Load SAM model"""
        if not SAM_AVAILABLE:
            return False, "Segment Anything not installed"
        
        try:
            # Define checkpoint paths for different model sizes (relative to app root)
            base_dir = Path(__file__).parent
            checkpoints = {
                'vit_h': base_dir / 'mask/checkpoints/sam_vit_h_4b8939.pth',  # Largest, most accurate
                'vit_l': base_dir / 'mask/checkpoints/sam_vit_l_0b3195.pth',  # Medium
                'vit_b': base_dir / 'mask/checkpoints/sam_vit_b_01ec64.pth'   # Smallest, fastest
            }

            checkpoint_path = checkpoints.get(model_type, checkpoints['vit_h'])

            # Try to load the model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
            sam.to(device=device)
            
            self.predictor = SamPredictor(sam)
            self.mask_generator = SamAutomaticMaskGenerator(sam)
            
            return True, f"SAM model loaded successfully on {device}"
            
        except Exception as e:
            return False, f"Error loading SAM: {str(e)}"
    
    def set_image(self, image):
        """Set image for SAM predictor"""
        if self.predictor is None:
            return False, "SAM model not loaded"
        
        try:
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            self.current_image = image_array
            self.predictor.set_image(image_array)
            self.points = []  # Reset points when new image is set
            self.labels = []
            return True, "Image set successfully"
            
        except Exception as e:
            return False, f"Error setting image: {str(e)}"
    
    def add_point(self, x, y, label=1):
        """Add a point for segmentation"""
        self.points.append([x, y])
        self.labels.append(label)
    
    def clear_points(self):
        """Clear all points"""
        self.points = []
        self.labels = []
    
    def predict_from_points(self, points=None, labels=None):
        """Generate mask from point prompts"""
        if self.predictor is None:
            return None, None, "SAM model not loaded"
        
        try:
            if points is None:
                points = self.points
                labels = self.labels
            
            if len(points) == 0:
                return None, None, "No points provided"
            
            input_points = np.array(points)
            input_labels = np.array(labels) if labels else np.ones(len(points))
            
            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            # Return best mask (highest score)
            best_mask_idx = np.argmax(scores)
            return masks[best_mask_idx], scores[best_mask_idx], "Success"
            
        except Exception as e:
            return None, None, f"Error in prediction: {str(e)}"
    
    def predict_from_box(self, box):
        """Generate mask from bounding box"""
        if self.predictor is None:
            return None, None, "SAM model not loaded"
        
        try:
            input_box = np.array(box)
            
            masks, scores, logits = self.predictor.predict(
                box=input_box,
                multimask_output=True,
            )
            
            best_mask_idx = np.argmax(scores)
            return masks[best_mask_idx], scores[best_mask_idx], "Success"
            
        except Exception as e:
            return None, None, f"Error in prediction: {str(e)}"
    
    def generate_everything_mask(self, image):
        """Generate masks for everything in the image"""
        if self.mask_generator is None:
            return None, "SAM model not loaded"
        
        try:
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            masks = self.mask_generator.generate(image_array)
            return masks, "Success"
            
        except Exception as e:
            return None, f"Error generating masks: {str(e)}"

# Global SAM instance
sam_instance = SAMMaskGenerator()

# Global variables for storing points and boxes
current_points = []
current_points_view1 = []
current_points_view2 = []
current_points_view3 = []
current_points_view4 = []
current_box = None
current_box_view1 = None
current_box_view2 = None
current_box_view3 = None
current_box_view4 = None
box_clicks_view1 = []
box_clicks_view2 = []
box_clicks_view3 = []
box_clicks_view4 = []

# Load default multi-view images from renders directory
DEFAULT_RENDERS_BASE_DIR = os.path.join(os.path.dirname(__file__), "evaluation/data/renders")
# Default object to load (can be changed to any object UID in the renders directory)
DEFAULT_OBJECT_UID = "7fb196a40a6e4697aad9ca2f75c8b33d"
DEFAULT_IMAGE_DIR = os.path.join(DEFAULT_RENDERS_BASE_DIR, DEFAULT_OBJECT_UID)

# ===== Project Management System =====

class ProjectManager:
    """Manages unified project folder structure for the complete pipeline"""

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.projects_dir = self.base_dir / "projects"
        self.projects_dir.mkdir(exist_ok=True)

    def create_project(self, project_name=None):
        """Create a new project with timestamped folder structure"""
        import time
        timestamp = time.strftime("%Y_%m_%d_%H%M%S")

        if project_name:
            # Sanitize project name
            project_name = "".join(c for c in project_name if c.isalnum() or c in ('-', '_'))
            folder_name = f"{project_name}_{timestamp}"
        else:
            folder_name = f"project_{timestamp}"

        project_path = self.projects_dir / folder_name

        # Create project subdirectories
        (project_path / "01_sam_masks").mkdir(parents=True, exist_ok=True)
        (project_path / "02_trained_model").mkdir(parents=True, exist_ok=True)
        (project_path / "03_multiview_images").mkdir(parents=True, exist_ok=True)
        (project_path / "04_gtr_3d").mkdir(parents=True, exist_ok=True)
        (project_path / "05_evaluation").mkdir(parents=True, exist_ok=True)

        # Create project info file
        info = {
            "created": timestamp,
            "name": project_name or folder_name,
            "steps_completed": []
        }
        import json
        with open(project_path / "project_info.json", 'w') as f:
            json.dump(info, f, indent=2)

        return str(project_path)

    def get_all_projects(self):
        """Get list of all projects"""
        if not self.projects_dir.exists():
            return []
        projects = [str(p) for p in self.projects_dir.iterdir() if p.is_dir()]
        return sorted(projects, reverse=True)  # Most recent first

    def get_project_info(self, project_path):
        """Get project information"""
        import json
        info_file = Path(project_path) / "project_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                return json.load(f)
        return None

    def update_project_step(self, project_path, step_name):
        """Mark a step as completed in project"""
        import json
        info_file = Path(project_path) / "project_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                info = json.load(f)
            if step_name not in info.get("steps_completed", []):
                info.setdefault("steps_completed", []).append(step_name)
            with open(info_file, 'w') as f:
                json.dump(info, f, indent=2)

    def save_prompts(self, project_path, prompt_type, prompts, metadata=None):
        """Save prompts used in training/inference to project_info.json

        Args:
            project_path: Path to project folder
            prompt_type: 'training' or 'inference'
            prompts: List of prompt strings or single prompt
            metadata: Optional dict with additional info (e.g., model_path, timestamp)
        """
        import json
        from datetime import datetime

        info_file = Path(project_path) / "project_info.json"
        if not info_file.exists():
            return

        with open(info_file, 'r') as f:
            info = json.load(f)

        # Initialize prompts section if not exists
        if "prompts" not in info:
            info["prompts"] = {"training": [], "inference": []}

        # Ensure prompt_type section exists
        if prompt_type not in info["prompts"]:
            info["prompts"][prompt_type] = []

        # Create prompt entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompts": prompts if isinstance(prompts, list) else [prompts]
        }
        if metadata:
            entry.update(metadata)

        info["prompts"][prompt_type].append(entry)

        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

    def get_latest_prompts(self, project_path):
        """Get the most recent prompts from project_info.json"""
        info = self.get_project_info(project_path)
        if not info or "prompts" not in info:
            return None, None

        training_prompts = None
        inference_prompts = None

        if info["prompts"].get("training"):
            latest = info["prompts"]["training"][-1]
            training_prompts = latest.get("prompts", [])

        if info["prompts"].get("inference"):
            latest = info["prompts"]["inference"][-1]
            inference_prompts = latest.get("prompts", [])

        return training_prompts, inference_prompts

# Global project manager
project_manager = None

# ===== Interactive 3D Mesh Visualizer =====

class MeshVisualizer:
    """
    Interactive 3D mesh visualization using Google model-viewer.
    Supports loading and displaying meshes from project directories.
    GLB and HTML files are saved alongside the original mesh files.
    """

    def __init__(self):
        pass

    def load_mesh_from_file(self, mesh_path):
        """Load mesh from file (supports .obj, .ply, .glb, etc.)"""
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            return mesh, f"Loaded mesh: {Path(mesh_path).name}"
        except Exception as e:
            return None, f"Error loading mesh: {str(e)}"

    def save_mesh_as_glb(self, mesh, output_path, apply_rotation=False, enhance_colors=True):
        """Save mesh as GLB file for model-viewer with texture preservation."""
        try:
            # Optional rotation fix (disabled by default to preserve GTR orientation)
            if apply_rotation:
                R = trimesh.transformations.rotation_matrix(np.deg2rad(-90), [1, 0, 0])
                mesh.apply_transform(R)

            # Check if mesh already has vertex colors (from OBJ file)
            has_vertex_colors = False
            if hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors
                # Check if colors are not all the same (default)
                if len(vertex_colors) > 0 and not np.all(vertex_colors[0] == vertex_colors):
                    has_vertex_colors = True
                    print(f"✓ Mesh has vertex colors, preserving texture")

                    # Enhance vertex colors for better visualization
                    if enhance_colors:
                        # Increase saturation and brightness slightly
                        colors_rgb = vertex_colors[:, :3].astype(np.float32)
                        # Normalize to 0-1
                        colors_rgb = colors_rgb / 255.0
                        # Increase brightness by 10% and clamp
                        colors_rgb = np.clip(colors_rgb * 1.1, 0, 1.0)
                        # Convert back to 0-255
                        vertex_colors[:, :3] = (colors_rgb * 255).astype(np.uint8)
                        mesh.visual.vertex_colors = vertex_colors
                        print(f"✓ Enhanced vertex colors for better visualization")

            # Only set default color if no texture exists
            if not has_vertex_colors:
                print(f"⚠ No vertex colors found, applying default skin color")
                color = (210, 180, 140, 255)
                n_vertices = len(mesh.vertices)
                colors = np.tile(color, (n_vertices, 1))
                mesh.visual.vertex_colors = colors

            mesh.export(output_path)
            return output_path
        except Exception as e:
            print(f"Error saving GLB: {e}")
            return None

    def create_model_viewer_html_base64(self, glb_path, title="3D Mesh"):
        """Create HTML with Google model-viewer using base64 GLB data."""
        try:
            with open(glb_path, 'rb') as f:
                glb_data = f.read()
            glb_base64 = base64.b64encode(glb_data).decode('utf-8')
            src_url = f"data:model/gltf-binary;base64,{glb_base64}"

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <title>{title}</title>
                <script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.1.1/dist/model-viewer.min.js"></script>
                <style>
                    body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
                    model-viewer {{
                        width: 100%;
                        height: 800px;
                        background-color: #f0f0f0;
                    }}
                    .info {{
                        position: absolute;
                        top: 10px;
                        left: 10px;
                        background: rgba(0,0,0,0.8);
                        color: white;
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 14px;
                        z-index: 100;
                        pointer-events: none;
                    }}
                </style>
            </head>
            <body>
                <div style="position: relative;">
                    <div class="info">{title}</div>
                    <model-viewer
                        src="{src_url}"
                        alt="{title}"
                        auto-rotate
                        camera-controls
                        shadow-intensity="0.5"
                        exposure="1.2"
                        tone-mapping="commerce"
                        environment-image="neutral"
                        background-color="#f0f0f0"
                        disable-zoom
                        style="width: 100%; height: 800px;">
                    </model-viewer>
                </div>
            </body>
            </html>
            """
            return html_content
        except Exception as e:
            print(f"Error creating model viewer HTML: {e}")
            return f"<div style='color:red; text-align:center; padding:50px;'>Error: {str(e)}</div>"

    def create_interactive_visualization(self, mesh_path):
        """Create interactive 3D visualization using model-viewer with iframe."""
        try:
            # Load mesh
            mesh, load_msg = self.load_mesh_from_file(mesh_path)

            if mesh is None:
                return f"<div style='color:red; text-align:center; padding:50px;'>⚠️ {load_msg}</div>", load_msg

            # Save GLB in the same directory as the mesh (e.g., projects/project_xxx/04_gtr_3d/...)
            mesh_path_obj = Path(mesh_path)
            output_dir = mesh_path_obj.parent

            # Extract timestamp from folder name (e.g., "generated_multiview_1761666524" -> "1761666524")
            folder_name = output_dir.name
            if '_' in folder_name:
                timestamp = folder_name.split('_')[-1]
            else:
                timestamp = str(int(time.time() * 1000))

            glb_filename = f"mesh_{timestamp}.glb"
            glb_path = output_dir / glb_filename

            if not self.save_mesh_as_glb(mesh, str(glb_path)):
                return "<div style='color:red; text-align:center; padding:50px;'>⚠️ Failed to save GLB</div>", "Export failed"

            # Create model-viewer HTML file
            title = f"3D Mesh: {Path(mesh_path).name}"
            html_content = self.create_model_viewer_html_base64(str(glb_path), title)

            # Save HTML file for iframe serving
            html_path = glb_path.with_suffix('.html')
            with open(html_path, 'w') as f:
                f.write(html_content)

            # Create iframe HTML that references the static file
            # Path relative to projects directory (e.g., "project_xxx/04_gtr_3d/generated_multiview_xxx/mesh_xxx.html")
            projects_dir = Path.cwd() / "projects"
            relative_path = html_path.relative_to(projects_dir)
            iframe_html = f'<iframe src="/mesh_static/{relative_path}" width="100%" height="820px" style="border:none; border-radius: 8px;"></iframe>'

            return iframe_html, f"✓ Loaded: {Path(mesh_path).name}\n📁 Saved: {glb_path.relative_to(Path.cwd())}"

        except Exception as e:
            error_msg = f"Error creating visualization: {str(e)}"
            return f"<div style='color:red; text-align:center; padding:50px;'>⚠️ {error_msg}</div>", error_msg

    def get_available_meshes(self, project_path=None):
        """Get list of available mesh files from project directories."""
        mesh_files = []

        if project_path:
            # Look in GTR 3D output directory
            gtr_dir = Path(project_path) / "04_gtr_3d"
            if gtr_dir.exists():
                mesh_files.extend(gtr_dir.glob("**/mesh.obj"))

        return [str(f) for f in mesh_files]

# Global mesh visualizer instance
mesh_visualizer = MeshVisualizer()

def load_default_images():
    """Load default multi-view images from renders directory

    Loads front, back, left, right views from the renders directory structure.
    Each object folder contains {uid}_front.png, {uid}_back.png, {uid}_left.png, {uid}_right.png
    Maps them to view_1, view_2, view_3, view_4 respectively.
    """
    default_images = {}

    # Check if renders directory exists
    if not os.path.exists(DEFAULT_RENDERS_BASE_DIR):
        print(f"Warning: Renders directory not found: {DEFAULT_RENDERS_BASE_DIR}")
        return default_images

    # Get all object folders in renders directory
    object_folders = [f for f in os.listdir(DEFAULT_RENDERS_BASE_DIR)
                     if os.path.isdir(os.path.join(DEFAULT_RENDERS_BASE_DIR, f))]

    if not object_folders:
        print(f"Warning: No object folders found in {DEFAULT_RENDERS_BASE_DIR}")
        return default_images

    # Use the default object UID or the first available
    object_uid = DEFAULT_OBJECT_UID if DEFAULT_OBJECT_UID in object_folders else object_folders[0]
    object_dir = os.path.join(DEFAULT_RENDERS_BASE_DIR, object_uid)

    # Map view names to the 4 standard views: front, back, left, right
    view_mapping = {
        "view_1": f"{object_uid}_front.png",
        "view_2": f"{object_uid}_back.png",
        "view_3": f"{object_uid}_left.png",
        "view_4": f"{object_uid}_right.png"
    }

    # Load each view
    for view_name, filename in view_mapping.items():
        img_path = os.path.join(object_dir, filename)
        if os.path.exists(img_path):
            default_images[view_name] = img_path
        else:
            print(f"Warning: Image not found: {img_path}")

    print(f"Loaded {len(default_images)} views from object: {object_uid}")
    return default_images

DEFAULT_IMAGES = load_default_images()

def get_available_render_objects():
    """Get list of all available object UIDs in the renders directory"""
    if not os.path.exists(DEFAULT_RENDERS_BASE_DIR):
        return []
    return [f for f in os.listdir(DEFAULT_RENDERS_BASE_DIR)
            if os.path.isdir(os.path.join(DEFAULT_RENDERS_BASE_DIR, f))]

def get_projects_with_renders():
    """Get list of projects that have a 00_renders folder with images"""
    projects_dir = Path(__file__).parent / "projects"
    if not projects_dir.exists():
        return []

    projects_with_renders = []
    for project_dir in projects_dir.iterdir():
        if project_dir.is_dir():
            renders_dir = project_dir / "00_renders"
            if renders_dir.exists() and any(renders_dir.glob("*.png")):
                projects_with_renders.append(project_dir.name)

    return sorted(projects_with_renders)

def load_images_from_project_renders(project_name):
    """Load multi-view images from a project's 00_renders folder

    Args:
        project_name: The project folder name

    Returns:
        dict: Dictionary mapping view names (view_1, view_2, view_3, view_4) to image paths
    """
    import glob
    images = {}
    renders_dir = Path(__file__).parent / "projects" / project_name / "00_renders"

    if not renders_dir.exists():
        print(f"Warning: Project renders directory not found: {renders_dir}")
        return images

    # Look for *_front.png, *_back.png, etc. files
    view_suffixes = {
        "view_1": "_front.png",
        "view_2": "_back.png",
        "view_3": "_left.png",
        "view_4": "_right.png"
    }

    for view_name, suffix in view_suffixes.items():
        pattern = str(renders_dir / f"*{suffix}")
        matches = glob.glob(pattern)
        if matches:
            images[view_name] = matches[0]  # Take first match

    return images

def load_all_views_from_project(project_name):
    """Load all 4 views from a project's 00_renders folder"""
    views = []

    if not project_name:
        return None, None, None, None

    # Load images from the project renders
    project_images = load_images_from_project_renders(project_name)

    # Load views in order: view_1, view_2, view_3, view_4 (front, back, left, right)
    for view_name in ["view_1", "view_2", "view_3", "view_4"]:
        if view_name in project_images:
            img_path = project_images[view_name]
            img = Image.open(img_path)
            views.append(img)
        else:
            views.append(None)

    # Return 4 images (or None if not available)
    while len(views) < 4:
        views.append(None)

    return views[0], views[1], views[2], views[3]

def load_images_from_object(object_uid):
    """Load multi-view images from a specific object UID

    Args:
        object_uid: The object UID (folder name) in the renders directory

    Returns:
        dict: Dictionary mapping view names (view_1, view_2, view_3, view_4) to image paths
    """
    images = {}
    object_dir = os.path.join(DEFAULT_RENDERS_BASE_DIR, object_uid)

    if not os.path.exists(object_dir):
        print(f"Warning: Object directory not found: {object_dir}")
        return images

    # First try exact match with folder name
    view_mapping = {
        "view_1": f"{object_uid}_front.png",
        "view_2": f"{object_uid}_back.png",
        "view_3": f"{object_uid}_left.png",
        "view_4": f"{object_uid}_right.png"
    }

    # Load each view
    for view_name, filename in view_mapping.items():
        img_path = os.path.join(object_dir, filename)
        if os.path.exists(img_path):
            images[view_name] = img_path

    # If no exact matches found, look for any *_front.png, *_back.png, etc. files
    if not images:
        import glob
        view_suffixes = {
            "view_1": "_front.png",
            "view_2": "_back.png",
            "view_3": "_left.png",
            "view_4": "_right.png"
        }

        for view_name, suffix in view_suffixes.items():
            pattern = os.path.join(object_dir, f"*{suffix}")
            matches = glob.glob(pattern)
            if matches:
                images[view_name] = matches[0]  # Take first match

    return images

def select_default_image(view_name):
    """Load selected default image"""
    if view_name in DEFAULT_IMAGES:
        return Image.open(DEFAULT_IMAGES[view_name])
    return None

def create_binary_mask(mask):
    """Convert mask to binary (white=255 for masked, black=0 for unmasked)"""
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask] = 255
    return binary_mask

def create_colored_overlay(image, mask, color=[255, 0, 0], alpha=0.5):
    """Create colored overlay on image"""
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask] = color
    
    result = cv2.addWeighted(overlay, 1-alpha, colored_mask, alpha, 0)
    return result

def draw_points_on_image(image, points, labels=None):
    """Draw points on image for visualization"""
    if image is None or len(points) == 0:
        return image
    
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
    draw = ImageDraw.Draw(img)
    
    for i, point in enumerate(points):
        x, y = point
        # Green for positive points, red for negative
        color = 'green' if (labels is None or labels[i] == 1) else 'red'
        # Draw a circle at the point
        radius = 5
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline='white', width=2)
    
    return np.array(img)

def draw_box_on_image(image, box):
    """Draw bounding box on image"""
    if image is None or box is None:
        return image
    
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.copy()
    draw = ImageDraw.Draw(img)
    
    x1, y1, x2, y2 = box
    draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
    
    return np.array(img)

def initialize_sam(model_choice):
    """Initialize SAM model"""
    if not SAM_AVAILABLE:
        return "❌ Segment Anything not installed. Please install with: pip install segment-anything"
    
    success, message = sam_instance.load_sam_model(model_choice)
    if success:
        return f"✅ {message}"
    else:
        return f"❌ {message}"

def handle_point_click(image, evt: gr.SelectData, point_type, view_num):
    """Handle point clicks on image for a specific view"""
    global current_points_view1, current_points_view2, current_points_view3, current_points_view4

    if image is None:
        return image, "Please load an image first"

    # Get click coordinates
    x, y = evt.index

    # Add point to list based on view
    label = 1 if point_type == "Positive (include)" else 0
    point_data = {"point": [x, y], "label": label}

    if view_num == 1:
        current_points_view1.append(point_data)
        points_list = current_points_view1
    elif view_num == 2:
        current_points_view2.append(point_data)
        points_list = current_points_view2
    elif view_num == 3:
        current_points_view3.append(point_data)
        points_list = current_points_view3
    elif view_num == 4:
        current_points_view4.append(point_data)
        points_list = current_points_view4

    # Draw points on image
    points = [p["point"] for p in points_list]
    labels = [p["label"] for p in points_list]

    image_with_points = draw_points_on_image(np.array(image), points, labels)

    return image_with_points, f"View {view_num}: Added {'positive' if label == 1 else 'negative'} point at ({x}, {y}). Total: {len(points_list)}"

def clear_points():
    """Clear all points for all views"""
    global current_points, current_points_view1, current_points_view2, current_points_view3, current_points_view4
    current_points = []
    current_points_view1 = []
    current_points_view2 = []
    current_points_view3 = []
    current_points_view4 = []
    sam_instance.clear_points()
    return None, None, None, None, "All points cleared for all views"

def segment_with_points(image):
    """Segment image using the collected points"""
    global current_points

    if image is None:
        return None, None, None, "No image provided"

    if len(current_points) == 0:
        return image, None, None, "No points selected. Click on the image to add points."

    # Set image in SAM
    success, msg = sam_instance.set_image(image)
    if not success:
        return image, None, None, msg

    # Extract points and labels
    points = [p["point"] for p in current_points]
    labels = [p["label"] for p in current_points]

    # Generate mask
    mask, score, status = sam_instance.predict_from_points(points, labels)

    if mask is None:
        return image, None, None, status

    # Create binary mask
    binary_mask = create_binary_mask(mask)

    # Create colored overlay
    overlay = create_colored_overlay(np.array(image), mask, [0, 255, 0], 0.4)

    # Draw points on overlay
    overlay = draw_points_on_image(overlay, points, labels)

    return overlay, binary_mask, mask, f"✅ Mask created with score: {score:.3f}"

def segment_all_views_with_points(img1, img2, img3, img4, project_path):
    """Segment all 4 views using points and save to project folder"""
    global current_points_view1, current_points_view2, current_points_view3, current_points_view4

    if not project_path:
        return [None]*4 + [None]*4 + ["Please select or create a project first"]

    view_points = [current_points_view1, current_points_view2, current_points_view3, current_points_view4]

    if all(len(pts) == 0 for pts in view_points):
        return [None]*4 + [None]*4 + ["No points added. Click on images to add points first."]

    results = []
    masks = []
    statuses = []
    output_base = Path(project_path) / "01_sam_masks"

    for i, (view, points_list) in enumerate(zip([img1, img2, img3, img4], view_points), 1):
        if view is None:
            results.append(None)
            masks.append(None)
            statuses.append(f"View {i}: No image")
        elif len(points_list) == 0:
            results.append(None)
            masks.append(None)
            statuses.append(f"View {i}: No points added")
        else:
            # Set image in SAM
            success, msg = sam_instance.set_image(view)
            if not success:
                results.append(None)
                masks.append(None)
                statuses.append(f"View {i}: {msg}")
                continue

            # Extract points and labels for this specific view
            points = [p["point"] for p in points_list]
            labels = [p["label"] for p in points_list]

            # Generate mask
            mask, score, status = sam_instance.predict_from_points(points, labels)

            if mask is None:
                results.append(None)
                masks.append(None)
                statuses.append(f"View {i}: {status}")
                continue

            # Create binary mask
            binary_mask = create_binary_mask(mask)

            # Create colored overlay
            overlay = create_colored_overlay(np.array(view), mask, [0, 255, 0], 0.4)
            overlay = draw_points_on_image(overlay, points, labels)

            results.append(overlay)
            masks.append(binary_mask)
            statuses.append(f"View {i}: ✅ Score {score:.3f}")

            # Save to project folder
            view_folder = output_base / f"view_{i}"
            view_folder.mkdir(parents=True, exist_ok=True)

            # Save original image (resized to 512x512)
            view_array = np.array(view)
            if view_array.shape[-1] == 4:
                view_array = view_array[:, :, :3]
            img_pil = Image.fromarray(view_array)
            img_pil = img_pil.resize((512, 512), Image.BILINEAR)
            img_path = view_folder / "img.jpg"
            img_pil.save(img_path)

            # Save mask (resized to 512x512)
            mask_pil = Image.fromarray(binary_mask)
            mask_pil = mask_pil.resize((512, 512), Image.NEAREST)
            mask_path = view_folder / "mask0.png"
            mask_pil.save(mask_path)
            statuses[-1] += f" | Saved to {mask_path}"

    status_text = "\n".join(statuses)
    if any(m is not None for m in masks):
        status_text += f"\n\n✅ All data saved to: {output_base}"
        # Mark step as completed
        if project_manager:
            project_manager.update_project_step(project_path, "SAM Segmentation")

    return results[0], results[1], results[2], results[3], masks[0], masks[1], masks[2], masks[3], status_text

def handle_box_click(image, evt: gr.SelectData, view_num):
    """Handle box clicks on image for a specific view - requires 2 clicks to form a box"""
    global current_box_view1, current_box_view2, current_box_view3, current_box_view4
    global box_clicks_view1, box_clicks_view2, box_clicks_view3, box_clicks_view4

    if image is None:
        return image, f"No image loaded for view {view_num}"

    x, y = evt.index

    # Get the appropriate box clicks list based on view
    if view_num == 1:
        box_clicks = box_clicks_view1
    elif view_num == 2:
        box_clicks = box_clicks_view2
    elif view_num == 3:
        box_clicks = box_clicks_view3
    elif view_num == 4:
        box_clicks = box_clicks_view4
    else:
        return image, "Invalid view number"

    box_clicks.append([x, y])

    # Need 2 clicks to form a box
    if len(box_clicks) >= 2:
        # Get first and last click to form box
        x1, y1 = box_clicks[0]
        x2, y2 = box_clicks[-1]

        # Create box [x1, y1, x2, y2]
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

        # Store box for this view
        if view_num == 1:
            current_box_view1 = box
            box_clicks_view1 = []  # Reset clicks
        elif view_num == 2:
            current_box_view2 = box
            box_clicks_view2 = []
        elif view_num == 3:
            current_box_view3 = box
            box_clicks_view3 = []
        elif view_num == 4:
            current_box_view4 = box
            box_clicks_view4 = []

        # Draw box on image
        annotated_image = draw_box_on_image(np.array(image), box)
        return annotated_image, f"View {view_num}: Box created at [{box[0]}, {box[1]}, {box[2]}, {box[3]}]"
    else:
        # Just show the first point
        annotated_image = np.array(image).copy()
        cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
        return annotated_image, f"View {view_num}: First corner selected at ({x}, {y}). Click again for second corner."

def clear_all_boxes():
    """Clear all boxes for all views"""
    global current_box_view1, current_box_view2, current_box_view3, current_box_view4
    global box_clicks_view1, box_clicks_view2, box_clicks_view3, box_clicks_view4

    current_box_view1 = None
    current_box_view2 = None
    current_box_view3 = None
    current_box_view4 = None
    box_clicks_view1 = []
    box_clicks_view2 = []
    box_clicks_view3 = []
    box_clicks_view4 = []

    return None, None, None, None, "All boxes cleared for all views"

def segment_all_views_with_boxes(img1, img2, img3, img4, project_path):
    """Segment all 4 views using boxes and save to project folder"""
    global current_box_view1, current_box_view2, current_box_view3, current_box_view4

    if not project_path:
        return [None]*4 + [None]*4 + ["Please select or create a project first"]

    view_boxes = [current_box_view1, current_box_view2, current_box_view3, current_box_view4]

    if all(box is None for box in view_boxes):
        return [None]*4 + [None]*4 + ["No boxes drawn. Click twice on each image to draw bounding boxes."]

    results = []
    masks = []
    statuses = []
    output_base = Path(project_path) / "01_sam_masks"

    for i, (view, box) in enumerate(zip([img1, img2, img3, img4], view_boxes), 1):
        if view is None:
            results.append(None)
            masks.append(None)
            statuses.append(f"View {i}: No image")
        elif box is None:
            results.append(None)
            masks.append(None)
            statuses.append(f"View {i}: No box drawn")
        else:
            # Set image in SAM
            success, msg = sam_instance.set_image(view)
            if not success:
                results.append(None)
                masks.append(None)
                statuses.append(f"View {i}: {msg}")
                continue

            # Generate mask
            mask, score, status = sam_instance.predict_from_box(box)

            if mask is None:
                results.append(None)
                masks.append(None)
                statuses.append(f"View {i}: {status}")
                continue

            # Create binary mask
            binary_mask = create_binary_mask(mask)

            # Create colored overlay
            overlay = create_colored_overlay(np.array(view), mask, [255, 165, 0], 0.4)
            overlay = draw_box_on_image(overlay, box)

            results.append(overlay)
            masks.append(binary_mask)
            statuses.append(f"View {i}: ✅ Score {score:.3f}")

            # Save to project folder
            view_folder = output_base / f"view_{i}"
            view_folder.mkdir(parents=True, exist_ok=True)

            # Save original image (resized to 512x512)
            view_array = np.array(view)
            if view_array.shape[-1] == 4:
                view_array = view_array[:, :, :3]
            img_pil = Image.fromarray(view_array)
            img_pil = img_pil.resize((512, 512), Image.BILINEAR)
            img_path = view_folder / "img.jpg"
            img_pil.save(img_path)

            # Save mask (resized to 512x512)
            mask_pil = Image.fromarray(binary_mask)
            mask_pil = mask_pil.resize((512, 512), Image.NEAREST)
            mask_path = view_folder / "mask0.png"
            mask_pil.save(mask_path)
            statuses[-1] += f" | Saved to {mask_path}"

    status_text = "\n".join(statuses)
    if any(m is not None for m in masks):
        status_text += f"\n\n✅ All data saved to: {output_base}"
        # Mark step as completed
        if project_manager:
            project_manager.update_project_step(project_path, "SAM Segmentation")

    return results[0], results[1], results[2], results[3], masks[0], masks[1], masks[2], masks[3], status_text

def segment_all_views_everything(img1, img2, img3, img4, project_path):
    """Automatically segment all objects in all 4 views and save to project folder"""
    if not project_path:
        return [None]*4 + [None]*4 + ["Please select or create a project first"]

    views = [img1, img2, img3, img4]
    if all(v is None for v in views):
        return [None]*4 + [None]*4 + ["No images loaded. Click 'Load All 4 Views' first."]

    results = []
    masks = []
    statuses = []
    output_base = Path(project_path) / "01_sam_masks"

    for i, view in enumerate(views, 1):
        if view is None:
            results.append(None)
            masks.append(None)
            statuses.append(f"View {i}: No image")
        else:
            # Generate masks for all objects
            masks_data, status = sam_instance.generate_everything_mask(view)

            if masks_data is None or len(masks_data) == 0:
                results.append(view)
                masks.append(None)
                statuses.append(f"View {i}: {status}")
                continue

            # Sort by area and take largest masks
            masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

            # Create combined visualization
            image_array = np.array(view)
            result = image_array.copy()

            # Different colors for different masks
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

            combined_mask = np.zeros(image_array.shape[:2], dtype=bool)

            for j, mask_data in enumerate(masks_data[:6]):  # Show top 6 masks
                mask = mask_data['segmentation']
                combined_mask = combined_mask | mask
                color = colors[j % len(colors)]
                result = create_colored_overlay(result, mask, color, 0.3)

            # Create binary mask for the combined result
            binary_mask = create_binary_mask(combined_mask)

            results.append(result)
            masks.append(binary_mask)
            statuses.append(f"View {i}: ✅ Found {len(masks_data)} objects, showing top 6")

            # Save to project folder
            view_folder = output_base / f"view_{i}"
            view_folder.mkdir(parents=True, exist_ok=True)

            # Save original image (resized to 512x512)
            view_array = np.array(view)
            if view_array.shape[-1] == 4:
                view_array = view_array[:, :, :3]
            img_pil = Image.fromarray(view_array)
            img_pil = img_pil.resize((512, 512), Image.BILINEAR)
            img_path = view_folder / "img.jpg"
            img_pil.save(img_path)

            # Save mask (resized to 512x512)
            mask_pil = Image.fromarray(binary_mask)
            mask_pil = mask_pil.resize((512, 512), Image.NEAREST)
            mask_path = view_folder / "mask0.png"
            mask_pil.save(mask_path)
            statuses[-1] += f" | Saved to {mask_path}"

    status_text = "\n".join(statuses)
    if any(m is not None for m in masks):
        status_text += f"\n\n✅ All data saved to: {output_base}"
        # Mark step as completed
        if project_manager:
            project_manager.update_project_step(project_path, "SAM Segmentation")

    return results[0], results[1], results[2], results[3], masks[0], masks[1], masks[2], masks[3], status_text

def handle_box_select(image, evt: gr.SelectData, box_state):
    """Handle box selection on image"""
    global current_box
    
    if image is None:
        return image, "Please upload an image first", box_state
    
    x, y = evt.index
    
    if box_state is None or len(box_state) == 0:
        # First click - start box
        box_state = [x, y]
        return image, f"Box started at ({x}, {y}). Click again to complete the box.", box_state
    elif len(box_state) == 2:
        # Second click - complete box
        x1, y1 = box_state
        x2, y2 = x, y
        
        # Ensure proper ordering
        current_box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        
        # Draw box on image
        image_with_box = draw_box_on_image(np.array(image), current_box)
        
        return image_with_box, f"Box created: ({current_box[0]}, {current_box[1]}) to ({current_box[2]}, {current_box[3]})", []
    
    return image, "Unexpected state", []

def clear_box():
    """Clear the bounding box"""
    global current_box
    current_box = None
    return None, "Box cleared", []

def segment_with_box(image):
    """Segment image using the drawn box"""
    global current_box
    
    if image is None:
        return None, None, None, "No image provided"
    
    if current_box is None:
        return image, None, None, "No bounding box drawn. Click twice on the image to create a box."
    
    # Set image in SAM
    success, msg = sam_instance.set_image(image)
    if not success:
        return image, None, None, msg
    
    # Generate mask
    mask, score, status = sam_instance.predict_from_box(current_box)
    
    if mask is None:
        return image, None, None, status
    
    # Create binary mask
    binary_mask = create_binary_mask(mask)
    
    # Create colored overlay
    overlay = create_colored_overlay(np.array(image), mask, [255, 165, 0], 0.4)
    
    # Draw box on overlay
    overlay = draw_box_on_image(overlay, current_box)
    
    return overlay, binary_mask, mask, f"✅ Mask created with score: {score:.3f}"

def segment_everything(image):
    """Generate masks for all objects in image"""
    if image is None:
        return None, None, "No image provided"

    masks_data, status = sam_instance.generate_everything_mask(image)

    if masks_data is None:
        return image, None, status

    # Combine all masks
    if len(masks_data) > 0:
        # Sort by area and take largest masks
        masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

        # Create combined visualization
        image_array = np.array(image)
        result = image_array.copy()

        # Different colors for different masks
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]

        combined_mask = np.zeros(image_array.shape[:2], dtype=bool)

        for i, mask_data in enumerate(masks_data[:6]):  # Show top 6 masks
            mask = mask_data['segmentation']
            combined_mask = combined_mask | mask
            color = colors[i % len(colors)]
            result = create_colored_overlay(result, mask, color, 0.3)

        # Create binary mask for the combined result
        binary_mask = create_binary_mask(combined_mask)

        return result, binary_mask, f"✅ Found {len(masks_data)} objects, showing top 6"

    return image, None, "No objects detected"

def segment_all_objects_individually(image, max_objects=10):
    """Generate individual binary masks for all detected objects"""
    if image is None:
        return None, [], "No image provided"

    masks_data, status = sam_instance.generate_everything_mask(image)

    if masks_data is None:
        return None, [], status

    if len(masks_data) > 0:
        # Sort by area and take largest masks
        masks_data = sorted(masks_data, key=lambda x: x['area'], reverse=True)

        # Create combined visualization
        image_array = np.array(image)
        result = image_array.copy()

        # Different colors for different masks
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255],
                  [128, 0, 0], [0, 128, 0], [0, 0, 128], [128, 128, 0]]

        # Generate individual binary masks
        individual_masks = []
        for i, mask_data in enumerate(masks_data[:max_objects]):
            mask = mask_data['segmentation']

            # Create binary mask for this object
            binary_mask = create_binary_mask(mask)
            individual_masks.append(binary_mask)

            # Add to overlay
            color = colors[i % len(colors)]
            result = create_colored_overlay(result, mask, color, 0.3)

        return result, individual_masks, f"✅ Found {len(masks_data)} objects, showing top {min(max_objects, len(masks_data))}"

    return image, [], "No objects detected"

def load_all_views(selected_object_uid=None):
    """Load all 4 views at once from the selected object UID"""
    views = []

    # If no object selected, use default
    if not selected_object_uid:
        selected_object_uid = DEFAULT_OBJECT_UID

    # Load images from the selected object
    object_images = load_images_from_object(selected_object_uid)

    # Load views in order: view_1, view_2, view_3, view_4 (front, back, left, right)
    for view_name in ["view_1", "view_2", "view_3", "view_4"]:
        if view_name in object_images:
            img_path = object_images[view_name]
            img = Image.open(img_path)
            views.append(img)
        else:
            views.append(None)

    # Return 4 images (or None if not available)
    while len(views) < 4:
        views.append(None)

    return views[0], views[1], views[2], views[3]

def segment_all_views(view1, view2, view3, view4, max_objects, project_path):
    """Segment all 4 views and return individual masks for each, save to project folder"""
    results = []
    all_masks = []
    statuses = []

    # Check if project is selected
    if not project_path:
        return [None]*4 + [[], "Please select or create a project first"]

    output_base = Path(project_path) / "01_sam_masks"

    for i, view in enumerate([view1, view2, view3, view4], 1):
        if view is None:
            results.append(None)
            all_masks.append([])
            statuses.append(f"View {i}: No image")
        else:
            result, masks, status = segment_all_objects_individually(view, max_objects)
            results.append(result)
            all_masks.append(masks)
            statuses.append(f"View {i}: {status}")

            # Save masks to project folder
            if masks:
                view_folder = output_base / f"view_{i}"
                view_folder.mkdir(parents=True, exist_ok=True)

                # Save original image (resized to 512x512)
                view_array = np.array(view)
                if view_array.shape[-1] == 4:
                    view_array = view_array[:, :, :3]
                img_pil = Image.fromarray(view_array)
                img_pil = img_pil.resize((512, 512), Image.BILINEAR)
                img_path = view_folder / "img.jpg"
                img_pil.save(img_path)

                # Save masks (use first mask as mask0.png)
                if len(masks) > 0:
                    mask_pil = Image.fromarray(masks[0])
                    mask_pil = mask_pil.resize((512, 512), Image.NEAREST)
                    mask_path = view_folder / "mask0.png"
                    mask_pil.save(mask_path)
                    statuses[-1] += f" | Saved to {mask_path}"

    # Combine all masks from all views into one gallery
    combined_masks = []
    for i, masks in enumerate(all_masks, 1):
        combined_masks.extend(masks)

    status_text = "\n".join(statuses)
    if any(all_masks):
        status_text += f"\n\n✅ All data saved to: {output_base}"
        # Mark step as completed
        if project_manager:
            project_manager.update_project_step(project_path, "SAM Segmentation")

    return results[0], results[1], results[2], results[3], combined_masks, status_text

def apply_brush_mask(image_with_mask):
    """Apply brush-drawn mask to image"""
    if image_with_mask is None:
        return None, None, None, "No image provided"

    try:
        # Handle Gradio's ImageMask format
        if isinstance(image_with_mask, dict):
            # New format: {'background': image, 'layers': [masks], 'composite': combined}
            if 'composite' in image_with_mask:
                composite = image_with_mask['composite']
                background = image_with_mask['background']

                if isinstance(composite, Image.Image):
                    composite = np.array(composite)
                if isinstance(background, Image.Image):
                    background = np.array(background)

                # Convert to RGB if needed (remove alpha channel)
                if composite.shape[-1] == 4:
                    composite = composite[:, :, :3]
                if background.shape[-1] == 4:
                    background = background[:, :, :3]

                # Extract mask by comparing composite with background
                mask = np.any(composite != background, axis=-1).astype(np.uint8) * 255
                original_image = background
            else:
                return None, None, None, f"Invalid format. Keys: {list(image_with_mask.keys())}"
        else:
            return None, None, None, f"Unexpected input type: {type(image_with_mask)}"

        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        binary_mask = mask > 50  # Threshold for mask detection

        # Create binary mask output (white=masked, black=background)
        binary_mask_output = np.zeros(original_image.shape[:2], dtype=np.uint8)
        binary_mask_output[binary_mask] = 255

        # Create colored overlay for visualization
        overlay = create_colored_overlay(original_image, binary_mask, [0, 255, 0], 0.4)

        return overlay, binary_mask_output, f"✅ Masked area contains {np.sum(binary_mask)} pixels"

    except Exception as e:
        return None, None, f"Error: {str(e)}"

def apply_brush_with_sam(image_with_mask):
    """Use SAM automatic segmentation, then find and merge segments that overlap with brushed area"""
    if image_with_mask is None:
        return None, None, "No image provided"

    try:
        # Extract brushed mask and original image
        if isinstance(image_with_mask, dict):
            if 'composite' in image_with_mask:
                composite = image_with_mask['composite']
                background = image_with_mask['background']

                if isinstance(composite, Image.Image):
                    composite = np.array(composite)
                if isinstance(background, Image.Image):
                    background = np.array(background)

                # Convert to RGB if needed
                if composite.shape[-1] == 4:
                    composite = composite[:, :, :3]
                if background.shape[-1] == 4:
                    background = background[:, :, :3]

                # Extract brushed mask
                brushed_mask = np.any(composite != background, axis=-1)
                original_image = background
            else:
                return None, None, f"Invalid format. Keys: {list(image_with_mask.keys())}"
        else:
            return None, None, f"Unexpected input type: {type(image_with_mask)}"

        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        # Check if there's any brushed area
        if not np.any(brushed_mask):
            return original_image, None, "No brushed area detected. Please draw on the image first."

        # Run SAM automatic segmentation
        original_pil = Image.fromarray(original_image)
        masks_data, status = sam_instance.generate_everything_mask(original_pil)

        if masks_data is None or len(masks_data) == 0:
            return original_image, None, f"SAM segmentation failed: {status}"

        # Find all SAM segments that are included/contained within the brushed area
        included_masks = []
        for mask_data in masks_data:
            sam_mask = mask_data['segmentation']

            # Calculate how much of the SAM segment is inside the brushed area
            overlap = np.logical_and(sam_mask, brushed_mask)
            sam_pixels = np.sum(sam_mask)
            overlap_pixels = np.sum(overlap)

            # If most of the SAM segment (>50%) is inside the brushed area, include it
            inclusion_ratio = overlap_pixels / sam_pixels if sam_pixels > 0 else 0

            if inclusion_ratio > 0.5:  # More than 50% of SAM segment is inside brushed area
                included_masks.append(sam_mask)

        if len(included_masks) == 0:
            return original_image, None, "No SAM segments found that are contained within the brushed area. Try brushing a larger area."

        # Merge all included masks
        merged_mask = np.zeros_like(brushed_mask, dtype=bool)
        for mask in included_masks:
            merged_mask = np.logical_or(merged_mask, mask)

        # Create binary mask output
        binary_mask_output = create_binary_mask(merged_mask)

        # Create colored overlay
        overlay = create_colored_overlay(original_image, merged_mask, [0, 255, 0], 0.4)

        return overlay, binary_mask_output, f"✅ Found and merged {len(included_masks)} SAM segments contained within brushed area. Total pixels: {np.sum(merged_mask)}"

    except Exception as e:
        return None, None, f"Error: {str(e)}"

def process_all_views_with_brush(view1_mask, view2_mask, view3_mask, view4_mask, project_path):
    """Process all 4 views with brush + SAM segmentation and save to project folder"""
    results = []
    masks = []
    statuses = []

    # Use project folder structure
    if not project_path:
        return [None]*4 + [None]*4 + ["Please select or create a project first", None]

    base_dir = Path(__file__).parent
    output_base = Path(project_path) / "01_sam_masks"

    # Extract main images from view masks and save folder structure
    saved_folder_path = None

    for i, view_mask in enumerate([view1_mask, view2_mask, view3_mask, view4_mask], 1):
        if view_mask is None:
            results.append(None)
            masks.append(None)
            statuses.append(f"View {i}: No image")
        else:
            result, mask, status = apply_brush_with_sam(view_mask)
            results.append(result)
            masks.append(mask)
            statuses.append(f"View {i}: {status}")

            # Save to the new folder structure
            if mask is not None:
                # Create view folder in mask subdirectory
                view_folder = output_base / f"view_{i}"
                view_folder.mkdir(parents=True, exist_ok=True)

                # Extract and save the original image
                if isinstance(view_mask, dict) and 'background' in view_mask:
                    background = view_mask['background']
                    if isinstance(background, Image.Image):
                        img_array = np.array(background)
                    else:
                        img_array = background

                    # Convert to RGB if needed
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]

                    # Resize to 512x512 to match training resolution
                    img_pil = Image.fromarray(img_array)
                    img_pil = img_pil.resize((512, 512), Image.BILINEAR)

                    # Save resized image as img.jpg
                    img_path = view_folder / "img.jpg"
                    img_pil.save(img_path)

                # Resize mask to 512x512 and save as mask0.png
                mask_path = view_folder / "mask0.png"
                mask_image = Image.fromarray(mask)
                mask_image = mask_image.resize((512, 512), Image.NEAREST)
                mask_image.save(mask_path)

                statuses[-1] += f" | Saved to {mask_path}"
                saved_folder_path = str(output_base)

    status_text = "\n".join(statuses)
    if saved_folder_path:
        status_text += f"\n\n✅ All data saved to: {saved_folder_path}"
        # Mark step as completed
        if project_manager:
            project_manager.update_project_step(project_path, "SAM Segmentation")

    return results[0], results[1], results[2], results[3], masks[0], masks[1], masks[2], masks[3], status_text, project_path

def process_all_views_with_brush_multi_area(view1_mask, view2_mask, view3_mask, view4_mask, project_path, mask_label, area_number):
    """Process all 4 views with brush + SAM segmentation for a specific labeled area and save separately"""
    results = []
    masks = []
    statuses = []

    # Use project folder structure
    if not project_path:
        return [None]*4 + [None]*4 + ["Please select or create a project first", None]

    if not mask_label or mask_label.strip() == "":
        return [None]*4 + [None]*4 + ["Please provide a label for this mask area (e.g., 'body', 'head')", None]

    # Sanitize label for filename
    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in mask_label.strip())

    base_dir = Path(__file__).parent
    output_base = Path(project_path) / "01_sam_masks"

    # Extract main images from view masks and save folder structure
    saved_folder_path = None

    for i, view_mask in enumerate([view1_mask, view2_mask, view3_mask, view4_mask], 1):
        if view_mask is None:
            results.append(None)
            masks.append(None)
            statuses.append(f"View {i}: No image")
        else:
            result, mask, status = apply_brush_with_sam(view_mask)
            results.append(result)
            masks.append(mask)
            statuses.append(f"View {i} [{mask_label}]: {status}")

            # Save to the new folder structure
            if mask is not None:
                # Create view folder in mask subdirectory
                view_folder = output_base / f"view_{i}"
                view_folder.mkdir(parents=True, exist_ok=True)

                # Extract and save the original image (only once, not per mask)
                img_path = view_folder / "img.jpg"
                if not img_path.exists():
                    if isinstance(view_mask, dict) and 'background' in view_mask:
                        background = view_mask['background']
                        if isinstance(background, Image.Image):
                            img_array = np.array(background)
                        else:
                            img_array = background

                        # Convert to RGB if needed
                        if img_array.shape[-1] == 4:
                            img_array = img_array[:, :, :3]

                        # Resize to 512x512 to match training resolution
                        img_pil = Image.fromarray(img_array)
                        img_pil = img_pil.resize((512, 512), Image.BILINEAR)

                        # Save resized image as img.jpg
                        img_pil.save(img_path)

                # Resize mask to 512x512
                mask_image = Image.fromarray(mask)
                mask_image = mask_image.resize((512, 512), Image.NEAREST)

                # Save with label (e.g., mask_body.png)
                mask_path_labeled = view_folder / f"mask_{safe_label}.png"
                mask_image.save(mask_path_labeled)

                # Also save with number for training compatibility (e.g., mask0.png, mask1.png)
                # Count existing numbered masks to determine the next index
                existing_numbered_masks = sorted([f for f in view_folder.glob("mask[0-9]*.png")])
                # Extract the mask index from area_number (1 -> mask0, 2 -> mask1, etc.)
                mask_index = area_number - 1
                mask_path_numbered = view_folder / f"mask{mask_index}.png"
                mask_image.save(mask_path_numbered)

                statuses[-1] += f" | Saved to {mask_path_labeled} and {mask_path_numbered}"
                saved_folder_path = str(output_base)

    status_text = "\n".join(statuses)
    if saved_folder_path:
        status_text += f"\n\n✅ Area '{mask_label}' masks saved to: {saved_folder_path}"
        # Mark step as completed
        if project_manager:
            project_manager.update_project_step(project_path, "SAM Segmentation")

    return results[0], results[1], results[2], results[3], masks[0], masks[1], masks[2], masks[3], status_text, project_path

def apply_background_removal(image_with_mask, blur_strength=10):
    """Remove or blur background based on mask"""
    if image_with_mask is None:
        return None, "No image provided"

    try:
        # Handle Gradio's ImageMask format
        if isinstance(image_with_mask, dict):
            # New format: {'background': image, 'layers': [masks], 'composite': combined}
            if 'composite' in image_with_mask:
                composite = image_with_mask['composite']
                background = image_with_mask['background']

                if isinstance(composite, Image.Image):
                    composite = np.array(composite)
                if isinstance(background, Image.Image):
                    background = np.array(background)

                # Convert to RGB if needed (remove alpha channel)
                if composite.shape[-1] == 4:
                    composite = composite[:, :, :3]
                if background.shape[-1] == 4:
                    background = background[:, :, :3]

                # Extract mask by comparing composite with background
                mask = np.any(composite != background, axis=-1).astype(np.uint8) * 255
                original_image = background
            else:
                return None, f"Invalid format. Keys: {list(image_with_mask.keys())}"
        else:
            return None, f"Unexpected input type: {type(image_with_mask)}"

        if isinstance(original_image, Image.Image):
            original_img_pil = original_image
            original_image = np.array(original_image)
        else:
            original_img_pil = Image.fromarray(original_image)

        binary_mask = mask > 50

        # Create blurred background
        blurred = original_img_pil.filter(ImageFilter.GaussianBlur(radius=blur_strength))
        blurred_array = np.array(blurred)

        # Combine: keep original where masked, blur elsewhere
        result = blurred_array.copy()
        result[binary_mask] = original_image[binary_mask]

        return result, f"Background blurred for {np.sum(~binary_mask)} pixels"

    except Exception as e:
        return None, f"Error: {str(e)}"

def extract_masked_object(image_with_mask, background_color=(255, 255, 255)):
    """Extract only the masked object with custom background"""
    if image_with_mask is None:
        return None, "No image provided"

    try:
        # Handle Gradio's ImageMask format
        if isinstance(image_with_mask, dict):
            # New format: {'background': image, 'layers': [masks], 'composite': combined}
            if 'composite' in image_with_mask:
                composite = image_with_mask['composite']
                background = image_with_mask['background']

                if isinstance(composite, Image.Image):
                    composite = np.array(composite)
                if isinstance(background, Image.Image):
                    background = np.array(background)

                # Convert to RGB if needed (remove alpha channel)
                if composite.shape[-1] == 4:
                    composite = composite[:, :, :3]
                if background.shape[-1] == 4:
                    background = background[:, :, :3]

                # Extract mask by comparing composite with background
                mask = np.any(composite != background, axis=-1).astype(np.uint8) * 255
                original_image = background
            else:
                return None, f"Invalid format. Keys: {list(image_with_mask.keys())}"
        else:
            return None, f"Unexpected input type: {type(image_with_mask)}"

        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        binary_mask = mask > 50

        # Create result with background color
        result = np.full_like(original_image, background_color)
        result[binary_mask] = original_image[binary_mask]

        return result, f"Object extracted with {np.sum(binary_mask)} pixels"

    except Exception as e:
        return None, f"Error: {str(e)}"

# ===== GTR Pipeline Integration =====

# Set paths for GTR pipeline (relative to DreamEdit3D root)
GTR_CKPT_PATH = "snap_gtr/ckpts/full_checkpoint.pth"
GTR_TEMP_DIR = "temp_gtr_gradio"
GTR_EXAMPLES_DIR = "examples"
GTR_SCRIPTS_DIR = "snap_gtr/scripts"

def get_gtr_input_images(current_project=None):
    """Get list of available multiview images for GTR from projects"""
    input_images = []
    base_dir = Path(__file__).parent

    # Check project folders for multiview images (03_multiview_images)
    if project_manager:
        if current_project:
            # Filter by current project only
            project_paths = [current_project]
        else:
            # Get all projects
            project_paths = project_manager.get_all_projects()

        for project_path in project_paths:
            project_name = Path(project_path).name

            # List individual image files from 03_multiview_images
            multiview_dir = Path(project_path) / "03_multiview_images"
            if multiview_dir.exists():
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    for img_file in sorted(multiview_dir.glob(ext), reverse=True):
                        input_images.append((f"[Project] {project_name}/{img_file.name}", str(img_file)))

    # Check examples directory for image files (only if no specific project selected)
    if not current_project:
        examples_path = base_dir / GTR_EXAMPLES_DIR
        if examples_path.exists():
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_file in sorted(examples_path.glob(ext)):
                    input_images.append((f"[Example] {img_file.name}", str(img_file)))

    return sorted(input_images, reverse=True)

def prepare_multiview_gtr(input_folder, selected_example=None, selected_folder=None, project_path=None, cam_radius=None, fov=None, elevation=None, azimuth_start=None, padding=None, elevation_list=None, azimuth_list=None):
    """Run prepare_mv.py script for GTR"""
    try:
        base_dir = Path(__file__).parent

        # Determine input source - priority: selected_folder > example > input_folder
        if selected_folder:
            input_path = Path(selected_folder)
        elif selected_example:
            input_path = Path(selected_example)
        elif input_folder:
            input_path = Path(input_folder)
        else:
            return None, None, "❌ Please select a multiview image or folder"

        if not input_path.exists():
            return None, None, f"❌ Input not found: {input_path}"

        # Get the image name for folder naming
        image_name = input_path.name if input_path.is_file() else None

        # Determine where to save prepared data
        if project_path and image_name:
            # Save in project's 04_gtr_3d/{image_stem}/prepared_mv/
            image_stem = Path(image_name).stem
            out_dir = Path(project_path) / "04_gtr_3d" / image_stem / "prepared_mv"
            out_dir.mkdir(exist_ok=True, parents=True)
        elif project_path:
            # No image name, use timestamp
            import time
            timestamp = int(time.time())
            out_dir = Path(project_path) / "04_gtr_3d" / f"gtr_output_{timestamp}" / "prepared_mv"
            out_dir.mkdir(exist_ok=True, parents=True)
        else:
            # Fallback to temp directory
            temp_dir = base_dir / GTR_TEMP_DIR
            temp_dir.mkdir(exist_ok=True)
            out_dir = temp_dir / "prepared_mv"
            out_dir.mkdir(exist_ok=True, parents=True)

        # Run prepare script
        prepare_script = base_dir / GTR_SCRIPTS_DIR / "prepare_mv.py"
        cmd = [
            "python", str(prepare_script),
            "--in_dir", str(input_path),
            "--out_dir", str(out_dir)
        ]

        # Add optional camera parameters
        if cam_radius is not None:
            cmd.extend(["--cam_radius", str(cam_radius)])
        if fov is not None:
            cmd.extend(["--fov", str(fov)])
        if elevation is not None:
            cmd.extend(["--elevation", str(elevation)])
        if azimuth_start is not None:
            cmd.extend(["--azimuth_start", str(azimuth_start)])
        if padding is not None and padding > 0:
            cmd.extend(["--padding", str(int(padding))])
        if elevation_list is not None:
            cmd.extend(["--elevation_list", str(elevation_list)])
        if azimuth_list is not None:
            cmd.extend(["--azimuth_list", str(azimuth_list)])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Get prepared images for preview
        prepared_images = sorted(list(out_dir.glob("rgb_*.png")))

        if not prepared_images:
            return None, None, f"❌ No images generated\n{result.stderr}"

        num_views = len(prepared_images)
        view_type = "6-view grid (Zero123++)" if num_views == 6 else f"{num_views}-view"

        # Check for separated views directory
        separated_dir = out_dir.parent / "separated_views"
        has_separated = separated_dir.exists()

        status_msg = f"✓ Auto-detected and prepared {view_type} format\n"
        status_msg += f"📊 Generated {num_views} camera views from {input_path.name}\n\n"

        status_msg += f"📁 **Output Directories:**\n"
        status_msg += f"  🔧 Prepared data: `{out_dir.name}/`\n"
        status_msg += f"     - rgb_000.png ... rgb_{num_views-1:03d}.png (processed)\n"
        status_msg += f"     - cam_000.txt ... cam_{num_views-1:03d}.txt (cameras)\n"

        if has_separated:
            separated_count = len(list(separated_dir.glob("view_*_original.png")))
            status_msg += f"\n  🖼️  Separated views: `separated_views/`\n"
            status_msg += f"     - view_000_original.png ... view_{separated_count-1:03d}_original.png\n"
            status_msg += f"     - view_000_processed.png ... view_{separated_count-1:03d}_processed.png\n"

        status_msg += f"\n  💎 Mesh output: `{out_dir.parent.name}/`\n"

        # Show which format was detected
        if num_views == 4:
            status_msg += f"\n🎯 **Detected Format:** 4-view horizontal (Front/Right/Back/Left)\n"
        elif num_views == 6:
            status_msg += f"\n🎯 **Detected Format:** 6-view Zero123++ grid\n"
        else:
            status_msg += f"\n🎯 **Detected Format:** {num_views}-view custom grid\n"

        if result.stdout:
            status_msg += f"\n📋 **Log:**\n{result.stdout}"

        return str(out_dir), image_name, status_msg

    except subprocess.CalledProcessError as e:
        return None, None, f"Error running prepare_mv.py:\n{e.stderr}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def create_gif_html(gif_path):
    """Create HTML to display animated GIF"""
    if not gif_path or not Path(gif_path).exists():
        return "<p>No GIF generated</p>"

    # Convert to base64 to embed in HTML
    import base64
    with open(gif_path, 'rb') as f:
        gif_data = base64.b64encode(f.read()).decode()

    html = f'''
    <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
        <img src="data:image/gif;base64,{gif_data}"
             style="max-width: 100%; max-height: 400px; object-fit: contain;"
             alt="Rendering"/>
    </div>
    '''
    return html

def bake_texture_from_vertex_colors(mesh_path, texture_size=2048):
    """Bake vertex colors to UV texture map with advanced rasterization"""
    try:
        import xatlas
        from PIL import Image, ImageFilter
        from scipy.ndimage import distance_transform_edt

        mesh = trimesh.load(mesh_path, force='mesh')

        # Check if mesh has vertex colors
        if not hasattr(mesh.visual, 'vertex_colors'):
            return None, "No vertex colors to bake"

        print(f"🎨 Baking texture from {len(mesh.vertices)} vertices...")

        # Generate UV coordinates using xatlas
        vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)

        # Remap vertices and colors according to atlas
        new_vertices = mesh.vertices[vmapping]
        new_faces = indices.reshape(-1, 3)
        new_colors = mesh.visual.vertex_colors[vmapping][:, :3].astype(np.float32) / 255.0

        # Enhance colors: increase brightness and saturation
        # Convert to HSV for better color enhancement
        from colorsys import rgb_to_hsv, hsv_to_rgb
        enhanced_colors = np.zeros_like(new_colors)
        for i in range(len(new_colors)):
            r, g, b = new_colors[i]
            h, s, v = rgb_to_hsv(r, g, b)
            # Increase saturation and brightness
            s = min(s * 1.3, 1.0)  # 30% more saturation
            v = min(v * 1.2, 1.0)  # 20% brighter
            enhanced_colors[i] = hsv_to_rgb(h, s, v)

        # Create texture with proper triangle rasterization
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
        weight_map = np.zeros((texture_size, texture_size), dtype=np.float32)

        print(f"🖼️  Rasterizing {len(new_faces)} triangles to {texture_size}x{texture_size} texture...")

        # Rasterize each triangle
        for face_idx, face in enumerate(new_faces):
            # Get triangle UVs and colors
            uv0, uv1, uv2 = uvs[face[0]], uvs[face[1]], uvs[face[2]]
            c0, c1, c2 = enhanced_colors[face[0]], enhanced_colors[face[1]], enhanced_colors[face[2]]

            # Convert to pixel coordinates
            p0 = (uv0 * (texture_size - 1)).astype(np.float32)
            p1 = (uv1 * (texture_size - 1)).astype(np.float32)
            p2 = (uv2 * (texture_size - 1)).astype(np.float32)

            # Bounding box
            min_x = max(0, int(min(p0[0], p1[0], p2[0])))
            max_x = min(texture_size - 1, int(max(p0[0], p1[0], p2[0])) + 1)
            min_y = max(0, int(min(p0[1], p1[1], p2[1])))
            max_y = min(texture_size - 1, int(max(p0[1], p1[1], p2[1])) + 1)

            # Rasterize triangle using barycentric coordinates
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    p = np.array([x, y], dtype=np.float32)

                    # Compute barycentric coordinates
                    v0 = p1 - p0
                    v1 = p2 - p0
                    v2 = p - p0

                    dot00 = np.dot(v0, v0)
                    dot01 = np.dot(v0, v1)
                    dot02 = np.dot(v0, v2)
                    dot11 = np.dot(v1, v1)
                    dot12 = np.dot(v1, v2)

                    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-10)
                    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

                    # Check if point is in triangle
                    if u >= -0.01 and v >= -0.01 and u + v <= 1.01:
                        w = 1.0 - u - v
                        # Interpolate color
                        color = w * c0 + u * c1 + v * c2
                        texture[y, x] = color
                        weight_map[y, x] = 1.0

        print(f"✓ Rasterization complete. Filling gaps...")

        # Fill gaps using distance transform inpainting
        mask = weight_map > 0
        if not np.all(mask):
            # Compute distance transform
            dist = distance_transform_edt(~mask)

            # Dilate multiple times with smooth interpolation
            for _ in range(20):
                dilated = mask.copy()
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                        dilated |= shifted

                new_pixels = dilated & ~mask
                if not np.any(new_pixels):
                    break

                # Smooth interpolation from neighbors
                for c in range(3):
                    neighbor_sum = np.zeros_like(texture[:, :, c])
                    neighbor_count = np.zeros_like(weight_map)

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            shifted_tex = np.roll(np.roll(texture[:, :, c], dy, axis=0), dx, axis=1)
                            shifted_mask = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
                            neighbor_sum += shifted_tex * shifted_mask
                            neighbor_count += shifted_mask

                    texture[:, :, c][new_pixels] = neighbor_sum[new_pixels] / (neighbor_count[new_pixels] + 1e-10)

                mask = dilated

        # Convert to 8-bit and apply slight blur for smoother appearance
        texture_8bit = (np.clip(texture, 0, 1) * 255).astype(np.uint8)
        texture_pil = Image.fromarray(texture_8bit)
        texture_pil = texture_pil.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Save texture
        texture_path = Path(mesh_path).parent / "mesh_texture.png"
        texture_pil.save(texture_path, quality=95)
        print(f"✓ Saved texture to {texture_path.name}")

        # Create new OBJ with UV mapping
        textured_mesh_path = Path(mesh_path).parent / "mesh_textured.obj"

        # Write OBJ file with UV coordinates and normals
        with open(textured_mesh_path, 'w') as f:
            f.write(f"# Textured mesh with UV mapping\n")
            f.write(f"mtllib mesh_texture.mtl\n")
            f.write(f"usemtl material_0\n\n")

            # Write vertices
            for v in new_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write UV coordinates (flip V for proper orientation)
            for uv in uvs:
                f.write(f"vt {uv[0]:.6f} {1.0 - uv[1]:.6f}\n")

            # Write faces with texture coordinates
            for face in new_faces:
                f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")

        # Create MTL file with better material properties
        mtl_path = Path(mesh_path).parent / "mesh_texture.mtl"
        with open(mtl_path, 'w') as f:
            f.write("# Material with UV texture\n")
            f.write("newmtl material_0\n")
            f.write("Ka 1.0 1.0 1.0\n")  # Ambient
            f.write("Kd 1.0 1.0 1.0\n")  # Diffuse
            f.write("Ks 0.1 0.1 0.1\n")  # Specular (slight shine)
            f.write("Ns 10.0\n")          # Specular exponent
            f.write("d 1.0\n")            # Dissolve (opaque)
            f.write("illum 2\n")          # Illumination model
            f.write(f"map_Kd mesh_texture.png\n")

        print(f"✓ Created textured mesh: {textured_mesh_path.name}")

        return str(textured_mesh_path), f"✓ Baked {texture_size}x{texture_size} UV texture with enhanced colors"

    except ImportError as e:
        return None, f"⚠ Missing library: {str(e)}. Run: pip install xatlas scipy Pillow"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"⚠ Texture baking failed: {str(e)}"

def run_gtr_inference(prepared_dir, checkpoint_path=GTR_CKPT_PATH, project_path=None, image_name=None, enable_mesh_export=True, export_glb=True, enable_mesh_gif=False, render_cam_distance=3.5, render_fov=50, render_elevation=20, render_num_frames=50, transparent_bg=False):
    """Run inference.py script for GTR"""
    try:
        base_dir = Path(__file__).parent

        if not prepared_dir:
            return None, None, None, None, "❌ Please run preparation first"

        prepared_path = Path(prepared_dir)
        if not prepared_path.exists():
            return None, None, None, None, f"❌ Prepared directory not found: {prepared_dir}"

        # Handle relative checkpoint path
        ckpt_path = base_dir / checkpoint_path if not Path(checkpoint_path).is_absolute() else Path(checkpoint_path)
        if not ckpt_path.exists():
            return None, None, None, None, f"❌ Checkpoint not found: {ckpt_path}"

        # Determine output directory
        # Camera files (cam_*.txt, rgb_*.png) are in: prepared_mv/
        # Mesh outputs (mesh.obj, mesh.gif, nerf.gif) will be saved to: parent directory
        # Example:
        #   Input:  .../04_gtr_3d/generated_multiview_1762095876/prepared_mv/ (camera files)
        #   Output: .../04_gtr_3d/generated_multiview_1762095876/ (mesh files)
        out_dir = prepared_path.parent  # Go up from prepared_mv to the image folder

        # Make sure the output directory exists
        out_dir.mkdir(exist_ok=True, parents=True)

        # Run inference script from snap_gtr directory (so config paths work)
        snap_gtr_dir = base_dir / "snap_gtr"

        # Convert paths to absolute
        abs_checkpoint = ckpt_path.resolve()
        abs_prepared_dir = Path(prepared_dir).resolve()
        abs_out_dir = out_dir.resolve()

        cmd = [
            "python", "scripts/inference.py",
            "--ckpt_path", str(abs_checkpoint),
            "--in_dir", str(abs_prepared_dir),
            "--out_dir", str(abs_out_dir),
            "--seed", "2025"
        ]

        # Add flags for mesh export and rendering
        if not enable_mesh_export:
            cmd.append("--skip_mesh_export")
        if not enable_mesh_gif:
            cmd.append("--skip_mesh_gif")
        if export_glb:
            cmd.append("--export_glb")

        # Add render parameters for GIF visualization
        cmd.extend([
            "--render_cam_distance", str(render_cam_distance),
            "--render_fov", str(render_fov),
            "--render_elevation", str(render_elevation),
            "--render_num_frames", str(int(render_num_frames))
        ])

        # Add transparent background flag
        if transparent_bg:
            cmd.append("--transparent")

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(snap_gtr_dir))

        # Get output files
        mesh_file = out_dir / "mesh.obj"
        mesh_file_glb = out_dir / "mesh.glb"
        mesh_gif = out_dir / "mesh.gif"
        nerf_gif = out_dir / "nerf.gif"

        # Check if at least NeRF was generated (minimum output)
        if not nerf_gif.exists():
            return None, None, None, None, f"❌ NeRF rendering not generated\n{result.stderr}"

        # Create HTML to display GIFs
        nerf_gif_html = create_gif_html(str(nerf_gif))

        # Handle mesh export (may be skipped if enable_mesh_export is False)
        if mesh_file.exists():
            # Bake texture from vertex colors for better visualization
            textured_mesh, bake_msg = bake_texture_from_vertex_colors(str(mesh_file))
            mesh_gif_html = create_gif_html(str(mesh_gif)) if mesh_gif.exists() else "<p>Mesh GIF not rendered (disabled or skipped)</p>"
            final_mesh = textured_mesh if textured_mesh else str(mesh_file.absolute())

            # Check for GLB file
            final_glb = str(mesh_file_glb.absolute()) if mesh_file_glb.exists() else None

            # Return results with mesh
            status_msg = f"✓ Inference completed\n"
            status_msg += f"📁 Camera files location: {prepared_path}\n"
            status_msg += f"📁 Outputs saved to: {out_dir}\n"
            status_msg += f"  📦 mesh.obj (vertex colors)\n"
            if final_glb:
                status_msg += f"  📦 mesh.glb (GLB format)\n"
            if textured_mesh:
                status_msg += f"  🎨 mesh_textured.obj (UV mapped)\n"
                status_msg += f"  🖼️ mesh_texture.png\n"
            if mesh_gif.exists():
                status_msg += f"  🎬 mesh.gif\n"
            status_msg += f"  🌟 nerf.gif\n"
            status_msg += f"\n{bake_msg}"
        else:
            # NeRF-only mode (mesh export was skipped)
            mesh_gif_html = "<p>Mesh export disabled</p>"
            final_mesh = None
            final_glb = None

            status_msg = f"✓ Inference completed (NeRF-only mode)\n"
            status_msg += f"📁 Camera files location: {prepared_path}\n"
            status_msg += f"📁 Outputs saved to: {out_dir}\n"
            status_msg += f"  🌟 nerf.gif\n"
            status_msg += f"\n⚡ Mesh export was disabled"

        if result.stdout:
            status_msg += f"\n\n{result.stdout[-500:]}"  # Last 500 chars

        return (
            final_mesh,
            final_glb,
            mesh_gif_html,
            nerf_gif_html,
            status_msg
        )

    except subprocess.CalledProcessError as e:
        return None, None, None, None, f"Error running inference.py:\n{e.stderr}"
    except Exception as e:
        return None, None, None, None, f"Error: {str(e)}"


def save_gtr_params_to_project(project_path, image_name, params):
    """Save GTR parameters to project_info.json"""
    import json
    from datetime import datetime

    if not project_path:
        return

    info_file = Path(project_path) / "project_info.json"
    if not info_file.exists():
        return

    try:
        with open(info_file, 'r') as f:
            info = json.load(f)

        # Initialize gtr section if not exists
        if "gtr" not in info:
            info["gtr"] = []

        # Create GTR entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "source_image": image_name,
            **params
        }

        info["gtr"].append(entry)

        with open(info_file, 'w') as f:
            json.dump(info, f, indent=2)

        print(f"💾 Saved GTR parameters to project_info.json")
    except Exception as e:
        print(f"Error saving GTR params: {e}")


def full_gtr_pipeline(input_folder, checkpoint_path, selected_example=None, selected_folder=None, project_path=None, cam_radius=None, fov=None, elevation=None, azimuth_start=None, padding=None, enable_mesh_export=True, export_glb=True, enable_mesh_gif=False, render_cam_distance=3.5, render_fov=50, render_elevation=20, render_num_frames=50, elevation_list=None, azimuth_list=None, transparent_bg=False):
    """Run both prepare and inference for GTR"""
    # Step 1: Prepare
    prepared_dir, image_name, prep_log = prepare_multiview_gtr(input_folder, selected_example, selected_folder, project_path, cam_radius, fov, elevation, azimuth_start, padding, elevation_list, azimuth_list)

    if not prepared_dir:
        return None, None, None, None, prep_log

    # Step 2: Inference (with configurable render params for visualization)
    mesh_file, glb_file, mesh_gif, nerf_gif, inf_log = run_gtr_inference(prepared_dir, checkpoint_path, project_path, image_name, enable_mesh_export, export_glb, enable_mesh_gif, render_cam_distance, render_fov, render_elevation, render_num_frames, transparent_bg)

    # Step 3: Save GTR parameters to project_info.json
    if project_path:
        gtr_params = {
            "checkpoint": checkpoint_path,
            "prepare_params": {
                "cam_radius": cam_radius,
                "fov": fov,
                "elevation_fallback": elevation,
                "azimuth_start_fallback": azimuth_start,
                "elevation_list": elevation_list,
                "azimuth_list": azimuth_list,
                "padding": padding
            },
            "render_params": {
                "cam_distance": render_cam_distance,
                "fov": render_fov,
                "elevation": render_elevation,
                "num_frames": render_num_frames
            },
            "output": {
                "mesh_file": str(mesh_file) if mesh_file else None,
                "glb_file": str(glb_file) if glb_file else None
            }
        }
        save_gtr_params_to_project(project_path, image_name, gtr_params)

    combined_log = f"=== PREPARATION ===\n{prep_log}\n\n=== INFERENCE ===\n{inf_log}"

    return mesh_file, glb_file, mesh_gif, nerf_gif, combined_log


def get_gtr_example_images():
    """Get list of example images from examples directory"""
    example_files = []
    base_dir = Path(__file__).parent
    examples_dir = base_dir / GTR_EXAMPLES_DIR
    if examples_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            example_files.extend(sorted(examples_dir.glob(ext)))
    return [str(f) for f in example_files]

def get_gtr_prepared_directories(current_project=None):
    """Get list of prepared multi-view directories from project and examples"""
    prepared_dirs = []
    base_dir = Path(__file__).parent

    # Check current project's 04_gtr_3d directory first (priority)
    if current_project:
        project_gtr_path = Path(current_project) / "04_gtr_3d"
        if project_gtr_path.exists():
            for subdir in project_gtr_path.iterdir():
                if subdir.is_dir():
                    # Check for prepared_mv subdirectory
                    prepared_mv = subdir / "prepared_mv"
                    if prepared_mv.exists():
                        rgb_files = list(prepared_mv.glob("rgb_*.png"))
                        cam_files = list(prepared_mv.glob("cam_*.txt"))
                        if rgb_files and cam_files:
                            project_name = Path(current_project).name
                            label = f"[Project] {project_name}/{subdir.name}/prepared_mv"
                            prepared_dirs.append((label, str(prepared_mv)))

    # Check examples directory (only if no specific project selected)
    if not current_project:
        examples_path = base_dir / GTR_EXAMPLES_DIR
        if examples_path.exists():
            # Check generated_multiview directories
            generated_mv_path = examples_path / "generated_multiview"
            if generated_mv_path.exists():
                for subdir in generated_mv_path.iterdir():
                    if subdir.is_dir():
                        # Check if it has rgb_*.png and cam_*.txt files
                        rgb_files = list(subdir.glob("rgb_*.png"))
                        cam_files = list(subdir.glob("cam_*.txt"))
                        if rgb_files and cam_files:
                            prepared_dirs.append((f"[Example] {subdir.name}", str(subdir)))

            # Check other prepared directories (like race-chicken-mv)
            for subdir in examples_path.iterdir():
                if subdir.is_dir() and subdir.name.endswith('-mv'):
                    # Check if it has view_* subdirectories or rgb_*.png files
                    view_dirs = list(subdir.glob("view_*"))
                    rgb_files = list(subdir.glob("rgb_*.png"))
                    if view_dirs or rgb_files:
                        prepared_dirs.append((f"[Example] {subdir.name}", str(subdir)))

    return sorted(prepared_dirs, reverse=True)

def create_interface():
    with gr.Blocks(
        title="DreamEdit3D: Complete 3D Pipeline",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 100% !important;
            padding: 10px !important;
        }
        .contain {
            max-width: 100% !important;
        }
        #component-0 {
            max-width: 100% !important;
        }
        footer {
            display: none !important;
        }
        .project-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        gr.Markdown("""
        # 🎯 DreamEdit3D: Complete 3D Generation Pipeline

        **End-to-end workflow**: Segment objects with SAM → Train 3D-aware models → Generate multi-view images → Create 3D meshes
        """)

        # State variables
        box_state = gr.State([])
        current_project = gr.State(None)

        # Project Management Section
        with gr.Group():
            gr.Markdown("## 📁 Project Management")
            with gr.Row():
                with gr.Column(scale=2):
                    project_dropdown = gr.Dropdown(
                        choices=project_manager.get_all_projects() if project_manager else [],
                        label="Select Existing Project",
                        value=None,
                        interactive=True
                    )
                with gr.Column(scale=1):
                    project_name_input = gr.Textbox(
                        label="New Project Name (optional)",
                        placeholder="e.g., my_chicken_model"
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        create_project_btn = gr.Button("➕ Create New Project", variant="primary")
                        refresh_projects_btn = gr.Button("🔄", scale=0, min_width=50)

            project_info_display = gr.Markdown("**No project selected**")

        # Project management functions
        def create_new_project(project_name):
            if project_manager:
                project_path = project_manager.create_project(project_name if project_name else None)
                projects = project_manager.get_all_projects()
                info = f"✅ **Project created:** `{Path(project_path).name}`"
                return project_path, gr.Dropdown(choices=projects, value=project_path), info
            return None, gr.Dropdown(), "Error: Project manager not initialized"

        def refresh_projects():
            if project_manager:
                projects = project_manager.get_all_projects()
                return gr.Dropdown(choices=projects)
            return gr.Dropdown()

        def update_project_info(project_path):
            if not project_path or not project_manager:
                return "**No project selected**", project_path
            info = project_manager.get_project_info(project_path)
            if info:
                steps = ", ".join(info.get("steps_completed", [])) or "None"
                display = f"""
**Current Project:** `{Path(project_path).name}`
**Created:** {info.get('created', 'Unknown')}
**Steps Completed:** {steps}
"""
                return display, project_path
            return f"**Project:** `{Path(project_path).name}`", project_path

        create_project_btn.click(
            create_new_project,
            inputs=[project_name_input],
            outputs=[current_project, project_dropdown, project_info_display]
        )

        refresh_projects_btn.click(
            refresh_projects,
            outputs=[project_dropdown]
        )

        # Update project info when project changes
        project_dropdown.change(
            update_project_info,
            inputs=[project_dropdown],
            outputs=[project_info_display, current_project]
        )

        with gr.Tab("📥 Data Preprocessing"):
            gr.Markdown("""
            ## Data Preprocessing Pipeline

            Load 3D models from Objaverse or local GSO directory and generate multi-view renders for training.

            **Workflow:**
            1. Select source (Objaverse or GSO) and load GLB file
            2. Preview the 3D model
            3. Configure rendering parameters (views, resolution, camera angles)
            4. Generate multi-view renders and/or **Input GIF** for before/after comparison

            💡 **Input GIF** uses the same camera setup as GTR output (distance, FOV, elevation, frames) for consistent before/after comparison.
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Step 1: Load 3D Model")

                    # Source selection
                    model_source = gr.Radio(
                        label="Select Source",
                        choices=["Objaverse", "GSO (Local)"],
                        value="Objaverse",
                        info="Choose where to load the 3D model from"
                    )

                    # --- Objaverse Section ---
                    with gr.Group(visible=True) as objaverse_section:
                        gr.Markdown("#### Objaverse")
                        # Load UIDs from CSV file
                        objaverse_csv_path = Path("examples/objaverse.csv")
                        objaverse_uids = []
                        if objaverse_csv_path.exists():
                            with open(objaverse_csv_path, 'r') as f:
                                objaverse_uids = [line.strip() for line in f if line.strip()]

                        objaverse_uid_dropdown = gr.Dropdown(
                            label="Select from Examples",
                            choices=objaverse_uids,
                            value=objaverse_uids[0] if objaverse_uids else None,
                            info="Select a UID from examples"
                        )

                        objaverse_uid = gr.Textbox(
                            label="Or Enter Custom UID",
                            placeholder="e.g., 000074a334c541878360457c672b6c2e",
                            info="Enter a custom Objaverse object UID"
                        )

                    # --- GSO Section ---
                    GSO_DIR = "/home/ai/gr/DreamEdit3D/evaluation/data/gso"

                    def get_gso_models():
                        """Get list of available GSO models"""
                        if os.path.exists(GSO_DIR):
                            glb_files = glob.glob(os.path.join(GSO_DIR, "*.glb"))
                            return [Path(f).stem for f in sorted(glb_files)]
                        return []

                    with gr.Group(visible=False) as gso_section:
                        gr.Markdown("#### GSO (Google Scanned Objects)")
                        gso_models = get_gso_models()
                        gso_model_dropdown = gr.Dropdown(
                            label="Select GSO Model",
                            choices=gso_models,
                            value=gso_models[0] if gso_models else None,
                            info=f"Available models in {GSO_DIR}"
                        )
                        gso_refresh_btn = gr.Button("🔄 Refresh GSO Models", size="sm")

                    download_btn = gr.Button("📥 Load GLB", variant="primary")
                    download_status = gr.Textbox(label="Status", max_lines=2)

                    downloaded_glb_path = gr.Textbox(label="Loaded GLB Path", visible=False)

                with gr.Column(scale=1):
                    gr.Markdown("### Step 2: Preview 3D Model")

                    preview_viewer = gr.HTML(
                        label="3D Model Preview",
                        value="<div style='text-align:center; padding:100px; color:#888; background:#f5f5f5; border-radius:8px;'>Download a model to preview</div>"
                    )

            # ===== Step 2.5: 3D Segmentation =====
            gr.Markdown("### Step 2.5: 3D Segmentation (Optional)")
            gr.Markdown("Extract specific parts from the 3D model using one of these methods:")

            with gr.Tabs():
                # Tab 1: Bounding Box Selection (now first since user prefers this)
                with gr.Tab("📦 Bounding Box"):
                    gr.Markdown("""
                    **Select a 3D region using X/Y/Z sliders.** Good for cutting specific areas.
                    """)

                    with gr.Row():
                        bbox_load_btn = gr.Button("📏 Load Mesh Bounds", variant="secondary")
                        bbox_status = gr.Textbox(label="Status", max_lines=2, scale=2)

                    gr.Markdown("**Adjust the selection box:**")
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("**X Axis (Left/Right)**")
                            bbox_x_min = gr.Slider(label="X Min", minimum=-2, maximum=2, value=-1, step=0.01)
                            bbox_x_max = gr.Slider(label="X Max", minimum=-2, maximum=2, value=1, step=0.01)
                        with gr.Column():
                            gr.Markdown("**Y Axis (Up/Down)**")
                            bbox_y_min = gr.Slider(label="Y Min", minimum=-2, maximum=2, value=-1, step=0.01)
                            bbox_y_max = gr.Slider(label="Y Max", minimum=-2, maximum=2, value=1, step=0.01)
                        with gr.Column():
                            gr.Markdown("**Z Axis (Front/Back)**")
                            bbox_z_min = gr.Slider(label="Z Min", minimum=-2, maximum=2, value=-1, step=0.01)
                            bbox_z_max = gr.Slider(label="Z Max", minimum=-2, maximum=2, value=1, step=0.01)

                    with gr.Row():
                        bbox_preview_btn = gr.Button("👁️ Preview Selection", variant="secondary")
                        bbox_extract_btn = gr.Button("✂️ Extract & Center", variant="primary")

                    with gr.Row():
                        bbox_preview_img = gr.Image(label="Preview", height=250)
                        bbox_result = gr.Textbox(label="Result", max_lines=5)

                    bbox_extracted_path = gr.Textbox(visible=False)

                # Tab 2: Disconnected Components
                with gr.Tab("🧩 Disconnected Parts"):
                    gr.Markdown("""
                    **Auto-detect disconnected mesh parts.** Works when parts don't share vertices.
                    """)

                    with gr.Row():
                        analyze_btn = gr.Button("🔍 Analyze Components", variant="secondary", scale=1)
                        component_status = gr.Textbox(label="Status", max_lines=2, scale=2)

                    component_gallery = gr.Gallery(
                        label="Component Thumbnails (largest first)",
                        columns=5,
                        rows=2,
                        height=220,
                        object_fit="contain",
                        show_label=True
                    )

                    component_dropdown = gr.Dropdown(
                        label="Select Components (Ctrl/Cmd+click for multiple)",
                        choices=[],
                        value=[],
                        multiselect=True
                    )

                    with gr.Row():
                        extract_btn = gr.Button("✂️ Extract & Center Selected", variant="primary", size="lg")

                    extraction_status = gr.Textbox(label="Extraction Result", max_lines=3)

            # Hidden state variables (shared)
            extracted_glb_path = gr.Textbox(visible=False)
            component_info_state = gr.Textbox(visible=False, value="[]")
            component_choices = gr.Textbox(visible=False, value="[]")

            gr.Markdown("### Step 3: Configure Rendering Parameters\n*Uses UNLIT rendering (nvdiffrast) for consistent colors throughout the pipeline*")

            with gr.Row():
                render_source = gr.Radio(
                    label="Render Source",
                    choices=["Original Model", "Extracted Component"],
                    value="Original Model",
                    info="Choose which mesh to render"
                )

            with gr.Row():
                with gr.Column():
                    render_resolution = gr.Number(
                        label="Render Resolution",
                        value=512,
                        minimum=256,
                        maximum=2048,
                        step=64,
                        info="Resolution for rendered images"
                    )
                    num_views = gr.Number(
                        label="Number of Views",
                        value=4,
                        minimum=1,
                        maximum=16,
                        step=1,
                        info="Number of views to render"
                    )

                with gr.Column():
                    camera_distance = gr.Number(
                        label="Camera Distance",
                        value=3.5,
                        minimum=0.5,
                        maximum=10.0,
                        step=0.1,
                        info="Distance from camera to object (same as GTR output)"
                    )
                    camera_fov = gr.Number(
                        label="Camera FOV",
                        value=50.0,
                        minimum=10,
                        maximum=120,
                        step=5,
                        info="Field of view in degrees (same as GTR output)"
                    )

                with gr.Column():
                    camera_elevation = gr.Number(
                        label="Camera Elevation",
                        value=20,
                        minimum=-90,
                        maximum=90,
                        step=5,
                        info="Camera elevation angle (same as GTR output)"
                    )
                    gif_num_frames = gr.Number(
                        label="GIF Frames",
                        value=50,
                        minimum=10,
                        maximum=100,
                        step=5,
                        info="Number of frames for input GIF"
                    )

            with gr.Row():
                with gr.Column():
                    gif_start_azimuth = gr.Number(
                        label="GIF Start Azimuth",
                        value=0,
                        minimum=0,
                        maximum=360,
                        step=15,
                        info="Starting angle for GIF rotation (degrees)"
                    )
                with gr.Column():
                    gif_background = gr.Dropdown(
                        label="GIF Background",
                        choices=["white", "black", "gray", "transparent"],
                        value="white",
                        info="Background color for input GIF"
                    )

            with gr.Row():
                with gr.Column():
                    azimuth_start = gr.Number(
                        label="Azimuth Start",
                        value=90,
                        minimum=0,
                        maximum=360,
                        step=15,
                        info="Starting azimuth angle (degrees)"
                    )
                    azimuth_span = gr.Number(
                        label="Azimuth Span",
                        value=360,
                        minimum=0,
                        maximum=360,
                        step=30,
                        info="Azimuth span for views (degrees)"
                    )

                with gr.Column():
                    light_intensity = gr.Number(
                        label="Light Intensity (ignored)",
                        value=1.0,
                        minimum=0.0,
                        maximum=5.0,
                        step=0.1,
                        info="⚠️ Not used - UNLIT rendering for consistent colors",
                        visible=False  # Hide since we use UNLIT rendering
                    )
                    background_color = gr.Dropdown(
                        label="Background Color",
                        choices=["white", "black", "gray"],
                        value="white",
                        info="Background color for renders"
                    )

            with gr.Row():
                with gr.Column():
                    vertical_offset = gr.Number(
                        label="Vertical Offset",
                        value=0.0,
                        minimum=-1.0,
                        maximum=1.0,
                        step=0.05,
                        info="Shift look-at point up (+) or down (-) to reframe the object"
                    )

            with gr.Row():
                render_views_btn = gr.Button("🎨 Generate Multi-View Renders", variant="primary", size="lg")
                generate_input_gif_btn = gr.Button("🎬 Generate Input GIF", variant="secondary", size="lg")

            gr.Markdown("### Rendered Views & Input GIF")
            with gr.Row():
                render_output_1 = gr.Image(label="View 1")
                render_output_2 = gr.Image(label="View 2")
                render_output_3 = gr.Image(label="View 3")
                render_output_4 = gr.Image(label="View 4")

            with gr.Row():
                with gr.Column(scale=1, min_width=200):
                    input_gif_output = gr.Image(label="🎬 Input GIF", type="filepath", show_download_button=True)
                with gr.Column(scale=3):
                    gr.Markdown("")  # spacer

            render_status = gr.Textbox(label="Rendering Status", max_lines=4)

            # Source visibility toggle
            def toggle_source_visibility(source):
                """Toggle visibility between Objaverse and GSO sections"""
                if source == "Objaverse":
                    return gr.update(visible=True), gr.update(visible=False)
                else:
                    return gr.update(visible=False), gr.update(visible=True)

            # GSO refresh function
            def refresh_gso_models():
                """Refresh the list of available GSO models"""
                models = get_gso_models()
                return gr.update(choices=models, value=models[0] if models else None)

            # Sync dropdown selection to textbox
            def sync_uid_from_dropdown(selected_uid):
                """Sync selected UID from dropdown to textbox"""
                return selected_uid if selected_uid else ""

            # Load model function (supports both Objaverse and GSO)
            def load_model(source, uid, gso_model, project_path):
                """Load GLB file from Objaverse or GSO"""
                if not project_path:
                    return "❌ Please select or create a project first", "", None

                try:
                    import shutil
                    import base64
                    import time

                    project_dir = Path(project_path)

                    if source == "GSO (Local)":
                        # Load from GSO directory
                        if not gso_model:
                            return "❌ Please select a GSO model", "", None

                        source_path = os.path.join(GSO_DIR, f"{gso_model}.glb")
                        if not os.path.exists(source_path):
                            return f"❌ GSO model not found: {source_path}", "", None

                        # Create GSO directory in project
                        downloads_dir = project_dir / "00_gso_models"
                        downloads_dir.mkdir(exist_ok=True)

                        dest_path = downloads_dir / f"{gso_model}.glb"
                        shutil.copy2(source_path, str(dest_path))

                        model_name = gso_model
                        print(f"✓ GSO model loaded: {dest_path}")

                    else:
                        # Download from Objaverse
                        if not uid:
                            return "❌ Please enter an Objaverse UID", "", None

                        import objaverse

                        print(f"Downloading Objaverse model: {uid}")
                        downloaded = objaverse.load_objects(uids=[uid])

                        if uid not in downloaded:
                            return f"❌ Failed to download model {uid}", "", None

                        source_path = downloaded[uid]

                        # Create downloads directory in project
                        downloads_dir = project_dir / "00_objaverse_downloads"
                        downloads_dir.mkdir(exist_ok=True)

                        dest_path = downloads_dir / f"{uid}.glb"
                        shutil.copy2(source_path, str(dest_path))

                        model_name = uid
                        print(f"✓ Objaverse model saved to: {dest_path}")

                    # Create HTML viewer with base64-encoded GLB and save as file for iframe
                    with open(dest_path, 'rb') as f:
                        glb_data = f.read()
                    glb_base64 = base64.b64encode(glb_data).decode('utf-8')

                    source_label = "GSO" if source == "GSO (Local)" else "Objaverse"
                    html_content = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <meta charset="utf-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1">
                        <title>{source_label} Model: {model_name}</title>
                        <script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@3.1.1/dist/model-viewer.min.js"></script>
                        <style>
                            body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
                            model-viewer {{
                                width: 100%;
                                height: 300px;
                                background-color: #f0f0f0;
                            }}
                        </style>
                    </head>
                    <body>
                        <model-viewer
                            src="data:model/gltf-binary;base64,{glb_base64}"
                            alt="{source_label} model {model_name}"
                            auto-rotate
                            camera-controls
                            shadow-intensity="1"
                            environment-image="neutral"
                            background-color="#f0f0f0">
                        </model-viewer>
                    </body>
                    </html>
                    """

                    # Save HTML file with timestamp to avoid caching issues
                    timestamp = int(time.time() * 1000)
                    html_path = downloads_dir / f"{model_name}_{timestamp}.html"
                    with open(html_path, 'w') as f:
                        f.write(html_content)

                    # Create iframe HTML using /mesh_static/ path with cache-busting parameter
                    projects_dir = Path.cwd() / "projects"
                    relative_path = html_path.relative_to(projects_dir)
                    iframe_html = f'<iframe src="/mesh_static/{relative_path}?v={timestamp}" width="100%" height="300px" style="border:none; border-radius: 8px;"></iframe>'

                    return f"✅ {source_label} model '{model_name}' loaded successfully", str(dest_path), iframe_html

                except ImportError as ie:
                    if "objaverse" in str(ie).lower():
                        return "❌ Error: objaverse library not installed. Run: pip install objaverse", "", None
                    return f"❌ Import error: {str(ie)}", "", None
                except Exception as e:
                    return f"❌ Error: {str(e)}", "", None

            def render_multiview_images(glb_path, project_path, resolution, num_views, camera_dist, camera_fov, camera_elev,
                                       azim_start, azim_span, light_intensity, bg_color, vertical_offset=0.0):
                """Render multi-view images from GLB file using nvdiffrast (UNLIT for consistent colors).

                This uses the same rendering method as mesh.gif and evaluation scripts,
                ensuring color consistency throughout the pipeline.
                """
                if not glb_path:
                    return None, None, None, None, "❌ Please download a model first"

                if not project_path:
                    return None, None, None, None, "❌ Please select or create a project first"

                try:
                    import os
                    import nvdiffrast.torch as dr

                    # ===== Inline camera generation (from snap_gtr/utils/render_utils.py) =====
                    def get_projection_matrix(fovy_deg, aspect_ratio, near_clip, far_clip):
                        fovy = torch.deg2rad(fovy_deg)
                        batch_size = fovy.shape[0]
                        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
                        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_ratio)
                        proj_mtx[:, 1, 1] = 1.0 / torch.tan(fovy / 2.0)
                        proj_mtx[:, 2, 2] = (far_clip + near_clip) / (near_clip - far_clip)
                        proj_mtx[:, 2, 3] = (2 * far_clip * near_clip) / (near_clip - far_clip)
                        proj_mtx[:, 3, 2] = -1.0
                        return proj_mtx

                    def c2w_to_mv(c2w):
                        w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
                        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
                        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
                        w2c[:, 3, 3] = 1.0
                        return w2c

                    def get_cameras_inline(azimuth_deg, elevation_deg, width, height, fov, camera_distance, v_offset=0.0):
                        import torch.nn.functional as F
                        azimuth = torch.deg2rad(azimuth_deg)
                        elevation = torch.deg2rad(elevation_deg)
                        batch_size = len(azimuth)
                        camera_distances = torch.ones(batch_size) * camera_distance

                        camera_positions = torch.stack([
                            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                            camera_distances * torch.sin(elevation),
                            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                        ], dim=-1)

                        center = torch.zeros_like(camera_positions)
                        center[:, 1] = v_offset
                        up = torch.as_tensor([0, 1, 0], dtype=torch.float32)[None, :].repeat(batch_size, 1)
                        lookat = F.normalize(center - camera_positions, dim=-1)
                        right = F.normalize(torch.cross(lookat, up), dim=-1)
                        up = F.normalize(torch.cross(right, lookat), dim=-1)
                        c2w3x4 = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)
                        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
                        c2w[:, 3, 3] = 1.0

                        fovy_deg = torch.ones(batch_size) * fov
                        proj_mtx = get_projection_matrix(fovy_deg, width/height, 0.1, 1000.0)
                        mv = c2w_to_mv(c2w)
                        mvp_mtx = proj_mtx @ mv

                        return {"mvp_mtx": mvp_mtx, "height": height, "width": width}

                    # ===== Inline nvdiffrast context =====
                    class NVDiffRastContextInline:
                        def __init__(self, device):
                            self.device = device
                            self.ctx = dr.RasterizeCudaContext(device=device)

                        def vertex_transform(self, verts, mvp_mtx):
                            verts_homo = torch.cat([verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1)
                            return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

                        def rasterize(self, pos, tri, resolution):
                            return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

                        def interpolate(self, attr, rast, tri):
                            return dr.interpolate(attr.float(), rast, tri.int())

                        def texture(self, tex, uv, filter_mode='linear'):
                            return dr.texture(tex, uv, filter_mode=filter_mode)

                    # Create renders directory in project
                    project_dir = Path(project_path)
                    renders_dir = project_dir / "00_renders"
                    renders_dir.mkdir(exist_ok=True)

                    resolution = int(resolution)
                    num_render_views = int(num_views) if num_views else 4
                    uid = Path(glb_path).stem

                    # Parse background color
                    bg_colors = {
                        "white": (1.0, 1.0, 1.0),
                        "black": (0.0, 0.0, 0.0),
                        "gray": (0.5, 0.5, 0.5),
                        "grey": (0.5, 0.5, 0.5),
                    }
                    bg_rgb = np.array(bg_colors.get(bg_color, (1.0, 1.0, 1.0)))

                    print(f"🎨 UNLIT Rendering (nvdiffrast)")
                    print(f"Camera: distance={camera_dist}, FOV={camera_fov}, elevation={camera_elev}°")
                    print(f"Azimuth: start={azim_start}°, span={azim_span}°")
                    print(f"Background: {bg_color}")

                    # Load mesh
                    mesh = trimesh.load(glb_path, force='mesh')

                    # Normalize mesh (center + scale to [-1, 1])
                    vertices = np.array(mesh.vertices)
                    centroid = vertices.mean(axis=0)
                    vertices -= centroid
                    extents = vertices.max(axis=0) - vertices.min(axis=0)
                    scale = 2.0 / np.max(extents)
                    vertices *= scale
                    triangles = np.array(mesh.faces)

                    # Setup device and context
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ctx = NVDiffRastContextInline(device)

                    # Check for texture (high quality) vs vertex colors (fallback)
                    use_texture = False
                    tex_tensor = None
                    uv_tensor = None

                    if isinstance(mesh.visual, trimesh.visual.TextureVisuals):
                        material = getattr(mesh.visual, 'material', None)
                        uv = getattr(mesh.visual, 'uv', None)

                        tex_image = None
                        if material is not None:
                            if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
                                tex_image = material.baseColorTexture
                            elif hasattr(material, 'image') and material.image is not None:
                                tex_image = material.image

                        if tex_image is not None and uv is not None and len(uv) == len(vertices):
                            # Prepare texture tensor (1, H, W, C)
                            tex_np = np.array(tex_image, dtype=np.float32) / 255.0
                            if tex_np.ndim == 2:  # Grayscale
                                tex_np = np.stack([tex_np, tex_np, tex_np, np.ones_like(tex_np)], axis=-1)
                            elif tex_np.shape[-1] == 3:  # RGB -> RGBA
                                tex_np = np.concatenate([tex_np, np.ones((*tex_np.shape[:-1], 1), dtype=np.float32)], axis=-1)
                            tex_tensor = torch.from_numpy(tex_np).unsqueeze(0).contiguous().to(device)

                            # Prepare UV tensor - flip V coordinate for nvdiffrast convention
                            uv_np = np.array(uv, dtype=np.float32).copy()
                            uv_np[:, 1] = 1.0 - uv_np[:, 1]  # Flip V
                            uv_tensor = torch.from_numpy(uv_np).contiguous().to(device)

                            use_texture = True
                            print(f"✅ Texture mode: {tex_image.size[0]}x{tex_image.size[1]} (per-pixel sampling)")

                    # Fallback to vertex colors
                    if not use_texture:
                        if hasattr(mesh.visual, 'to_color'):
                            mesh.visual = mesh.visual.to_color()
                        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                            vertex_colors = np.array(mesh.visual.vertex_colors)
                        else:
                            vertex_colors = np.full((len(vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
                        vc_tensor = torch.from_numpy(vertex_colors.astype(np.float32)).contiguous().to(device) / 255.0
                        print(f"⚠️ Vertex color mode: {len(vertex_colors)} colors (lower quality)")

                    print(f"Mesh: {len(vertices)} vertices, {len(triangles)} faces")

                    # Convert geometry to torch tensors
                    v = torch.from_numpy(vertices.astype(np.float32)).contiguous().to(device)
                    f = torch.from_numpy(triangles.astype(np.int32)).contiguous().to(device)

                    # Generate camera poses
                    azimuths = np.linspace(
                        azim_start, azim_start + azim_span,
                        num=num_render_views, endpoint=False, dtype=np.float32
                    )
                    elevations = np.full(num_render_views, camera_elev, dtype=np.float32)

                    cameras = get_cameras_inline(
                        azimuth_deg=torch.from_numpy(azimuths),
                        elevation_deg=torch.from_numpy(elevations),
                        width=resolution,
                        height=resolution,
                        fov=camera_fov,
                        camera_distance=camera_dist,
                        v_offset=vertical_offset,
                    )

                    mvp_mtx = cameras["mvp_mtx"].to(device)
                    h, w = cameras["height"], cameras["width"]

                    # Render each view
                    rendered_images = []
                    view_names = ['front', 'right', 'back', 'left', 'view_4', 'view_5', 'view_6', 'view_7']

                    for i in range(num_render_views):
                        # Transform vertices to clip space
                        v_clip = ctx.vertex_transform(v, mvp_mtx[i:i+1])

                        # Rasterize
                        rast, rast_db = ctx.rasterize(v_clip, f, (h, w))

                        if use_texture:
                            # Interpolate UV coordinates per-pixel
                            uv_interp, _ = ctx.interpolate(uv_tensor, rast, f)
                            # Sample texture at interpolated UVs (per-pixel, high quality!)
                            out = ctx.texture(tex_tensor, uv_interp, filter_mode='linear')
                        else:
                            # Fallback: interpolate vertex colors
                            out, _ = ctx.interpolate(vc_tensor, rast, f)

                        # Get mask for valid pixels (where geometry exists)
                        mask = (rast[..., 3:4] > 0).float()

                        # Flip Y axis and convert to numpy
                        out_np = out.cpu().numpy()[0, ::-1, :, :]
                        mask_np = mask.cpu().numpy()[0, ::-1, :, :]

                        # Alpha composite with background
                        alpha = out_np[..., 3:4] * mask_np
                        rgb = out_np[..., :3]
                        composited = rgb * alpha + bg_rgb * (1 - alpha)
                        composited = (composited * 255.0).clip(0, 255).astype(np.uint8)

                        # Save image
                        view_name = view_names[i] if i < len(view_names) else f'view_{i}'
                        img_path = renders_dir / f"{uid}_{view_name}.png"
                        img = Image.fromarray(composited)
                        img.save(img_path)
                        rendered_images.append(img)

                        print(f"  Rendered view {i+1}/{num_render_views}: {view_name}")

                    # Return up to 4 images for display
                    img1 = rendered_images[0] if len(rendered_images) > 0 else None
                    img2 = rendered_images[1] if len(rendered_images) > 1 else None
                    img3 = rendered_images[2] if len(rendered_images) > 2 else None
                    img4 = rendered_images[3] if len(rendered_images) > 3 else None

                    render_mode = "texture (high quality)" if use_texture else "vertex colors"
                    status = f"✅ Successfully rendered {len(rendered_images)} views at {resolution}x{resolution}\n"
                    status += f"🎨 Method: UNLIT ({render_mode})\n"
                    status += f"Camera: distance={camera_dist}, elevation={camera_elev}°\n"
                    status += f"Background: {bg_color}\n"
                    status += f"Saved to: {renders_dir}"

                    return img1, img2, img3, img4, status

                except Exception as e:
                    import traceback
                    return None, None, None, None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"

            # ===== Component Segmentation Functions =====

            def split_mesh_components(glb_path):
                """Load GLB and split into disconnected components.
                Returns tuple: (component_info_list, status_message)
                """
                if not glb_path or not os.path.exists(glb_path):
                    return [], "❌ Please load a GLB file first"

                try:
                    mesh = trimesh.load(glb_path, force='mesh')
                    components = mesh.split(only_watertight=False)

                    component_info = []
                    for i, comp in enumerate(components):
                        center = comp.centroid
                        bounds = comp.bounds
                        size = bounds[1] - bounds[0]
                        component_info.append({
                            'index': i,
                            'vertices': len(comp.vertices),
                            'faces': len(comp.faces),
                            'center': center.tolist(),
                            'size': size.tolist(),
                            'mesh': comp
                        })

                    # Sort by vertex count (largest first)
                    component_info.sort(key=lambda x: x['vertices'], reverse=True)

                    status = f"✅ Found {len(components)} components"
                    return component_info, status

                except Exception as e:
                    return [], f"❌ Error splitting mesh: {str(e)}"

            def get_component_choices(component_info):
                """Generate checkbox choices from component info."""
                choices = []
                for comp in component_info:
                    label = f"Component {comp['index']} ({comp['vertices']} vertices, {comp['faces']} faces)"
                    choices.append(label)
                return choices

            def render_single_component(component_mesh, resolution=200):
                """Render a single component mesh and return PIL Image."""
                try:
                    # Center and normalize the component
                    comp = component_mesh.copy()
                    center = (comp.vertices.max(0) + comp.vertices.min(0)) / 2
                    comp.vertices -= center
                    scale = np.abs(comp.vertices).max()
                    if scale > 0:
                        comp.vertices /= scale

                    # Save to temp file
                    temp_glb = tempfile.NamedTemporaryFile(suffix='.glb', delete=False)
                    comp.export(temp_glb.name)
                    temp_glb.close()

                    # Render using Blender
                    temp_output_dir = tempfile.mkdtemp()
                    blender_path = shutil.which("blender") or "/snap/blender/current/blender"
                    script_path = Path(__file__).parent / "render_glb_blender.py"

                    cmd = [
                        blender_path,
                        "--background",
                        "--python", str(script_path),
                        "--",
                        temp_glb.name,
                        temp_output_dir,
                        str(resolution),
                        "thumb",
                        "2.0", "0.5", "2.0",  # Camera position
                        "2.5",  # Light intensity
                        "white"  # Background
                    ]

                    subprocess.run(cmd, capture_output=True, text=True, timeout=30)

                    # Load rendered image
                    preview_path = os.path.join(temp_output_dir, f"{Path(temp_glb.name).stem}_thumb.png")
                    img = None
                    if os.path.exists(preview_path):
                        img = Image.open(preview_path).copy()

                    # Clean up
                    os.unlink(temp_glb.name)
                    shutil.rmtree(temp_output_dir, ignore_errors=True)
                    return img

                except Exception as e:
                    print(f"Error rendering component: {e}")
                    return None

            def render_component_thumbnails(glb_path, max_components=10, resolution=200):
                """Render thumbnails for the largest components.
                Returns list of (image, label) tuples for Gallery.
                """
                if not glb_path or not os.path.exists(glb_path):
                    return []

                try:
                    mesh = trimesh.load(glb_path, force='mesh')
                    components = mesh.split(only_watertight=False)

                    # Sort by vertex count (largest first)
                    indexed_components = [(i, comp) for i, comp in enumerate(components)]
                    indexed_components.sort(key=lambda x: len(x[1].vertices), reverse=True)

                    # Render top components
                    gallery_items = []
                    for idx, comp in indexed_components[:max_components]:
                        img = render_single_component(comp, resolution)
                        if img:
                            label = f"#{idx} ({len(comp.vertices)} verts)"
                            gallery_items.append((img, label))

                    return gallery_items

                except Exception as e:
                    print(f"Error rendering thumbnails: {e}")
                    return []

            def extract_and_center_components(glb_path, selected_labels, component_info):
                """Extract selected components, combine, center, and save as new GLB.
                Returns tuple: (new_glb_path, status_message)
                """
                if not glb_path or not selected_labels or not component_info:
                    return None, "❌ Please select at least one component"

                try:
                    # Parse selected indices from labels
                    selected_indices = []
                    for label in selected_labels:
                        # Extract component index from label like "Component 0 (3730 vertices, 6778 faces)"
                        idx = int(label.split()[1])
                        selected_indices.append(idx)

                    # Load mesh and split
                    mesh = trimesh.load(glb_path, force='mesh')
                    components = mesh.split(only_watertight=False)

                    # Extract selected components
                    selected_meshes = []
                    for idx in selected_indices:
                        if idx < len(components):
                            selected_meshes.append(components[idx])

                    if not selected_meshes:
                        return None, "❌ No valid components selected"

                    # Combine selected meshes
                    if len(selected_meshes) == 1:
                        combined = selected_meshes[0]
                    else:
                        combined = trimesh.util.concatenate(selected_meshes)

                    # Center at origin
                    center = (combined.vertices.max(0) + combined.vertices.min(0)) / 2
                    combined.vertices -= center

                    # Normalize scale to fit in [-1, 1]
                    scale = np.abs(combined.vertices).max()
                    if scale > 0:
                        combined.vertices /= scale

                    # Save to new GLB file
                    output_dir = Path(glb_path).parent
                    original_name = Path(glb_path).stem
                    component_suffix = "_".join([str(i) for i in sorted(selected_indices)])
                    new_name = f"{original_name}_comp{component_suffix}_centered.glb"
                    new_path = output_dir / new_name

                    combined.export(str(new_path))

                    status = f"✅ Extracted {len(selected_meshes)} component(s)\n"
                    status += f"Total: {len(combined.vertices)} vertices, {len(combined.faces)} faces\n"
                    status += f"Saved to: {new_path}"

                    return str(new_path), status

                except Exception as e:
                    import traceback
                    return None, f"❌ Error extracting components: {str(e)}\n{traceback.format_exc()}"

            def analyze_and_preview_components(glb_path):
                """Combined function to split mesh and render thumbnails.
                Returns tuple: (component_choices, gallery_items, status, component_info_json)
                """
                if not glb_path or not os.path.exists(glb_path):
                    return [], [], "❌ Please load a GLB file first", "[]"

                # Split into components
                component_info, status = split_mesh_components(glb_path)

                if not component_info:
                    return [], [], status, "[]"

                # Generate choices for checkbox (top 10 by size)
                top_components = component_info[:10]
                choices = get_component_choices(top_components)

                # Render individual thumbnails for gallery
                gallery_items = render_component_thumbnails(glb_path, max_components=10, resolution=200)

                # Store component info as JSON for later use (without mesh objects)
                import json
                info_for_json = [{
                    'index': c['index'],
                    'vertices': c['vertices'],
                    'faces': c['faces'],
                    'center': c['center'],
                    'size': c['size']
                } for c in component_info]

                status += f"\nShowing top {len(gallery_items)} components (sorted by size)"

                return choices, gallery_items, status, json.dumps(info_for_json)

            # ===== Bounding Box Selection Functions =====

            def get_mesh_bounds(glb_path):
                """Get the bounding box of a mesh.
                Returns: (min_coords, max_coords, status)
                """
                if not glb_path or not os.path.exists(glb_path):
                    return None, None, "❌ Please load a GLB file first"

                try:
                    mesh = trimesh.load(glb_path, force='mesh')
                    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
                    return bounds[0].tolist(), bounds[1].tolist(), "✅ Mesh bounds loaded"
                except Exception as e:
                    return None, None, f"❌ Error: {str(e)}"

            def extract_by_bounding_box(glb_path, x_min, x_max, y_min, y_max, z_min, z_max):
                """Extract mesh vertices within the specified bounding box.
                Returns: (new_glb_path, preview_image, status)
                """
                if not glb_path or not os.path.exists(glb_path):
                    return None, None, "❌ Please load a GLB file first"

                try:
                    mesh = trimesh.load(glb_path, force='mesh')
                    vertices = mesh.vertices
                    faces = mesh.faces

                    # Find vertices inside bounding box
                    inside_mask = (
                        (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                        (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
                        (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
                    )

                    # Find faces where all vertices are inside the box
                    face_mask = inside_mask[faces].all(axis=1)
                    selected_faces = faces[face_mask]

                    if len(selected_faces) == 0:
                        return None, None, "❌ No geometry in selected region. Adjust the bounding box."

                    # Get unique vertices used by selected faces
                    unique_verts = np.unique(selected_faces.flatten())

                    # Create vertex index mapping
                    vert_map = {old: new for new, old in enumerate(unique_verts)}

                    # Remap face indices
                    new_faces = np.array([[vert_map[v] for v in face] for face in selected_faces])
                    new_vertices = vertices[unique_verts]

                    # Create new mesh
                    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)

                    # Copy vertex colors if available
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        new_mesh.visual.vertex_colors = mesh.visual.vertex_colors[unique_verts]

                    # Center the extracted mesh
                    center = (new_mesh.vertices.max(0) + new_mesh.vertices.min(0)) / 2
                    new_mesh.vertices -= center

                    # Normalize scale
                    scale = np.abs(new_mesh.vertices).max()
                    if scale > 0:
                        new_mesh.vertices /= scale

                    # Save to file
                    output_dir = Path(glb_path).parent
                    original_name = Path(glb_path).stem
                    new_name = f"{original_name}_bbox_extracted.glb"
                    new_path = output_dir / new_name
                    new_mesh.export(str(new_path))

                    # Render preview
                    preview_img = render_single_component(new_mesh, resolution=300)

                    status = f"✅ Extracted {len(new_vertices)} vertices, {len(new_faces)} faces\n"
                    status += f"Saved to: {new_path}"

                    return str(new_path), preview_img, status

                except Exception as e:
                    import traceback
                    return None, None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"

            def preview_bbox_selection(glb_path, x_min, x_max, y_min, y_max, z_min, z_max):
                """Preview what would be selected by the bounding box (without saving)."""
                if not glb_path or not os.path.exists(glb_path):
                    return None, "❌ Please load a GLB file first"

                try:
                    mesh = trimesh.load(glb_path, force='mesh')
                    vertices = mesh.vertices
                    faces = mesh.faces

                    # Find vertices inside bounding box
                    inside_mask = (
                        (vertices[:, 0] >= x_min) & (vertices[:, 0] <= x_max) &
                        (vertices[:, 1] >= y_min) & (vertices[:, 1] <= y_max) &
                        (vertices[:, 2] >= z_min) & (vertices[:, 2] <= z_max)
                    )

                    # Find faces where all vertices are inside
                    face_mask = inside_mask[faces].all(axis=1)

                    num_verts = inside_mask.sum()
                    num_faces = face_mask.sum()
                    total_verts = len(vertices)
                    total_faces = len(faces)

                    status = f"Selection: {num_verts}/{total_verts} vertices ({100*num_verts/total_verts:.1f}%)\n"
                    status += f"           {num_faces}/{total_faces} faces ({100*num_faces/total_faces:.1f}%)"

                    if num_faces == 0:
                        return None, status + "\n⚠️ No complete faces in selection"

                    # Create preview mesh
                    selected_faces = faces[face_mask]
                    unique_verts = np.unique(selected_faces.flatten())
                    vert_map = {old: new for new, old in enumerate(unique_verts)}
                    new_faces = np.array([[vert_map[v] for v in face] for face in selected_faces])
                    new_vertices = vertices[unique_verts]

                    preview_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
                    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                        preview_mesh.visual.vertex_colors = mesh.visual.vertex_colors[unique_verts]

                    # Render preview
                    preview_img = render_single_component(preview_mesh, resolution=300)

                    return preview_img, status

                except Exception as e:
                    return None, f"❌ Error: {str(e)}"

            # Source toggle visibility
            model_source.change(
                fn=toggle_source_visibility,
                inputs=[model_source],
                outputs=[objaverse_section, gso_section]
            )

            # GSO refresh button
            gso_refresh_btn.click(
                fn=refresh_gso_models,
                inputs=[],
                outputs=[gso_model_dropdown]
            )

            # Sync dropdown to textbox
            objaverse_uid_dropdown.change(
                fn=sync_uid_from_dropdown,
                inputs=[objaverse_uid_dropdown],
                outputs=[objaverse_uid]
            )

            # Connect load button (supports both Objaverse and GSO)
            download_btn.click(
                fn=load_model,
                inputs=[model_source, objaverse_uid, gso_model_dropdown, current_project],
                outputs=[download_status, downloaded_glb_path, preview_viewer]
            )

            # Connect render button
            def render_with_source_selection(render_source, original_path, extracted_path, project_path,
                                              resolution, num_views, camera_dist, camera_fov_val, camera_elev,
                                              azim_start, azim_span, light_int, bg_color, vert_offset):
                """Wrapper to select correct GLB path based on render source selection."""
                if render_source == "Extracted Component":
                    if extracted_path and os.path.exists(extracted_path):
                        glb_path = extracted_path
                    else:
                        return None, None, None, None, "❌ No extracted component available. Please extract a component first."
                else:
                    glb_path = original_path

                return render_multiview_images(glb_path, project_path, resolution, num_views,
                                             camera_dist, camera_fov_val, camera_elev, azim_start, azim_span,
                                             light_int, bg_color, vert_offset)

            def generate_input_gif(render_source, original_path, extracted_path, project_path,
                                   resolution, camera_dist, camera_fov_val, camera_elev, num_frames,
                                   start_azimuth, gif_bg_color):
                """Generate input GIF with same camera setup as GTR output for before/after comparison."""
                import imageio.v2 as imageio
                import torch.nn.functional as F

                if render_source == "Extracted Component":
                    if extracted_path and os.path.exists(extracted_path):
                        glb_path = extracted_path
                    else:
                        return None, "❌ No extracted component available."
                else:
                    glb_path = original_path

                if not glb_path:
                    return None, "❌ Please download a model first"
                if not project_path:
                    return None, "❌ Please select or create a project first"

                try:
                    import nvdiffrast.torch as dr

                    # Inline camera generation (same as render_multiview_images)
                    def get_projection_matrix(fovy_deg, aspect_ratio, near_clip, far_clip):
                        fovy = torch.deg2rad(fovy_deg)
                        batch_size = fovy.shape[0]
                        proj_mtx = torch.zeros(batch_size, 4, 4, dtype=torch.float32)
                        proj_mtx[:, 0, 0] = 1.0 / (torch.tan(fovy / 2.0) * aspect_ratio)
                        proj_mtx[:, 1, 1] = 1.0 / torch.tan(fovy / 2.0)
                        proj_mtx[:, 2, 2] = (far_clip + near_clip) / (near_clip - far_clip)
                        proj_mtx[:, 2, 3] = (2 * far_clip * near_clip) / (near_clip - far_clip)
                        proj_mtx[:, 3, 2] = -1.0
                        return proj_mtx

                    def c2w_to_mv(c2w):
                        w2c = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
                        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
                        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
                        w2c[:, 3, 3] = 1.0
                        return w2c

                    def get_cameras_inline(azimuth_deg, elevation_deg, width, height, fov, camera_distance):
                        azimuth = torch.deg2rad(azimuth_deg)
                        elevation = torch.deg2rad(elevation_deg)
                        batch_size = len(azimuth)
                        camera_distances = torch.ones(batch_size) * camera_distance

                        camera_positions = torch.stack([
                            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                            camera_distances * torch.sin(elevation),
                            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                        ], dim=-1)

                        center = torch.zeros_like(camera_positions)
                        up = torch.as_tensor([0, 1, 0], dtype=torch.float32)[None, :].repeat(batch_size, 1)
                        lookat = F.normalize(center - camera_positions, dim=-1)
                        right = F.normalize(torch.cross(lookat, up), dim=-1)
                        up = F.normalize(torch.cross(right, lookat), dim=-1)
                        c2w3x4 = torch.cat([torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]], dim=-1)
                        c2w = torch.cat([c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1)
                        c2w[:, 3, 3] = 1.0

                        fovy_deg = torch.ones(batch_size) * fov
                        proj_mtx = get_projection_matrix(fovy_deg, width/height, 0.1, 1000.0)
                        mv = c2w_to_mv(c2w)
                        mvp_mtx = proj_mtx @ mv

                        return {"mvp_mtx": mvp_mtx, "height": height, "width": width}

                    class NVDiffRastContextInline:
                        def __init__(self, device):
                            self.device = device
                            self.ctx = dr.RasterizeCudaContext(device=device)

                        def vertex_transform(self, verts, mvp_mtx):
                            verts_homo = torch.cat([verts, torch.ones([verts.shape[0], 1]).to(verts)], dim=-1)
                            return torch.matmul(verts_homo, mvp_mtx.permute(0, 2, 1))

                        def rasterize(self, pos, tri, resolution):
                            return dr.rasterize(self.ctx, pos.float(), tri.int(), resolution, grad_db=True)

                        def interpolate(self, attr, rast, tri):
                            return dr.interpolate(attr.float(), rast, tri.int())

                        def texture(self, tex, uv, filter_mode='linear'):
                            return dr.texture(tex, uv, filter_mode=filter_mode)

                    # Parse background color
                    is_transparent = (gif_bg_color == "transparent")
                    bg_colors = {"white": (1.0, 1.0, 1.0), "black": (0.0, 0.0, 0.0), "gray": (0.5, 0.5, 0.5), "transparent": (0.0, 0.0, 0.0)}
                    bg_rgb = np.array(bg_colors.get(gif_bg_color, (1.0, 1.0, 1.0)))

                    resolution = int(resolution)
                    num_frames = int(num_frames)
                    start_azimuth = float(start_azimuth) if start_azimuth else 0.0

                    print(f"🎬 Generating Input GIF (same camera setup as GTR output)")
                    print(f"Camera: distance={camera_dist}, FOV={camera_fov_val}, elevation={camera_elev}°")
                    print(f"Frames: {num_frames}, Start azimuth: {start_azimuth}°, Background: {gif_bg_color}")

                    # Load mesh
                    mesh = trimesh.load(glb_path, force='mesh')

                    # Normalize mesh
                    vertices = np.array(mesh.vertices)
                    centroid = vertices.mean(axis=0)
                    vertices -= centroid
                    extents = vertices.max(axis=0) - vertices.min(axis=0)
                    scale = 2.0 / np.max(extents)
                    vertices *= scale
                    triangles = np.array(mesh.faces)

                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    ctx = NVDiffRastContextInline(device)

                    v = torch.from_numpy(vertices.astype(np.float32)).contiguous().to(device)
                    f = torch.from_numpy(triangles.astype(np.int32)).contiguous().to(device)

                    # Check for texture (better quality) vs vertex colors
                    tex_tensor = None
                    uv_tensor = None
                    use_texture = False

                    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
                        material = mesh.visual.material
                        tex_image = getattr(material, 'image', None) or getattr(material, 'baseColorTexture', None)
                        uv = getattr(mesh.visual, 'uv', None)

                        if tex_image is not None and uv is not None and len(uv) == len(vertices):
                            print(f"  📷 Using texture sampling for better quality")
                            tex_np = np.array(tex_image)
                            if tex_np.ndim == 2:
                                tex_np = np.stack([tex_np]*3, axis=-1)
                            if tex_np.shape[-1] == 3:
                                tex_np = np.concatenate([tex_np, np.full((*tex_np.shape[:-1], 1), 255, dtype=tex_np.dtype)], axis=-1)
                            tex_tensor = torch.from_numpy(tex_np.astype(np.float32) / 255.0).contiguous().to(device)[None]
                            uv_np = np.array(uv, dtype=np.float32).copy()
                            uv_np[:, 1] = 1.0 - uv_np[:, 1]  # Flip V
                            uv_tensor = torch.from_numpy(uv_np).contiguous().to(device)
                            use_texture = True

                    if not use_texture:
                        # Fall back to vertex colors
                        if hasattr(mesh.visual, 'to_color'):
                            mesh.visual = mesh.visual.to_color()
                        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                            vertex_colors = np.array(mesh.visual.vertex_colors)
                        else:
                            vertex_colors = np.full((len(vertices), 4), [180, 180, 180, 255], dtype=np.uint8)
                        vc = torch.from_numpy(vertex_colors.astype(np.float32)).contiguous().to(device) / 255.0
                        print(f"  🎨 Using vertex colors")

                    # Generate camera poses for full 360° rotation (same as GTR)
                    azimuths = np.linspace(start_azimuth, start_azimuth + 360, num=num_frames, endpoint=False, dtype=np.float32)
                    elevations = np.full(num_frames, camera_elev, dtype=np.float32)

                    cameras = get_cameras_inline(
                        azimuth_deg=torch.from_numpy(azimuths),
                        elevation_deg=torch.from_numpy(elevations),
                        width=resolution,
                        height=resolution,
                        fov=camera_fov_val,
                        camera_distance=camera_dist,
                    )

                    mvp_mtx = cameras["mvp_mtx"].to(device)
                    h, w = cameras["height"], cameras["width"]

                    # Render all frames
                    frames = []
                    for i in range(num_frames):
                        v_clip = ctx.vertex_transform(v, mvp_mtx[i:i+1])
                        rast, _ = ctx.rasterize(v_clip, f, (h, w))

                        # Get alpha mask from rasterization (where triangles exist)
                        rast_mask = (rast[..., 3:4] > 0).float()

                        if use_texture:
                            # Texture sampling (better quality)
                            uv_interp, _ = ctx.interpolate(uv_tensor, rast, f)
                            out = ctx.texture(tex_tensor, uv_interp, filter_mode='linear')
                            # Mask out background pixels
                            out = out * rast_mask
                        else:
                            # Vertex color interpolation
                            out, _ = ctx.interpolate(vc, rast, f)

                        img = out.cpu().numpy()[0, ::-1, :, :]
                        alpha = rast_mask.cpu().numpy()[0, ::-1, :, :]  # Use rast mask as alpha
                        rgb = img[..., :3]

                        if is_transparent:
                            # Keep RGBA for transparent background
                            rgba = np.concatenate([rgb, alpha], axis=-1)
                            rgba = (rgba * 255.0).clip(0, 255).astype(np.uint8)
                            frames.append(rgba)
                        else:
                            # Composite with background color
                            composited = rgb * alpha + bg_rgb * (1 - alpha)
                            composited = (composited * 255.0).clip(0, 255).astype(np.uint8)
                            frames.append(composited)

                    # Save GIF (using same method as GTR for consistent speed)
                    project_dir = Path(project_path)
                    renders_dir = project_dir / "00_renders"
                    renders_dir.mkdir(exist_ok=True)
                    gif_path = renders_dir / "input.gif"

                    # Use fps=15 same as GTR's save_images_to_gif
                    if is_transparent:
                        # For transparent GIF, need to handle alpha channel
                        with imageio.get_writer(str(gif_path), mode='I', fps=15, loop=0, disposal=2) as writer:
                            for frame in frames:
                                writer.append_data(frame)
                    else:
                        with imageio.get_writer(str(gif_path), mode='I', fps=15, loop=0) as writer:
                            for frame in frames:
                                writer.append_data(frame)

                    bg_info = "transparent" if is_transparent else gif_bg_color
                    return str(gif_path), f"✅ Input GIF saved: {gif_path}\n📊 {num_frames} frames, {resolution}x{resolution}\n📷 distance={camera_dist}, FOV={camera_fov_val}, elevation={camera_elev}, start={start_azimuth}°\n🎨 Background: {bg_info}"

                except Exception as e:
                    import traceback
                    return None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"

            render_views_btn.click(
                fn=render_with_source_selection,
                inputs=[render_source, downloaded_glb_path, extracted_glb_path, current_project,
                       render_resolution, num_views, camera_distance, camera_fov, camera_elevation,
                       azimuth_start, azimuth_span, light_intensity, background_color, vertical_offset],
                outputs=[render_output_1, render_output_2, render_output_3, render_output_4, render_status]
            )

            generate_input_gif_btn.click(
                fn=generate_input_gif,
                inputs=[render_source, downloaded_glb_path, extracted_glb_path, current_project,
                       render_resolution, camera_distance, camera_fov, camera_elevation, gif_num_frames,
                       gif_start_azimuth, gif_background],
                outputs=[input_gif_output, render_status]
            )

            # Component segmentation event handlers
            def analyze_and_update_dropdown(glb_path):
                """Analyze components and update dropdown choices."""
                choices, gallery, status, info_json = analyze_and_preview_components(glb_path)
                # Return dropdown update with choices
                return gr.update(choices=choices, value=[]), gallery, status, info_json

            analyze_btn.click(
                fn=analyze_and_update_dropdown,
                inputs=[downloaded_glb_path],
                outputs=[component_dropdown, component_gallery, component_status, component_info_state]
            )

            def extract_wrapper(glb_path, selected, info_json):
                """Wrapper to parse JSON and call extract function."""
                import json
                if not selected:
                    return None, "❌ Please select at least one component from the dropdown"
                if not info_json or info_json == "[]":
                    return None, "❌ Please analyze components first"
                info = json.loads(info_json)
                return extract_and_center_components(glb_path, selected, info)

            extract_btn.click(
                fn=extract_wrapper,
                inputs=[downloaded_glb_path, component_dropdown, component_info_state],
                outputs=[extracted_glb_path, extraction_status]
            )

            # Bounding box event handlers
            def load_bounds_and_update_sliders(glb_path):
                """Load mesh bounds and update slider ranges."""
                min_coords, max_coords, status = get_mesh_bounds(glb_path)
                if min_coords is None:
                    return (
                        gr.update(), gr.update(), gr.update(),
                        gr.update(), gr.update(), gr.update(),
                        status
                    )

                # Add some padding to the bounds
                padding = 0.1
                x_min, y_min, z_min = min_coords
                x_max, y_max, z_max = max_coords

                return (
                    gr.update(minimum=x_min - padding, maximum=x_max + padding, value=x_min),
                    gr.update(minimum=x_min - padding, maximum=x_max + padding, value=x_max),
                    gr.update(minimum=y_min - padding, maximum=y_max + padding, value=y_min),
                    gr.update(minimum=y_min - padding, maximum=y_max + padding, value=y_max),
                    gr.update(minimum=z_min - padding, maximum=z_max + padding, value=z_min),
                    gr.update(minimum=z_min - padding, maximum=z_max + padding, value=z_max),
                    f"✅ Bounds: X[{x_min:.2f}, {x_max:.2f}] Y[{y_min:.2f}, {y_max:.2f}] Z[{z_min:.2f}, {z_max:.2f}]"
                )

            bbox_load_btn.click(
                fn=load_bounds_and_update_sliders,
                inputs=[downloaded_glb_path],
                outputs=[bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max, bbox_status]
            )

            bbox_preview_btn.click(
                fn=preview_bbox_selection,
                inputs=[downloaded_glb_path, bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max],
                outputs=[bbox_preview_img, bbox_result]
            )

            def bbox_extract_and_update(glb_path, x_min, x_max, y_min, y_max, z_min, z_max):
                """Extract by bounding box and update the extracted path."""
                new_path, preview, status = extract_by_bounding_box(
                    glb_path, x_min, x_max, y_min, y_max, z_min, z_max
                )
                return new_path, preview, status, new_path  # Also update extracted_glb_path

            bbox_extract_btn.click(
                fn=bbox_extract_and_update,
                inputs=[downloaded_glb_path, bbox_x_min, bbox_x_max, bbox_y_min, bbox_y_max, bbox_z_min, bbox_z_max],
                outputs=[bbox_extracted_path, bbox_preview_img, bbox_result, extracted_glb_path]
            )

        with gr.Tab("✂️ SAM Segmentation"):
            gr.Markdown("""
            ## Segment Anything Model (SAM) - Interactive Object Segmentation

            Create precise masks for objects using various segmentation methods.
            **Output format**: White (255) = Object, Black (0) = Background
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Model Settings")
                    model_choice = gr.Radio(
                        choices=['vit_b', 'vit_l', 'vit_h'],
                        value='vit_h',
                        label="SAM Model Size",
                        info="vit_h=best quality, vit_b=fastest"
                    )
                    init_btn = gr.Button("🚀 Initialize SAM Model", variant="primary")
                    init_status = gr.Textbox(label="Model Status", interactive=False)

            init_btn.click(fn=initialize_sam, inputs=[model_choice], outputs=[init_status])

            gr.Markdown("---")

            with gr.Tab("🎯 Point-based Segmentation"):
                gr.Markdown("""
                **Instructions:**
                1. Select image source: Renders Directory (evaluation data) or Project Renders (from your project's 00_renders folder)
                2. Make sure you've created or selected a project in the Project Management section above
                3. Click "Load All 4 Views" to load all multi-view images
                4. Select point type (Positive = include, Negative = exclude)
                5. Click on each image to add points
                6. Click "Generate Masks" to create segmentation for all views
                7. Masks will be saved to your project's `01_sam_masks/` folder
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Image source selector
                        point_source_selector = gr.Radio(
                            choices=["Renders Directory", "Project Renders"],
                            value="Renders Directory",
                            label="Image Source",
                            info="Choose where to load images from"
                        )

                        # Object selector (for Renders Directory)
                        with gr.Row(visible=True) as point_renders_row:
                            point_object_selector = gr.Dropdown(
                                choices=get_available_render_objects(),
                                value=DEFAULT_OBJECT_UID if (get_available_render_objects() and DEFAULT_OBJECT_UID in get_available_render_objects()) else (get_available_render_objects()[0] if get_available_render_objects() else None),
                                label="Select Object from Renders",
                                scale=4,
                                interactive=True
                            )
                            point_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        # Project selector (for Project Renders)
                        with gr.Row(visible=False) as point_project_row:
                            point_project_selector = gr.Dropdown(
                                choices=get_projects_with_renders(),
                                value=None,
                                label="Select Project with Renders",
                                scale=4,
                                interactive=True
                            )
                            point_project_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        load_point_views_btn = gr.Button("Load All 4 Views", variant="secondary")

                        point_type = gr.Radio(
                            choices=["Positive (include)", "Negative (exclude)"],
                            value="Positive (include)",
                            label="Point Type"
                        )

                        with gr.Row():
                            clear_points_btn = gr.Button("Clear Points", variant="secondary")
                            segment_points_btn = gr.Button("Generate Masks for All Views", variant="primary")

                gr.Markdown("### Click on Images to Add Points (annotations will appear on the same images)")
                with gr.Row():
                    point_image1 = gr.Image(label="View 1", type="pil")
                    point_image2 = gr.Image(label="View 2", type="pil")
                    point_image3 = gr.Image(label="View 3", type="pil")
                    point_image4 = gr.Image(label="View 4", type="pil")

                gr.Markdown("### Segmentation Results")
                with gr.Row():
                    point_result1 = gr.Image(label="View 1 - Result")
                    point_result2 = gr.Image(label="View 2 - Result")
                    point_result3 = gr.Image(label="View 3 - Result")
                    point_result4 = gr.Image(label="View 4 - Result")

                gr.Markdown("### Binary Masks")
                with gr.Row():
                    point_mask1 = gr.Image(label="View 1 - Binary Mask")
                    point_mask2 = gr.Image(label="View 2 - Binary Mask")
                    point_mask3 = gr.Image(label="View 3 - Binary Mask")
                    point_mask4 = gr.Image(label="View 4 - Binary Mask")

                point_status = gr.Textbox(label="Status", lines=4)

                # Handle source selector change
                def update_point_source_visibility(source):
                    if source == "Renders Directory":
                        return gr.Row(visible=True), gr.Row(visible=False)
                    else:
                        return gr.Row(visible=False), gr.Row(visible=True)

                point_source_selector.change(
                    fn=update_point_source_visibility,
                    inputs=[point_source_selector],
                    outputs=[point_renders_row, point_project_row]
                )

                # Load all views (from appropriate source)
                def load_point_views_from_source(source, object_uid, project_name):
                    if source == "Renders Directory":
                        return load_all_views(object_uid)
                    else:
                        return load_all_views_from_project(project_name)

                load_point_views_btn.click(
                    fn=load_point_views_from_source,
                    inputs=[point_source_selector, point_object_selector, point_project_selector],
                    outputs=[point_image1, point_image2, point_image3, point_image4]
                )

                # Refresh buttons
                def refresh_point_object_list():
                    objects = get_available_render_objects()
                    return gr.Dropdown(choices=objects, value=objects[0] if objects else None)

                def refresh_point_project_list():
                    projects = get_projects_with_renders()
                    return gr.Dropdown(choices=projects, value=projects[0] if projects else None)

                point_refresh_btn.click(
                    fn=refresh_point_object_list,
                    outputs=[point_object_selector]
                )

                point_project_refresh_btn.click(
                    fn=refresh_point_project_list,
                    outputs=[point_project_selector]
                )

                # Handle point clicks on all 4 views
                def handle_point_click_view1(img, pt, evt: gr.SelectData):
                    return handle_point_click(img, evt, pt, 1)

                def handle_point_click_view2(img, pt, evt: gr.SelectData):
                    return handle_point_click(img, evt, pt, 2)

                def handle_point_click_view3(img, pt, evt: gr.SelectData):
                    return handle_point_click(img, evt, pt, 3)

                def handle_point_click_view4(img, pt, evt: gr.SelectData):
                    return handle_point_click(img, evt, pt, 4)

                point_image1.select(
                    fn=handle_point_click_view1,
                    inputs=[point_image1, point_type],
                    outputs=[point_image1, point_status]
                )
                point_image2.select(
                    fn=handle_point_click_view2,
                    inputs=[point_image2, point_type],
                    outputs=[point_image2, point_status]
                )
                point_image3.select(
                    fn=handle_point_click_view3,
                    inputs=[point_image3, point_type],
                    outputs=[point_image3, point_status]
                )
                point_image4.select(
                    fn=handle_point_click_view4,
                    inputs=[point_image4, point_type],
                    outputs=[point_image4, point_status]
                )

                # Clear points
                clear_points_btn.click(
                    fn=clear_points,
                    outputs=[point_image1, point_image2, point_image3, point_image4, point_status]
                )

                # Generate masks for all views
                segment_points_btn.click(
                    fn=segment_all_views_with_points,
                    inputs=[point_image1, point_image2, point_image3, point_image4, current_project],
                    outputs=[point_result1, point_result2, point_result3, point_result4,
                            point_mask1, point_mask2, point_mask3, point_mask4, point_status]
                )

            with gr.Tab("📦 Box-based Segmentation"):
                gr.Markdown("""
                **Instructions:**
                1. Select image source: Renders Directory (evaluation data) or Project Renders (from your project's 00_renders folder)
                2. Make sure you've created or selected a project in the Project Management section above
                3. Click "Load All 4 Views" to load all multi-view images
                4. Click twice on each image to draw bounding boxes (first click = top-left, second click = bottom-right)
                5. Click "Generate Masks" to create segmentation for all views
                6. Masks will be saved to your project's `01_sam_masks/` folder
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Image source selector
                        box_source_selector = gr.Radio(
                            choices=["Renders Directory", "Project Renders"],
                            value="Renders Directory",
                            label="Image Source",
                            info="Choose where to load images from"
                        )

                        # Object selector (for Renders Directory)
                        with gr.Row(visible=True) as box_renders_row:
                            box_object_selector = gr.Dropdown(
                                choices=get_available_render_objects(),
                                value=DEFAULT_OBJECT_UID if (get_available_render_objects() and DEFAULT_OBJECT_UID in get_available_render_objects()) else (get_available_render_objects()[0] if get_available_render_objects() else None),
                                label="Select Object from Renders",
                                scale=4,
                                interactive=True
                            )
                            box_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        # Project selector (for Project Renders)
                        with gr.Row(visible=False) as box_project_row:
                            box_project_selector = gr.Dropdown(
                                choices=get_projects_with_renders(),
                                value=None,
                                label="Select Project with Renders",
                                scale=4,
                                interactive=True
                            )
                            box_project_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        load_box_views_btn = gr.Button("Load All 4 Views", variant="secondary")

                        with gr.Row():
                            clear_box_btn = gr.Button("Clear Boxes", variant="secondary")
                            segment_box_btn = gr.Button("Generate Masks for All Views", variant="primary")

                gr.Markdown("### Click Twice on Images to Draw Boxes (annotations will appear on the same images)")
                with gr.Row():
                    box_image1 = gr.Image(label="View 1", type="pil")
                    box_image2 = gr.Image(label="View 2", type="pil")
                    box_image3 = gr.Image(label="View 3", type="pil")
                    box_image4 = gr.Image(label="View 4", type="pil")

                gr.Markdown("### Segmentation Results")
                with gr.Row():
                    box_result1 = gr.Image(label="View 1 - Result")
                    box_result2 = gr.Image(label="View 2 - Result")
                    box_result3 = gr.Image(label="View 3 - Result")
                    box_result4 = gr.Image(label="View 4 - Result")

                gr.Markdown("### Binary Masks")
                with gr.Row():
                    box_mask1 = gr.Image(label="View 1 - Binary Mask")
                    box_mask2 = gr.Image(label="View 2 - Binary Mask")
                    box_mask3 = gr.Image(label="View 3 - Binary Mask")
                    box_mask4 = gr.Image(label="View 4 - Binary Mask")

                box_status = gr.Textbox(label="Status", lines=4)

                # Handle source selector change
                def update_box_source_visibility(source):
                    if source == "Renders Directory":
                        return gr.Row(visible=True), gr.Row(visible=False)
                    else:
                        return gr.Row(visible=False), gr.Row(visible=True)

                box_source_selector.change(
                    fn=update_box_source_visibility,
                    inputs=[box_source_selector],
                    outputs=[box_renders_row, box_project_row]
                )

                # Load all views (from appropriate source)
                def load_box_views_from_source(source, object_uid, project_name):
                    if source == "Renders Directory":
                        return load_all_views(object_uid)
                    else:
                        return load_all_views_from_project(project_name)

                load_box_views_btn.click(
                    fn=load_box_views_from_source,
                    inputs=[box_source_selector, box_object_selector, box_project_selector],
                    outputs=[box_image1, box_image2, box_image3, box_image4]
                )

                # Refresh buttons
                def refresh_box_object_list():
                    objects = get_available_render_objects()
                    return gr.Dropdown(choices=objects, value=objects[0] if objects else None)

                def refresh_box_project_list():
                    projects = get_projects_with_renders()
                    return gr.Dropdown(choices=projects, value=projects[0] if projects else None)

                box_refresh_btn.click(
                    fn=refresh_box_object_list,
                    outputs=[box_object_selector]
                )

                box_project_refresh_btn.click(
                    fn=refresh_box_project_list,
                    outputs=[box_project_selector]
                )

                # Create wrapper functions for box click handlers
                def handle_box_click_view1(img, evt: gr.SelectData):
                    return handle_box_click(img, evt, 1)

                def handle_box_click_view2(img, evt: gr.SelectData):
                    return handle_box_click(img, evt, 2)

                def handle_box_click_view3(img, evt: gr.SelectData):
                    return handle_box_click(img, evt, 3)

                def handle_box_click_view4(img, evt: gr.SelectData):
                    return handle_box_click(img, evt, 4)

                # Handle box selection on each view
                box_image1.select(
                    fn=handle_box_click_view1,
                    inputs=[box_image1],
                    outputs=[box_image1, box_status]
                )
                box_image2.select(
                    fn=handle_box_click_view2,
                    inputs=[box_image2],
                    outputs=[box_image2, box_status]
                )
                box_image3.select(
                    fn=handle_box_click_view3,
                    inputs=[box_image3],
                    outputs=[box_image3, box_status]
                )
                box_image4.select(
                    fn=handle_box_click_view4,
                    inputs=[box_image4],
                    outputs=[box_image4, box_status]
                )

                # Clear boxes
                clear_box_btn.click(
                    fn=clear_all_boxes,
                    outputs=[box_image1, box_image2, box_image3, box_image4, box_status]
                )

                # Generate masks for all views
                segment_box_btn.click(
                    fn=segment_all_views_with_boxes,
                    inputs=[box_image1, box_image2, box_image3, box_image4, current_project],
                    outputs=[box_result1, box_result2, box_result3, box_result4,
                            box_mask1, box_mask2, box_mask3, box_mask4, box_status]
                )
        
            with gr.Tab("🔍 Segment Everything"):
                gr.Markdown("""
                **Automatically find and segment all objects in all views**

                **Instructions:**
                1. Select image source: Renders Directory (evaluation data) or Project Renders (from your project's 00_renders folder)
                2. Make sure you've created or selected a project in the Project Management section above
                3. Click "Load All 4 Views" to load all multi-view images
                4. Click "Segment All Objects" to automatically detect and segment everything in all views
                5. Masks will be saved to your project's `01_sam_masks/` folder
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Image source selector
                        everything_source_selector = gr.Radio(
                            choices=["Renders Directory", "Project Renders"],
                            value="Renders Directory",
                            label="Image Source",
                            info="Choose where to load images from"
                        )

                        # Object selector (for Renders Directory)
                        with gr.Row(visible=True) as everything_renders_row:
                            everything_object_selector = gr.Dropdown(
                                choices=get_available_render_objects(),
                                value=DEFAULT_OBJECT_UID if (get_available_render_objects() and DEFAULT_OBJECT_UID in get_available_render_objects()) else (get_available_render_objects()[0] if get_available_render_objects() else None),
                                label="Select Object from Renders",
                                scale=4,
                                interactive=True
                            )
                            everything_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        # Project selector (for Project Renders)
                        with gr.Row(visible=False) as everything_project_row:
                            everything_project_selector = gr.Dropdown(
                                choices=get_projects_with_renders(),
                                value=None,
                                label="Select Project with Renders",
                                scale=4,
                                interactive=True
                            )
                            everything_project_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        load_everything_views_btn = gr.Button("Load All 4 Views", variant="secondary")
                        segment_everything_all_btn = gr.Button("Segment All Objects in All Views", variant="primary")

                gr.Markdown("### Loaded Images")
                with gr.Row():
                    everything_image1 = gr.Image(label="View 1", type="pil")
                    everything_image2 = gr.Image(label="View 2", type="pil")
                    everything_image3 = gr.Image(label="View 3", type="pil")
                    everything_image4 = gr.Image(label="View 4", type="pil")

                gr.Markdown("### Segmentation Results")
                with gr.Row():
                    everything_result1 = gr.Image(label="View 1 - All Objects")
                    everything_result2 = gr.Image(label="View 2 - All Objects")
                    everything_result3 = gr.Image(label="View 3 - All Objects")
                    everything_result4 = gr.Image(label="View 4 - All Objects")

                gr.Markdown("### Binary Masks")
                with gr.Row():
                    everything_mask1 = gr.Image(label="View 1 - Combined Mask")
                    everything_mask2 = gr.Image(label="View 2 - Combined Mask")
                    everything_mask3 = gr.Image(label="View 3 - Combined Mask")
                    everything_mask4 = gr.Image(label="View 4 - Combined Mask")

                everything_status = gr.Textbox(label="Status", lines=4)

                # Handle source selector change
                def update_everything_source_visibility(source):
                    if source == "Renders Directory":
                        return gr.Row(visible=True), gr.Row(visible=False)
                    else:
                        return gr.Row(visible=False), gr.Row(visible=True)

                everything_source_selector.change(
                    fn=update_everything_source_visibility,
                    inputs=[everything_source_selector],
                    outputs=[everything_renders_row, everything_project_row]
                )

                # Load all views (from appropriate source)
                def load_everything_views_from_source(source, object_uid, project_name):
                    if source == "Renders Directory":
                        return load_all_views(object_uid)
                    else:
                        return load_all_views_from_project(project_name)

                load_everything_views_btn.click(
                    fn=load_everything_views_from_source,
                    inputs=[everything_source_selector, everything_object_selector, everything_project_selector],
                    outputs=[everything_image1, everything_image2, everything_image3, everything_image4]
                )

                # Refresh buttons
                def refresh_everything_object_list():
                    objects = get_available_render_objects()
                    return gr.Dropdown(choices=objects, value=objects[0] if objects else None)

                def refresh_everything_project_list():
                    projects = get_projects_with_renders()
                    return gr.Dropdown(choices=projects, value=projects[0] if projects else None)

                everything_refresh_btn.click(
                    fn=refresh_everything_object_list,
                    outputs=[everything_object_selector]
                )

                everything_project_refresh_btn.click(
                    fn=refresh_everything_project_list,
                    outputs=[everything_project_selector]
                )

                # Segment everything in all views
                segment_everything_all_btn.click(
                    fn=segment_all_views_everything,
                    inputs=[everything_image1, everything_image2, everything_image3, everything_image4, current_project],
                    outputs=[everything_result1, everything_result2, everything_result3, everything_result4,
                            everything_mask1, everything_mask2, everything_mask3, everything_mask4, everything_status]
                )

            with gr.Tab("🎨 Auto-Segment Individual Objects"):
                gr.Markdown("""
                **Automatically segment all objects and show individual binary masks**

                **Instructions:**
                1. Select image source: Renders Directory (evaluation data) or Project Renders (from your project's 00_renders folder)
                2. Click "Load All 4 Views" to load all multi-view images at once
                3. Choose how many objects to detect per view (max 20)
                4. Click "Auto-Segment All Views" to generate individual masks for each object in all 4 views
                5. Each detected object from all views gets its own binary mask displayed in the gallery
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Image source selector
                        auto_source_selector = gr.Radio(
                            choices=["Renders Directory", "Project Renders"],
                            value="Renders Directory",
                            label="Image Source",
                            info="Choose where to load images from"
                        )

                        # Add object selector dropdown (for Renders Directory)
                        available_objects_auto = get_available_render_objects()
                        print(f"[Auto-Segment Tab] Found {len(available_objects_auto)} objects for dropdown")  # Debug

                        with gr.Row(visible=True) as auto_renders_row:
                            auto_object_selector = gr.Dropdown(
                                choices=available_objects_auto,
                                value=DEFAULT_OBJECT_UID if (available_objects_auto and DEFAULT_OBJECT_UID in available_objects_auto) else (available_objects_auto[0] if available_objects_auto else None),
                                label="Select Object from Renders",
                                info=f"Choose from {len(available_objects_auto)} available rendered objects" if available_objects_auto else "No objects found in renders directory",
                                scale=4,
                                interactive=True
                            )
                            auto_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        # Project selector (for Project Renders)
                        with gr.Row(visible=False) as auto_project_row:
                            auto_project_selector = gr.Dropdown(
                                choices=get_projects_with_renders(),
                                value=None,
                                label="Select Project with Renders",
                                scale=4,
                                interactive=True
                            )
                            auto_project_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        max_objects_slider = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Maximum Number of Objects to Detect (per view)"
                        )

                        load_all_views_btn = gr.Button("Load All 4 Views", variant="secondary")
                        auto_segment_all_btn = gr.Button("Auto-Segment All Views", variant="primary")

                # Display all 4 views
                with gr.Row():
                    view1_image = gr.Image(label="View 1", type="pil")
                    view2_image = gr.Image(label="View 2", type="pil")
                    view3_image = gr.Image(label="View 3", type="pil")
                    view4_image = gr.Image(label="View 4", type="pil")

                # Display segmentation results for all 4 views
                gr.Markdown("### Segmentation Results")
                with gr.Row():
                    view1_result = gr.Image(label="View 1 - Segmented")
                    view2_result = gr.Image(label="View 2 - Segmented")
                    view3_result = gr.Image(label="View 3 - Segmented")
                    view4_result = gr.Image(label="View 4 - Segmented")

                auto_seg_status = gr.Textbox(label="Status", lines=4)

                # Gallery to show individual masks from all views
                individual_masks_gallery = gr.Gallery(
                    label="All Individual Binary Masks from All Views (White=Object, Black=Background)",
                    columns=5,
                    rows=4,
                    height="auto"
                )

                # Handle source selector change
                def update_auto_source_visibility(source):
                    if source == "Renders Directory":
                        return gr.Row(visible=True), gr.Row(visible=False)
                    else:
                        return gr.Row(visible=False), gr.Row(visible=True)

                auto_source_selector.change(
                    fn=update_auto_source_visibility,
                    inputs=[auto_source_selector],
                    outputs=[auto_renders_row, auto_project_row]
                )

                # Load all views (from appropriate source)
                def load_auto_views_from_source(source, object_uid, project_name):
                    if source == "Renders Directory":
                        return load_all_views(object_uid)
                    else:
                        return load_all_views_from_project(project_name)

                load_all_views_btn.click(
                    fn=load_auto_views_from_source,
                    inputs=[auto_source_selector, auto_object_selector, auto_project_selector],
                    outputs=[view1_image, view2_image, view3_image, view4_image]
                )

                # Segment all views
                auto_segment_all_btn.click(
                    fn=segment_all_views,
                    inputs=[view1_image, view2_image, view3_image, view4_image, max_objects_slider, current_project],
                    outputs=[view1_result, view2_result, view3_result, view4_result, individual_masks_gallery, auto_seg_status]
                )

                # Refresh button functionalities for Auto-Segment tab
                def refresh_auto_object_list():
                    objects = get_available_render_objects()
                    print(f"[Auto-Segment Refresh] Found {len(objects)} objects")  # Debug
                    return gr.Dropdown(choices=objects, value=objects[0] if objects else None)

                def refresh_auto_project_list():
                    projects = get_projects_with_renders()
                    return gr.Dropdown(choices=projects, value=projects[0] if projects else None)

                auto_refresh_btn.click(
                    fn=refresh_auto_object_list,
                    inputs=[],
                    outputs=[auto_object_selector]
                )

                auto_project_refresh_btn.click(
                    fn=refresh_auto_project_list,
                    inputs=[],
                    outputs=[auto_project_selector]
                )

            with gr.Tab("✏️ Brush Masking with SAM"):
                gr.Markdown("""
                **Draw on images + SAM automatic segmentation to find and merge overlapping segments**

                **Instructions:**
                1. Select image source: Renders Directory (evaluation data) or Project Renders (from your project's 00_renders folder)
                2. Make sure you've created or selected a project in the Project Management section above
                3. Click "Load All 4 Views" to load all multi-view images
                4. Draw on each view to roughly mark the area of interest
                5. Click "Process All Views with SAM" - SAM will automatically segment and merge all segments that overlap with your brushed areas
                6. Masks will be saved to your project's `01_sam_masks/` folder
                7. These masks can be used in the DreamEdit3D Training tab
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Image source selector
                        brush_source_selector = gr.Radio(
                            choices=["Renders Directory", "Project Renders"],
                            value="Renders Directory",
                            label="Image Source",
                            info="Choose where to load images from"
                        )

                        # Add object selector dropdown (for Renders Directory)
                        available_objects = get_available_render_objects()
                        print(f"[Brush Tab] Found {len(available_objects)} objects for dropdown")  # Debug

                        with gr.Row(visible=True) as brush_renders_row:
                            brush_object_selector = gr.Dropdown(
                                choices=available_objects,
                                value=DEFAULT_OBJECT_UID if (available_objects and DEFAULT_OBJECT_UID in available_objects) else (available_objects[0] if available_objects else None),
                                label="Select Object from Renders",
                                info=f"Choose from {len(available_objects)} available rendered objects" if available_objects else "No objects found in renders directory",
                                scale=4,
                                interactive=True
                            )
                            brush_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        # Project selector (for Project Renders)
                        with gr.Row(visible=False) as brush_project_row:
                            brush_project_selector = gr.Dropdown(
                                choices=get_projects_with_renders(),
                                value=None,
                                label="Select Project with Renders",
                                scale=4,
                                interactive=True
                            )
                            brush_project_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        load_brush_views_btn = gr.Button("Load All 4 Views", variant="secondary")
                        process_brush_btn = gr.Button("Process All Views with SAM", variant="primary")

                # Handle source selector change
                def update_brush_source_visibility(source):
                    if source == "Renders Directory":
                        return gr.Row(visible=True), gr.Row(visible=False)
                    else:
                        return gr.Row(visible=False), gr.Row(visible=True)

                brush_source_selector.change(
                    fn=update_brush_source_visibility,
                    inputs=[brush_source_selector],
                    outputs=[brush_renders_row, brush_project_row]
                )

                # Refresh button functionalities for Brush tab
                def refresh_brush_object_list():
                    objects = get_available_render_objects()
                    print(f"[Brush Refresh] Found {len(objects)} objects")  # Debug
                    return gr.Dropdown(choices=objects, value=objects[0] if objects else None)

                def refresh_brush_project_list():
                    projects = get_projects_with_renders()
                    return gr.Dropdown(choices=projects, value=projects[0] if projects else None)

                brush_refresh_btn.click(
                    fn=refresh_brush_object_list,
                    inputs=[],
                    outputs=[brush_object_selector]
                )

                brush_project_refresh_btn.click(
                    fn=refresh_brush_project_list,
                    inputs=[],
                    outputs=[brush_project_selector]
                )

                gr.Markdown("### Draw on Images (Brush Areas of Interest)")
                with gr.Row():
                    view1_brush_input = gr.ImageMask(
                        label="View 1 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )
                    view2_brush_input = gr.ImageMask(
                        label="View 2 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )
                    view3_brush_input = gr.ImageMask(
                        label="View 3 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )
                    view4_brush_input = gr.ImageMask(
                        label="View 4 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )

                gr.Markdown("### SAM Segmentation Results")
                with gr.Row():
                    view1_brush_result = gr.Image(label="View 1 - SAM Refined")
                    view2_brush_result = gr.Image(label="View 2 - SAM Refined")
                    view3_brush_result = gr.Image(label="View 3 - SAM Refined")
                    view4_brush_result = gr.Image(label="View 4 - SAM Refined")

                gr.Markdown("### Binary Masks")
                with gr.Row():
                    view1_brush_mask = gr.Image(label="View 1 - Binary Mask")
                    view2_brush_mask = gr.Image(label="View 2 - Binary Mask")
                    view3_brush_mask = gr.Image(label="View 3 - Binary Mask")
                    view4_brush_mask = gr.Image(label="View 4 - Binary Mask")

                brush_status = gr.Textbox(label="Status", lines=4)
                brush_saved_path = gr.State()

                # Load images for brushing (supports both sources)
                def load_all_views_for_brush(source, selected_object_uid, project_name):
                    """Load 4 views from the selected source"""
                    views = []

                    # Determine which source to use
                    if source == "Project Renders" and project_name:
                        print(f"[Brush Load] Loading views from project: {project_name}")
                        object_images = load_images_from_project_renders(project_name)
                    else:
                        # Default to renders directory
                        if not selected_object_uid:
                            selected_object_uid = DEFAULT_OBJECT_UID
                        print(f"[Brush Load] Loading views for object: {selected_object_uid}")
                        object_images = load_images_from_object(selected_object_uid)

                    print(f"[Brush Load] Found {len(object_images)} views")

                    # Load views in order: view_1, view_2, view_3, view_4 (front, back, left, right)
                    for view_name in ["view_1", "view_2", "view_3", "view_4"]:
                        if view_name in object_images:
                            img_path = object_images[view_name]
                            img = Image.open(img_path)
                            # Convert to numpy array for proper ImageMask format
                            img_array = np.array(img)
                            # Create the proper format for ImageMask with brush ready
                            views.append({
                                "background": img_array,
                                "layers": [],
                                "composite": img_array
                            })
                        else:
                            views.append(None)

                    while len(views) < 4:
                        views.append(None)

                    return views[0], views[1], views[2], views[3]

                load_brush_views_btn.click(
                    fn=load_all_views_for_brush,
                    inputs=[brush_source_selector, brush_object_selector, brush_project_selector],
                    outputs=[view1_brush_input, view2_brush_input, view3_brush_input, view4_brush_input]
                )

                # Process all views with brush + SAM
                process_brush_btn.click(
                    fn=process_all_views_with_brush,
                    inputs=[view1_brush_input, view2_brush_input, view3_brush_input, view4_brush_input, current_project],
                    outputs=[view1_brush_result, view2_brush_result, view3_brush_result, view4_brush_result,
                            view1_brush_mask, view2_brush_mask, view3_brush_mask, view4_brush_mask, brush_status, brush_saved_path]
                )

            with gr.Tab("✏️ Multi-Area Brush Masking"):
                gr.Markdown("""
                **Create multiple labeled mask areas (e.g., body, head) sequentially with SAM**

                **Instructions:**
                1. Select image source: Renders Directory (evaluation data) or Project Renders (from your project's 00_renders folder)
                2. Make sure you've created or selected a project in the Project Management section above
                3. Click "Load All 4 Views" to load all multi-view images

                **For Area 1:**
                4. Enter a label (e.g., "body") in the Area 1 Label field
                5. Draw on each view to mark Area 1
                6. Click "Process Area 1 with SAM" - creates mask_body.png

                **For Area 2:**
                7. Enter a label (e.g., "head") in the Area 2 Label field
                8. Draw on each view to mark Area 2
                9. Click "Process Area 2 with SAM" - creates mask_head.png

                Both masks are saved separately to `01_sam_masks/view_N/mask_<label>.png`
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Image source selector
                        multi_source_selector = gr.Radio(
                            choices=["Renders Directory", "Project Renders"],
                            value="Renders Directory",
                            label="Image Source",
                            info="Choose where to load images from"
                        )

                        # Add object selector dropdown (for Renders Directory)
                        available_objects_multi = get_available_render_objects()
                        print(f"[Multi-Area Brush Tab] Found {len(available_objects_multi)} objects for dropdown")

                        with gr.Row(visible=True) as multi_renders_row:
                            brush_multi_object_selector = gr.Dropdown(
                                choices=available_objects_multi,
                                value=DEFAULT_OBJECT_UID if (available_objects_multi and DEFAULT_OBJECT_UID in available_objects_multi) else (available_objects_multi[0] if available_objects_multi else None),
                                label="Select Object from Renders",
                                info=f"Choose from {len(available_objects_multi)} available rendered objects" if available_objects_multi else "No objects found in renders directory",
                                scale=4,
                                interactive=True
                            )
                            brush_multi_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        # Project selector (for Project Renders)
                        with gr.Row(visible=False) as multi_project_row:
                            multi_project_selector = gr.Dropdown(
                                choices=get_projects_with_renders(),
                                value=None,
                                label="Select Project with Renders",
                                scale=4,
                                interactive=True
                            )
                            multi_project_refresh_btn = gr.Button("🔄", scale=1, min_width=50)

                        load_brush_multi_views_btn = gr.Button("Load All 4 Views", variant="secondary")

                # Handle source selector change
                def update_multi_source_visibility(source):
                    if source == "Renders Directory":
                        return gr.Row(visible=True), gr.Row(visible=False)
                    else:
                        return gr.Row(visible=False), gr.Row(visible=True)

                multi_source_selector.change(
                    fn=update_multi_source_visibility,
                    inputs=[multi_source_selector],
                    outputs=[multi_renders_row, multi_project_row]
                )

                # Refresh button functionalities
                def refresh_brush_multi_object_list():
                    objects = get_available_render_objects()
                    print(f"[Multi-Area Brush Refresh] Found {len(objects)} objects")
                    return gr.Dropdown(choices=objects, value=objects[0] if objects else None)

                def refresh_multi_project_list():
                    projects = get_projects_with_renders()
                    return gr.Dropdown(choices=projects, value=projects[0] if projects else None)

                brush_multi_refresh_btn.click(
                    fn=refresh_brush_multi_object_list,
                    inputs=[],
                    outputs=[brush_multi_object_selector]
                )

                multi_project_refresh_btn.click(
                    fn=refresh_multi_project_list,
                    inputs=[],
                    outputs=[multi_project_selector]
                )

                # Area 1
                gr.Markdown("### Area 1 - Draw First Mask Area")
                area1_label = gr.Textbox(label="Area 1 Label", placeholder="e.g., body", value="body")

                gr.Markdown("#### Draw on Images for Area 1")
                with gr.Row():
                    view1_brush_area1 = gr.ImageMask(
                        label="View 1 - Area 1",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )
                    view2_brush_area1 = gr.ImageMask(
                        label="View 2 - Area 1",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )
                    view3_brush_area1 = gr.ImageMask(
                        label="View 3 - Area 1",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )
                    view4_brush_area1 = gr.ImageMask(
                        label="View 4 - Area 1",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"])
                    )

                process_area1_btn = gr.Button("Process Area 1 with SAM", variant="primary")

                gr.Markdown("#### Area 1 Results")
                with gr.Row():
                    view1_result_area1 = gr.Image(label="View 1 - Area 1 Result")
                    view2_result_area1 = gr.Image(label="View 2 - Area 1 Result")
                    view3_result_area1 = gr.Image(label="View 3 - Area 1 Result")
                    view4_result_area1 = gr.Image(label="View 4 - Area 1 Result")

                area1_status = gr.Textbox(label="Area 1 Status", lines=3)

                # Area 2
                gr.Markdown("### Area 2 - Draw Second Mask Area")
                area2_label = gr.Textbox(label="Area 2 Label", placeholder="e.g., head", value="head")

                gr.Markdown("#### Draw on Images for Area 2")
                with gr.Row():
                    view1_brush_area2 = gr.ImageMask(
                        label="View 1 - Area 2",
                        brush=gr.Brush(default_size=20, colors=["#FF00FF"])
                    )
                    view2_brush_area2 = gr.ImageMask(
                        label="View 2 - Area 2",
                        brush=gr.Brush(default_size=20, colors=["#FF00FF"])
                    )
                    view3_brush_area2 = gr.ImageMask(
                        label="View 3 - Area 2",
                        brush=gr.Brush(default_size=20, colors=["#FF00FF"])
                    )
                    view4_brush_area2 = gr.ImageMask(
                        label="View 4 - Area 2",
                        brush=gr.Brush(default_size=20, colors=["#FF00FF"])
                    )

                process_area2_btn = gr.Button("Process Area 2 with SAM", variant="primary")

                gr.Markdown("#### Area 2 Results")
                with gr.Row():
                    view1_result_area2 = gr.Image(label="View 1 - Area 2 Result")
                    view2_result_area2 = gr.Image(label="View 2 - Area 2 Result")
                    view3_result_area2 = gr.Image(label="View 3 - Area 2 Result")
                    view4_result_area2 = gr.Image(label="View 4 - Area 2 Result")

                area2_status = gr.Textbox(label="Area 2 Status", lines=3)

                # Load views for both areas
                def load_all_views_for_multi_area(source, selected_object_uid, project_name):
                    """Load 4 views from the selected source for both area inputs"""
                    views = []

                    # Determine which source to use
                    if source == "Project Renders" and project_name:
                        print(f"[Multi-Area Load] Loading views from project: {project_name}")
                        object_images = load_images_from_project_renders(project_name)
                    else:
                        # Default to renders directory
                        if not selected_object_uid:
                            selected_object_uid = DEFAULT_OBJECT_UID
                        print(f"[Multi-Area Load] Loading views for object: {selected_object_uid}")
                        object_images = load_images_from_object(selected_object_uid)

                    print(f"[Multi-Area Load] Found {len(object_images)} views")

                    # Load views in order
                    for view_name in ["view_1", "view_2", "view_3", "view_4"]:
                        if view_name in object_images:
                            img_path = object_images[view_name]
                            img = Image.open(img_path)
                            img_array = np.array(img)
                            views.append({
                                "background": img_array,
                                "layers": [],
                                "composite": img_array
                            })
                        else:
                            views.append(None)

                    while len(views) < 4:
                        views.append(None)

                    # Return for both area 1 and area 2 inputs (8 outputs total)
                    return views[0], views[1], views[2], views[3], views[0], views[1], views[2], views[3]

                load_brush_multi_views_btn.click(
                    fn=load_all_views_for_multi_area,
                    inputs=[multi_source_selector, brush_multi_object_selector, multi_project_selector],
                    outputs=[view1_brush_area1, view2_brush_area1, view3_brush_area1, view4_brush_area1,
                            view1_brush_area2, view2_brush_area2, view3_brush_area2, view4_brush_area2]
                )

                # Process Area 1
                process_area1_btn.click(
                    fn=process_all_views_with_brush_multi_area,
                    inputs=[view1_brush_area1, view2_brush_area1, view3_brush_area1, view4_brush_area1,
                           current_project, area1_label, gr.State(1)],
                    outputs=[view1_result_area1, view2_result_area1, view3_result_area1, view4_result_area1,
                            gr.State(), gr.State(), gr.State(), gr.State(), area1_status, gr.State()]
                )

                # Process Area 2
                process_area2_btn.click(
                    fn=process_all_views_with_brush_multi_area,
                    inputs=[view1_brush_area2, view2_brush_area2, view3_brush_area2, view4_brush_area2,
                           current_project, area2_label, gr.State(2)],
                    outputs=[view1_result_area2, view2_result_area2, view3_result_area2, view4_result_area2,
                            gr.State(), gr.State(), gr.State(), gr.State(), area2_status, gr.State()]
                )

            with gr.Tab("🌫️ Background Effects"):
                gr.Markdown("""
                **Remove or blur backgrounds using brush masks** (No SAM required)

                **Instructions:**
                1. Upload an image or select a default multi-view image, then draw mask on the **foreground object** you want to keep
                2. Choose "Blur Background" to blur everything except the masked area
                3. Choose "Extract Object" to isolate the masked object on a solid background
                """)

                with gr.Row():
                    with gr.Column():
                        # Default image selector
                        if DEFAULT_IMAGES:
                            bg_default_selector = gr.Dropdown(
                                choices=["None"] + list(DEFAULT_IMAGES.keys()),
                                value="None",
                                label="Select Default Multi-View Image"
                            )

                        bg_input = gr.ImageMask(label="Draw mask on foreground object")
                        blur_slider = gr.Slider(
                            minimum=1, maximum=30, value=10,
                            label="Blur Strength"
                        )

                        with gr.Row():
                            bg_remove_btn = gr.Button("Blur Background", variant="primary")
                            extract_btn = gr.Button("Extract Object", variant="secondary")

                    with gr.Column():
                        bg_output = gr.Image(label="Result")
                        bg_status = gr.Textbox(label="Status", interactive=False)

                        # Background color picker for extraction
                        bg_color = gr.ColorPicker(label="Background Color", value="#FFFFFF")

                # Load default image when selected (for ImageMask, need special handling)
                def load_image_for_mask_bg(view_name):
                    if view_name == "None":
                        return None
                    img = select_default_image(view_name)
                    if img:
                        return {"background": img, "layers": [], "composite": None}
                    return None

                if DEFAULT_IMAGES:
                    bg_default_selector.change(
                        fn=load_image_for_mask_bg,
                        inputs=[bg_default_selector],
                        outputs=[bg_input]
                    )

                bg_remove_btn.click(
                    fn=apply_background_removal,
                    inputs=[bg_input, blur_slider],
                    outputs=[bg_output, bg_status]
                )

                def extract_with_color(image_mask, color):
                    # Convert hex color to RGB tuple
                    color = color.lstrip('#')
                    rgb_color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                    return extract_masked_object(image_mask, rgb_color)

                extract_btn.click(
                    fn=extract_with_color,
                    inputs=[bg_input, bg_color],
                    outputs=[bg_output, bg_status]
                )

        with gr.Tab("🎨 DreamEdit3D Training"):
            gr.Markdown("""
            ## Train DreamEdit3D Models with Masked Images

            After creating masks using SAM, use them to train 3D-aware multi-concept models.
            """)

            def load_masked_folder(folder_path, view_index=1, selected_indices=None):
                """Load image and masks from a saved folder structure, returns (image, masks, status, concept_names)

                Args:
                    folder_path: Path to the folder containing masks
                    view_index: Which view to load (1-4 or "All Views")
                    selected_indices: List of mask indices to load (e.g., [0] or [1] or [0,1]). None means load all.
                """
                if not folder_path or not os.path.exists(folder_path):
                    return None, [], "Folder not found", ""

                folder = Path(folder_path)

                # Check if it's a multi-view structure
                view_dirs = sorted([d for d in folder.iterdir() if d.is_dir() and d.name.startswith('view_')])

                # Handle "All Views" option
                if view_index == "All Views" and view_dirs:
                    # Create a 2x2 grid showing all 4 views
                    from PIL import Image
                    import numpy as np

                    all_images = []
                    all_masks = []
                    view_numbers = []
                    mask_labels = []  # Track unique mask labels

                    for view_dir in view_dirs[:4]:  # Limit to 4 views
                        # Find image
                        img_path = None
                        for img_file in ["img.jpg", "img.png", "image.jpg", "image.png"]:
                            img_candidate = view_dir / img_file
                            if img_candidate.exists():
                                img_path = img_candidate
                                break

                        if img_path:
                            all_images.append(Image.open(img_path))
                            view_numbers.append(view_dir.name.replace('view_', ''))

                            # Collect only numbered masks from this view (mask0.png, mask1.png, etc)
                            # Ignore labeled masks (mask_body.png, mask_head.png) to avoid duplicates
                            all_mask_files = list(view_dir.glob("mask*.png"))
                            numbered_masks = sorted([f for f in all_mask_files if f.stem.replace("mask", "").isdigit()])
                            labeled_masks = sorted([f for f in all_mask_files if f.stem.startswith("mask_")])

                            # Filter by selected indices if specified
                            if selected_indices is not None:
                                numbered_masks = [m for m in numbered_masks if int(m.stem.replace("mask", "")) in selected_indices]
                                labeled_masks = [m for i, m in enumerate(sorted(labeled_masks)) if i in selected_indices]

                            # Use numbered masks for training
                            all_masks.extend([str(f) for f in numbered_masks])

                            # Extract labels from labeled mask filenames for concept names
                            for mask_file in labeled_masks:
                                label = mask_file.stem[5:]  # Remove "mask_" prefix (e.g., mask_body.png -> body)
                                if label and label not in mask_labels:
                                    mask_labels.append(label)

                    # After collecting all views, try GPT-4V auto-detection if no labeled masks
                    if not mask_labels and all_masks and all_images:
                        print("  🤖 Attempting GPT-4V auto-detection for concept names...")
                        try:
                            # Use first view's image for detection
                            first_img_path = view_dirs[0] / "img.jpg"
                            if not first_img_path.exists():
                                first_img_path = view_dirs[0] / "img.png"

                            # Get first view's masks for detection
                            first_view_masks = [m for m in all_masks if view_dirs[0].name in m]

                            if first_img_path.exists() and first_view_masks:
                                detected_names = detect_concept_names_batch(
                                    image_path=str(first_img_path),
                                    mask_paths=first_view_masks,
                                    api_key=os.environ.get('OPENAI_API_KEY')
                                )
                                mask_labels = detected_names
                                print(f"  ✓ Auto-detected: {', '.join(detected_names)}")
                        except Exception as e:
                            print(f"  ⚠ Auto-detection failed: {e}, using generic names")

                    # Final fallback: use generic numbering if still no labels
                    if not mask_labels and all_masks:
                        num_masks_per_view = len(all_masks) // len(all_images) if all_images else 1
                        for i in range(num_masks_per_view):
                            mask_labels.append(f"object{i}" if i > 0 else "object")

                    if len(all_images) >= 4:
                        # Create 2x2 grid
                        img_width, img_height = all_images[0].size
                        grid_img = Image.new('RGB', (img_width * 2, img_height * 2))

                        for idx, img in enumerate(all_images[:4]):
                            x = (idx % 2) * img_width
                            y = (idx // 2) * img_height
                            grid_img.paste(img, (x, y))

                        # Save grid to temp location
                        temp_grid_path = folder / "all_views_grid.jpg"
                        grid_img.save(temp_grid_path, quality=95)

                        # Generate concept names from labels
                        concept_names = ", ".join(mask_labels) if mask_labels else ""

                        # Return grid image, all masks, status, and concept names
                        return str(temp_grid_path), all_masks, f"✓ Loaded view {','.join(view_numbers)}: {len(all_masks)} mask(s) from {folder_path}", concept_names
                    else:
                        return None, [], f"Not enough views found (found {len(all_images)}, need 4)", ""

                # Handle single view selection
                if view_dirs:
                    # Use specified view
                    view_dir = folder / f"view_{view_index}"
                    if not view_dir.exists():
                        view_dir = view_dirs[0]  # Fallback to first view
                else:
                    view_dir = folder

                # Find main image
                main_image = None
                for img_file in ["img.jpg", "img.png", "image.jpg", "image.png"]:
                    img_path = view_dir / img_file
                    if img_path.exists():
                        main_image = str(img_path)
                        break

                # Find mask files - only numbered masks for training
                all_mask_files = list(view_dir.glob("mask*.png"))
                numbered_masks = sorted([f for f in all_mask_files if f.stem.replace("mask", "").isdigit()])
                labeled_masks = sorted([f for f in all_mask_files if f.stem.startswith("mask_")])

                # Filter by selected indices if specified
                if selected_indices is not None:
                    numbered_masks = [m for m in numbered_masks if int(m.stem.replace("mask", "")) in selected_indices]
                    labeled_masks = [m for i, m in enumerate(sorted(labeled_masks)) if i in selected_indices]

                # Extract concept names from labeled mask filenames
                mask_labels = []
                for mask_file in labeled_masks:
                    label = mask_file.stem[5:]  # Remove "mask_" prefix
                    if label:
                        mask_labels.append(label)

                # If no labeled masks, try GPT-4V auto-detection
                if not mask_labels and numbered_masks and main_image:
                    print("  🤖 Attempting GPT-4V auto-detection for concept names...")
                    try:
                        detected_names = detect_concept_names_batch(
                            image_path=main_image,
                            mask_paths=[str(m) for m in numbered_masks],
                            api_key=os.environ.get('OPENAI_API_KEY')
                        )
                        mask_labels = detected_names
                        print(f"  ✓ Auto-detected: {', '.join(detected_names)}")
                    except Exception as e:
                        print(f"  ⚠ Auto-detection failed: {e}, using generic names")
                        # Fallback to generic names
                        for i in range(len(numbered_masks)):
                            mask_labels.append(f"object{i}" if i > 0 else "object")

                # Final fallback: generic names
                if not mask_labels and numbered_masks:
                    for i in range(len(numbered_masks)):
                        mask_labels.append(f"object{i}" if i > 0 else "object")

                concept_names = ", ".join(mask_labels) if mask_labels else ""

                if main_image and numbered_masks:
                    return main_image, [str(m) for m in numbered_masks], f"Loaded view {view_index}: {len(numbered_masks)} mask(s) from {folder_path}", concept_names
                else:
                    return None, [], "No valid image/masks found in folder", ""

            # Initialize DreamEdit3D app
            d3d_app = DreamEdit3DApp()

            # Wrapper functions to integrate with project system
            def get_available_models_with_projects():
                """Get models from both trained_models and projects"""
                import json
                models = []

                # Get models from standard location
                models.extend(d3d_app.get_available_models())

                # Get models from projects
                if project_manager:
                    base_dir = Path(__file__).parent
                    for project_path in project_manager.get_all_projects():
                        project_models_dir = Path(project_path) / "02_trained_model"
                        if project_models_dir.exists():
                            for model_dir in project_models_dir.glob("model_*"):
                                if model_dir.is_dir():
                                    project_name = Path(project_path).name
                                    info_file = model_dir / "model_info.json"
                                    if info_file.exists():
                                        with open(info_file) as f:
                                            info = json.load(f)
                                        display_name = f"[Project: {project_name}] {model_dir.name} ({info.get('timestamp', 'unknown')})"
                                    else:
                                        display_name = f"[Project: {project_name}] {model_dir.name}"
                                    models.append((display_name, str(model_dir)))

                return sorted(models, reverse=True, key=lambda x: x[0])

            def get_models_from_current_project(project_path):
                """Get models only from the currently selected project"""
                import json
                models = []

                if not project_path:
                    return []

                project_models_dir = Path(project_path) / "02_trained_model"
                if project_models_dir.exists():
                    for model_dir in project_models_dir.glob("model_*"):
                        if model_dir.is_dir():
                            info_file = model_dir / "model_info.json"
                            if info_file.exists():
                                with open(info_file) as f:
                                    info = json.load(f)
                                display_name = f"{model_dir.name} ({info.get('timestamp', 'unknown')})"
                            else:
                                display_name = model_dir.name
                            models.append((display_name, str(model_dir)))

                return sorted(models, reverse=True, key=lambda x: x[0])

            def refresh_models_with_projects():
                """Refresh model list including projects"""
                return gr.Dropdown(choices=get_available_models_with_projects())

            def train_model_with_project(main_image, mask_files, concept_names, phase1_steps, phase2_steps, mvdream_training_mode, project_path):
                """Train model and save directly to project folder"""
                if not project_path:
                    return "❌ Please select or create a project first", None, None

                # Set project directories for temp workspace and model output
                project_temp_dir = Path(project_path) / "temp_workspace"
                project_models_dir = Path(project_path) / "02_trained_model"

                # Check if "All Views" was selected (grid image or masks from all views)
                multiview_data_dir = None
                sam_masks_dir = Path(project_path) / "01_sam_masks"

                if sam_masks_dir.exists() and isinstance(main_image, str):
                    # Check if image is the grid (all_views_grid.jpg) or if we have 4+ mask files
                    is_grid_image = "all_views_grid.jpg" in main_image
                    has_multiview_masks = mask_files and len(mask_files) >= 4

                    if is_grid_image or has_multiview_masks:
                        # Multi-view mode - check if we have all 4 views
                        view_dirs = sorted([d for d in sam_masks_dir.iterdir() if d.is_dir() and d.name.startswith('view_')])
                        if len(view_dirs) >= 4:
                            # Create a temporary directory with only the selected masks
                            import shutil
                            temp_multiview_dir = project_temp_dir / "selected_masks"
                            temp_multiview_dir.mkdir(parents=True, exist_ok=True)

                            # Determine which mask indices are being used based on the loaded mask_files
                            mask_indices_used = set()
                            for mask_file in mask_files:
                                mask_name = Path(mask_file).stem
                                if mask_name.replace("mask", "").isdigit():
                                    mask_indices_used.add(int(mask_name.replace("mask", "")))

                            # Copy only the selected masks to temp directory
                            for view_dir in view_dirs:
                                temp_view_dir = temp_multiview_dir / view_dir.name
                                temp_view_dir.mkdir(exist_ok=True)

                                # Copy image
                                img_path = view_dir / "img.jpg"
                                if img_path.exists():
                                    shutil.copy(img_path, temp_view_dir / "img.jpg")

                                # Copy only selected masks
                                for idx in sorted(mask_indices_used):
                                    mask_path = view_dir / f"mask{idx}.png"
                                    if mask_path.exists():
                                        # Renumber masks sequentially starting from 0
                                        new_idx = list(sorted(mask_indices_used)).index(idx)
                                        shutil.copy(mask_path, temp_view_dir / f"mask{new_idx}.png")

                            multiview_data_dir = str(temp_multiview_dir)

                # Train model directly to project folders
                status, output, model_path = d3d_app.train_model(
                    main_image, mask_files, concept_names, phase1_steps, phase2_steps,
                    temp_dir=str(project_temp_dir),
                    models_dir=str(project_models_dir),
                    multiview_data_dir=multiview_data_dir,
                    mvdream_training_mode=mvdream_training_mode
                )

                # Update project completion and save prompts
                if model_path and project_manager:
                    project_manager.update_project_step(project_path, "DreamEdit3D Training")
                    # Save concept names used in training
                    if concept_names:
                        prompts_list = [p.strip() for p in concept_names.split(",") if p.strip()]
                        project_manager.save_prompts(
                            project_path,
                            "training",
                            prompts_list,
                            metadata={"model_path": model_path, "phase1_steps": phase1_steps, "phase2_steps": phase2_steps}
                        )

                return status, output, model_path

            def generate_image_with_project(model_dropdown, prompt, filename, project_path,
                                           size, num_frames, steps, scale,
                                           camera_elev, camera_azim, camera_azim_span,
                                           negative_prompt, positive_prompt, elevation_list_str="",
                                           num_generations=1, seed=-1):
                """Generate image and save directly to project folder with custom parameters"""
                if not project_path:
                    return "❌ Please select or create a project first", None

                # Set output directory to project folder
                project_images_dir = Path(project_path) / "03_multiview_images"

                # Parse elevation_list if provided
                elevation_list_param = elevation_list_str.strip() if elevation_list_str and elevation_list_str.strip() else None

                # Generate image directly to project folder with all parameters
                status, image = d3d_app.generate_image(
                    model_dropdown, prompt, filename,
                    output_dir=str(project_images_dir),
                    size=size,
                    num_frames=num_frames,
                    steps=steps,
                    scale=scale,
                    camera_elev=camera_elev,
                    camera_azim=camera_azim,
                    camera_azim_span=camera_azim_span,
                    negative_prompt=negative_prompt,
                    positive_prompt=positive_prompt,
                    elevation_list=elevation_list_param,
                    num_generations=int(num_generations),
                    seed=int(seed)
                )

                # Update project completion and save prompt
                if image and project_manager:
                    project_manager.update_project_step(project_path, "Multiview Generation")
                    # Save inference prompt with actual generated filename
                    if prompt:
                        # Extract actual generated filename from image path
                        generated_filename = Path(image).name if isinstance(image, str) else filename

                        # Calculate azimuth and elevation for each view (single batch, mixed elevations)
                        angle_gap = camera_azim_span / num_frames
                        azimuth_list = [(camera_azim + i * angle_gap) % 360 for i in range(num_frames)]

                        # Handle per-view elevations (cycled through input elevations)
                        if elevation_list_param:
                            input_elevations = [float(e.strip()) for e in elevation_list_param.split(',')]
                            elevation_list = [input_elevations[i % len(input_elevations)] for i in range(num_frames)]
                        else:
                            elevation_list = [camera_elev] * num_frames

                        project_manager.save_prompts(
                            project_path,
                            "inference",
                            prompt,
                            metadata={
                                "model": model_dropdown,
                                "generated_file": generated_filename,
                                "full_path": str(image) if isinstance(image, str) else None,
                                "elevation": camera_elev,
                                "azimuth_start": camera_azim,
                                "azimuth_span": camera_azim_span,
                                "elevation_list": elevation_list,
                                "azimuth_list": azimuth_list
                            }
                        )

                return status, image

            def get_masked_output_folders():
                """Get all folders from masked_output directory and projects"""
                base_dir = Path(__file__).parent
                folders = []

                # Add folders from mask/masked_output (legacy)
                masked_output_dir = base_dir / "mask" / "masked_output"
                if masked_output_dir.exists():
                    for f in masked_output_dir.iterdir():
                        if f.is_dir():
                            folders.append(("Legacy: " + f.name, str(f)))

                # Add folders from projects (new system)
                if project_manager:
                    for project_path in project_manager.get_all_projects():
                        sam_masks_dir = Path(project_path) / "01_sam_masks"
                        if sam_masks_dir.exists():
                            project_name = Path(project_path).name
                            folders.append(("Project: " + project_name, str(sam_masks_dir)))

                return sorted(folders, reverse=True, key=lambda x: x[0])  # Most recent first

            def load_from_current_project(project_path, view_index, mask_selection):
                """Load SAM masks from current project and display preview"""
                if not project_path:
                    return None, [], "", "❌ No project selected. Please create or select a project first.", None, None, None, None, None

                sam_masks_path = Path(project_path) / "01_sam_masks"
                if not sam_masks_path.exists():
                    return None, [], "", f"❌ No SAM masks found in project. Run SAM segmentation first.", None, None, None, None, None

                # Determine which mask indices to load based on selection
                selected_indices = []
                if "mask0 (first area)" in mask_selection:
                    selected_indices.append(0)
                if "mask1 (second area)" in mask_selection:
                    selected_indices.append(1)

                if not selected_indices:
                    return None, [], "", "❌ Please select at least one mask area.", None, None, None, None, None

                # Load image, masks, status, and auto-extracted concept names
                image, masks, status, concept_names = load_masked_folder(str(sam_masks_path), view_index, selected_indices)

                # Determine which image to show in preview based on selection
                preview_image = None
                if view_index == "All Views":
                    # Show the grid image for all views
                    grid_path = sam_masks_path / "all_views_grid.jpg"
                    if grid_path.exists():
                        preview_image = str(grid_path)
                    else:
                        # If grid doesn't exist, use the first view
                        view_dir = sam_masks_path / "view_1"
                        if view_dir.exists():
                            for img_file in ["img.jpg", "img.png", "image.jpg", "image.png"]:
                                img_candidate = view_dir / img_file
                                if img_candidate.exists():
                                    preview_image = str(img_candidate)
                                    break
                else:
                    # Show only the selected view
                    if isinstance(view_index, int) and 1 <= view_index <= 4:
                        view_dir = sam_masks_path / f"view_{view_index}"
                        if view_dir.exists():
                            for img_file in ["img.jpg", "img.png", "image.jpg", "image.png"]:
                                img_candidate = view_dir / img_file
                                if img_candidate.exists():
                                    preview_image = str(img_candidate)
                                    break

                # concept_names is now auto-populated from mask filenames (e.g., "body, head")
                # User can still edit if needed

                # Return: main_image(hidden), masks, concept_names, status, preview_image, view1, view2, view3, view4 (all hidden)
                return image, masks, concept_names, status, preview_image, None, None, None, None

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Step 1: Prepare Your Data")

                    # Load from current project
                    gr.Markdown("**Load from Current Project**")
                    with gr.Row():
                        d3d_load_project_btn = gr.Button("📁 Load", variant="primary", scale=2)
                        d3d_view_selector = gr.Dropdown(
                            label="View",
                            choices=["All Views", 1, 2, 3, 4],
                            value="All Views",
                            scale=1
                        )

                    with gr.Row():
                        d3d_mask_selector = gr.CheckboxGroup(
                            label="Select Masks to Load",
                            choices=["mask0 (first area)", "mask1 (second area)"],
                            value=["mask0 (first area)", "mask1 (second area)"],
                            info="Choose which mask areas to use for training"
                        )

                    d3d_load_status = gr.Textbox(label="Load Status", max_lines=2)

                    d3d_mask_files = gr.File(
                        label="Loaded Masks",
                        file_count="multiple",
                        height=150,
                        interactive=False
                    )
                    d3d_concept_names = gr.Textbox(
                        label="Concept Names (comma-separated)",
                        placeholder="e.g., creature, bowl, stone",
                        info="Enter one name per mask",
                        interactive=True
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Preview")
                    d3d_preview_image = gr.Image(
                        label="Loaded Data Preview",
                        type="filepath",
                        interactive=False
                    )

                    # Hidden components for backward compatibility
                    d3d_main_image = gr.Textbox(visible=False)
                    d3d_preview_view1 = gr.Textbox(visible=False)
                    d3d_preview_view2 = gr.Textbox(visible=False)
                    d3d_preview_view3 = gr.Textbox(visible=False)
                    d3d_preview_view4 = gr.Textbox(visible=False)

            gr.Markdown("### Step 2: Training Parameters")
            with gr.Row():
                d3d_phase1_steps = gr.Number(
                    label="Phase 1 Steps",
                    value=400,
                    minimum=100,
                    maximum=2000,
                    step=50
                )
                d3d_phase2_steps = gr.Number(
                    label="Phase 2 Steps",
                    value=400,
                    minimum=100,
                    maximum=2000,
                    step=50
                )

            with gr.Accordion("⚙️ Advanced Training Parameters", open=False):
                gr.Markdown("**Learning Rate & Optimization**")
                with gr.Row():
                    d3d_learning_rate = gr.Number(
                        label="Learning Rate",
                        value=2e-6,
                        info="Learning rate for fine-tuning"
                    )
                    d3d_initial_learning_rate = gr.Number(
                        label="Initial Learning Rate",
                        value=5e-4,
                        info="Learning rate for textual inversion phase"
                    )
                    d3d_lr_scheduler = gr.Dropdown(
                        label="LR Scheduler",
                        choices=["constant", "linear", "cosine", "cosine_with_restarts", "polynomial", "constant_with_warmup"],
                        value="constant",
                        info="Learning rate scheduler type"
                    )

                with gr.Row():
                    d3d_lr_warmup_steps = gr.Number(
                        label="LR Warmup Steps",
                        value=0,
                        minimum=0,
                        step=10,
                        info="Number of warmup steps for LR scheduler"
                    )
                    d3d_adam_weight_decay = gr.Number(
                        label="Adam Weight Decay",
                        value=1e-2,
                        info="Weight decay for Adam optimizer"
                    )
                    d3d_max_grad_norm = gr.Number(
                        label="Max Gradient Norm",
                        value=1.0,
                        info="Maximum gradient norm for clipping"
                    )

                gr.Markdown("**Training Configuration**")
                with gr.Row():
                    d3d_train_batch_size = gr.Number(
                        label="Train Batch Size",
                        value=1,
                        minimum=1,
                        maximum=8,
                        step=1,
                        info="Batch size per device"
                    )
                    d3d_resolution = gr.Number(
                        label="Training Resolution",
                        value=256,
                        minimum=128,
                        maximum=512,
                        step=64,
                        info="Resolution for training images"
                    )
                    d3d_size = gr.Number(
                        label="Output Size",
                        value=256,
                        minimum=128,
                        maximum=512,
                        step=64,
                        info="Output image size"
                    )

                with gr.Row():
                    d3d_mixed_precision = gr.Dropdown(
                        label="Mixed Precision",
                        choices=["no", "fp16", "bf16"],
                        value="fp16",
                        info="Use mixed precision training"
                    )
                    d3d_gradient_checkpointing = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=False,
                        info="Use gradient checkpointing to save memory"
                    )
                    d3d_use_8bit_adam = gr.Checkbox(
                        label="Use 8-bit Adam",
                        value=True,
                        info="Use 8-bit Adam optimizer"
                    )
                    d3d_set_grads_to_none = gr.Checkbox(
                        label="Set Grads to None",
                        value=True,
                        info="Set gradients to None instead of zero"
                    )

                gr.Markdown("**MVDream & Camera Settings**")
                with gr.Row():
                    d3d_num_frames = gr.Number(
                        label="Number of Frames/Views",
                        value=4,
                        minimum=1,
                        maximum=8,
                        step=1,
                        info="Number of views to generate"
                    )
                    d3d_mvdream_training_mode = gr.Dropdown(
                        label="MVDream Training Mode",
                        choices=["2d", "3d", "mixed"],
                        value="2d",
                        info="2d: Standard training (8-12GB VRAM) | 3d: Multi-view consistency (16GB+ VRAM) | mixed: Balance (30% 3D)"
                    )
                    d3d_model_name = gr.Textbox(
                        label="Model Name",
                        value="sd-v2.1-base-4view",
                        info="Pre-trained model name"
                    )

                with gr.Row():
                    d3d_camera_elev = gr.Number(
                        label="Camera Elevation",
                        value=15,
                        minimum=-90,
                        maximum=90,
                        step=5,
                        info="Camera elevation angle"
                    )
                    d3d_camera_azim = gr.Number(
                        label="Camera Azimuth Start",
                        value=90,
                        minimum=0,
                        maximum=360,
                        step=15,
                        info="Starting azimuth angle"
                    )
                    d3d_camera_azim_span = gr.Number(
                        label="Camera Azimuth Span",
                        value=360,
                        minimum=0,
                        maximum=360,
                        step=30,
                        info="Azimuth span for views"
                    )
                    d3d_use_camera = gr.Checkbox(
                        label="Use Camera",
                        value=True,
                        info="Use camera conditioning"
                    )

                gr.Markdown("**Logging & Misc**")
                with gr.Row():
                    d3d_seed = gr.Number(
                        label="Random Seed",
                        value=23,
                        minimum=0,
                        maximum=999999,
                        step=1,
                        info="Random seed for reproducibility"
                    )
                    d3d_img_log_steps = gr.Number(
                        label="Image Log Steps",
                        value=100,
                        minimum=10,
                        maximum=500,
                        step=10,
                        info="Steps between logging images"
                    )
                    d3d_lambda_attention = gr.Number(
                        label="Lambda Attention",
                        value=1e-2,
                        info="Attention loss weight"
                    )

                with gr.Row():
                    d3d_train_text_encoder = gr.Checkbox(
                        label="Train Text Encoder",
                        value=True,
                        info="Whether to train the text encoder"
                    )
                    d3d_center_crop = gr.Checkbox(
                        label="Center Crop",
                        value=False,
                        info="Center crop images to resolution"
                    )
                    d3d_apply_masked_loss = gr.Checkbox(
                        label="Apply Masked Loss",
                        value=True,
                        info="Use masked loss for training"
                    )

            d3d_train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

            with gr.Row():
                d3d_training_status = gr.Textbox(label="Training Status", max_lines=5)
                d3d_training_output = gr.Textbox(label="Training Output", max_lines=10)

            d3d_trained_model_path = gr.Textbox(label="Trained Model Path", visible=False)

            # Connect "Load from Current Project" button
            d3d_load_project_btn.click(
                fn=load_from_current_project,
                inputs=[current_project, d3d_view_selector, d3d_mask_selector],
                outputs=[d3d_main_image, d3d_mask_files, d3d_concept_names, d3d_load_status,
                        d3d_preview_image, d3d_preview_view1, d3d_preview_view2, d3d_preview_view3, d3d_preview_view4]
            )

            # Connect training (with project integration)
            d3d_train_btn.click(
                fn=train_model_with_project,
                inputs=[d3d_main_image, d3d_mask_files, d3d_concept_names, d3d_phase1_steps, d3d_phase2_steps, d3d_mvdream_training_mode, current_project],
                outputs=[d3d_training_status, d3d_training_output, d3d_trained_model_path]
            )

            gr.Markdown("### Step 3: Generate Images")
            with gr.Row():
                with gr.Column():
                    d3d_model_dropdown = gr.Dropdown(
                        label="Select Trained Model (from current project)",
                        choices=[],
                        interactive=True
                    )

                    d3d_refresh_btn = gr.Button("🔄 Refresh Models")

                    d3d_prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="e.g., a photo of <asset0> at the beach",
                        info="Use <asset0>, <asset1>, etc. for your concepts"
                    )

                    d3d_output_filename = gr.Textbox(
                        label="Output Filename (optional)",
                        placeholder="generated_image"
                    )

                    # Advanced generation parameters
                    with gr.Accordion("⚙️ Advanced Generation Parameters", open=False):
                        with gr.Row():
                            d3d_size = gr.Slider(
                                label="Image Size",
                                minimum=128,
                                maximum=512,
                                step=64,
                                value=256,
                                info="Resolution of generated images (e.g., 256x256)"
                            )
                            d3d_num_frames = gr.Slider(
                                label="Number of Views",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=4,
                                info="How many views to generate"
                            )

                        with gr.Row():
                            d3d_steps = gr.Slider(
                                label="Inference Steps",
                                minimum=20,
                                maximum=100,
                                step=5,
                                value=50,
                                info="More steps = better quality but slower"
                            )
                            d3d_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=15.0,
                                step=0.5,
                                value=7.5,
                                info="How closely to follow the prompt"
                            )

                        with gr.Row():
                            d3d_camera_elev = gr.Slider(
                                label="Camera Elevation",
                                minimum=-30,
                                maximum=60,
                                step=5,
                                value=15,
                                info="Vertical camera angle in degrees (used if elevation list is empty)"
                            )
                            d3d_camera_azim = gr.Slider(
                                label="Camera Azimuth Start",
                                minimum=0,
                                maximum=360,
                                step=15,
                                value=90,
                                info="Starting horizontal camera angle"
                            )
                            d3d_camera_azim_span = gr.Slider(
                                label="Camera Azimuth Span",
                                minimum=90,
                                maximum=360,
                                step=30,
                                value=360,
                                info="Total horizontal rotation range"
                            )

                        d3d_elevation_list = gr.Textbox(
                            label="Elevation List (Per-View)",
                            value="",
                            placeholder="e.g., 20,-10 for alternating elevations",
                            info="Comma-separated elevations cycled per view (e.g., '20,-10' → views at 20°,-10°,20°,-10°). Better for 3D reconstruction!"
                        )

                        d3d_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            value="blurry, low quality, bad anatomy, distorted, ugly, noisy, artifacts, poorly rendered",
                            placeholder="Features to avoid in generation",
                            info="Helps improve image quality by avoiding unwanted features"
                        )

                        d3d_positive_prompt = gr.Textbox(
                            label="Positive Prompt (Quality Boost)",
                            value="high quality, detailed, sharp focus, professional lighting, good contrast, studio lighting, well-lit, crisp details",
                            placeholder="Quality keywords appended to your prompt",
                            info="Appended to your prompt to enhance quality"
                        )

                        with gr.Row():
                            d3d_num_generations = gr.Slider(
                                label="Number of Generations",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=1,
                                info="Generate multiple 1×N images in one run (model loads once, much faster!)"
                            )
                            d3d_seed = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="Random seed (-1 for random). First generation uses this seed."
                            )

                    d3d_generate_btn = gr.Button("🎨 Generate Image", variant="primary")

                with gr.Column():
                    d3d_generated_image = gr.Image(label="Generated Multi-View Image", height=400)
                    d3d_generation_status = gr.Textbox(label="Generation Status")

            # Update model dropdown when project changes
            project_dropdown.change(
                fn=lambda project_path: gr.Dropdown(choices=get_models_from_current_project(project_path)),
                inputs=[project_dropdown],
                outputs=[d3d_model_dropdown]
            )

            # Refresh models from current project
            d3d_refresh_btn.click(
                fn=lambda project_path: gr.Dropdown(choices=get_models_from_current_project(project_path)),
                inputs=[current_project],
                outputs=[d3d_model_dropdown]
            )

            d3d_generate_btn.click(
                fn=generate_image_with_project,
                inputs=[
                    d3d_model_dropdown, d3d_prompt_input, d3d_output_filename, current_project,
                    d3d_size, d3d_num_frames, d3d_steps, d3d_scale,
                    d3d_camera_elev, d3d_camera_azim, d3d_camera_azim_span,
                    d3d_negative_prompt, d3d_positive_prompt, d3d_elevation_list,
                    d3d_num_generations, d3d_seed
                ],
                outputs=[d3d_generation_status, d3d_generated_image]
            )

        with gr.Tab("🎯 GTR: Multi-view to 3D"):
            gr.Markdown("""
            # 🎯 GTR: Multi-view to 3D Mesh

            Transform multi-view images into high-quality 3D meshes using the GTR pipeline.
            """)

            with gr.Tab("⚡ Full Pipeline"):
                gr.Markdown("""
                ### Input Selection
                Choose your input source (priority: folder → generated images → uploaded file → examples)
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Add image selector for multi-view images (filtered by current project)
                        with gr.Group():
                            gr.Markdown("**🎨 Multi-view Images**")
                            with gr.Row():
                                gtr_image_dropdown = gr.Dropdown(
                                    choices=[],  # Will be populated when project is selected
                                    label="Select Image",
                                    value=None,
                                    info="Generated multiview images from selected project",
                                    container=False
                                )
                                gtr_refresh_images_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)


                        with gr.Group():
                            gr.Markdown("**⚙️ Configuration**")
                            gtr_checkpoint = gr.Textbox(
                                value=GTR_CKPT_PATH,
                                label="Checkpoint Path",
                                info="Path to GTR model checkpoint"
                            )

                            gr.Markdown("**📷 Input Prepare Parameters** (describe how input images were captured)")
                            with gr.Row():
                                gtr_cam_radius = gr.Number(
                                    value=4.0,
                                    label="Camera Radius",
                                    info="Default: 4 for 4/6-view, 3 for directory format",
                                    precision=2,
                                    step=0.1
                                )
                                gtr_fov = gr.Number(
                                    value=30.0,
                                    label="FOV",
                                    info="Default: 30 for 4/6-view, 60 for directory format",
                                    precision=2,
                                    step=0.1
                                )
                                gtr_elevation = gr.Number(
                                    value=15.0,
                                    label="Elevation (fallback)",
                                    info="Only used if Elevation List is empty",
                                    precision=2,
                                    step=0.1
                                )
                                gtr_azimuth_start = gr.Number(
                                    value=90.0,
                                    label="Azimuth Start (fallback)",
                                    info="Only used if Azimuth List is empty",
                                    precision=1,
                                    step=15
                                )

                            gr.Markdown("**📷 Per-View Camera Angles** (auto-loaded from project_info.json, takes precedence over fallback values)")
                            with gr.Row():
                                gtr_elevation_list = gr.Textbox(
                                    value="",
                                    label="Elevation List",
                                    info="Per-view elevations → theta = 90 - elevation",
                                    placeholder="e.g., 20,-10,20,-10"
                                )
                                gtr_azimuth_list = gr.Textbox(
                                    value="",
                                    label="Azimuth List",
                                    info="Per-view azimuths → phi = 90 - azimuth",
                                    placeholder="e.g., 90,180,270,0"
                                )

                            with gr.Row():
                                gtr_padding = gr.Number(
                                    value=0,
                                    label="Padding (px)",
                                    info="Padding around image to fix boundary blur (0=disabled)",
                                    precision=0,
                                    step=10,
                                    minimum=0,
                                    maximum=200
                                )

                            gr.Markdown("**🎬 Render Parameters** (for output GIF visualization)")
                            with gr.Row():
                                gtr_render_cam_distance = gr.Number(
                                    value=3.5,
                                    label="Render Distance",
                                    info="Camera distance for GIF rendering",
                                    precision=2,
                                    step=0.1
                                )
                                gtr_render_fov = gr.Number(
                                    value=50.0,
                                    label="Render FOV",
                                    info="FOV for GIF rendering",
                                    precision=2,
                                    step=0.1
                                )
                                gtr_render_elevation = gr.Number(
                                    value=20.0,
                                    label="Render Elevation",
                                    info="Elevation for GIF rendering",
                                    precision=2,
                                    step=0.1
                                )
                                gtr_render_num_frames = gr.Number(
                                    value=50,
                                    label="Frames",
                                    info="Number of frames in GIF",
                                    precision=0,
                                    step=1
                                )

                            gtr_enable_mesh_export = gr.Checkbox(
                                value=True,
                                label="📦 Export Mesh (OBJ)",
                                info="Extract and export mesh.obj file"
                            )
                            gtr_export_glb = gr.Checkbox(
                                value=True,
                                label="📦 Export GLB",
                                info="Also export mesh.glb (for web/AR/VR)"
                            )
                            gtr_enable_mesh_gif = gr.Checkbox(
                                value=False,
                                label="🎬 Render Mesh GIF",
                                info="Render mesh.gif animation (slower)"
                            )
                            gtr_transparent_bg = gr.Checkbox(
                                value=False,
                                label="🔲 Transparent Background",
                                info="Render GIF with transparent background"
                            )

                        with gr.Row():
                            gtr_run_btn = gr.Button("🚀 Run Single", variant="primary", size="lg")
                            gtr_run_all_btn = gr.Button("🚀 Run All Images", variant="secondary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Results")
                        with gr.Group():
                            with gr.Row():
                                gtr_output_mesh = gr.File(label="💎 OBJ Mesh")
                                gtr_output_glb = gr.File(label="📦 GLB Mesh")
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**🎬 Mesh Rendering**")
                                    gtr_output_mesh_gif = gr.HTML(label="Mesh GIF")
                                with gr.Column():
                                    gr.Markdown("**🌟 NeRF Rendering**")
                                    gtr_output_nerf_gif = gr.HTML(label="NeRF GIF")

                with gr.Accordion("📝 Execution Logs", open=False):
                    gtr_logs = gr.Textbox(label="Logs", lines=10, show_label=False)

                # Refresh images functionality (filter by current project)
                gtr_refresh_images_btn.click(
                    fn=lambda proj: gr.Dropdown(choices=get_gtr_input_images(proj)),
                    inputs=[current_project],
                    outputs=[gtr_image_dropdown]
                )

                def get_camera_params_for_image(project_path, selected_image):
                    """Look up camera params from project_info.json for a selected image"""
                    if not project_path or not selected_image:
                        return 15.0, 90.0, "", ""  # defaults

                    import json
                    info_file = Path(project_path) / "project_info.json"
                    if not info_file.exists():
                        return 15.0, 90.0, "", ""

                    try:
                        with open(info_file, 'r') as f:
                            info = json.load(f)

                        # Extract filename from path
                        image_filename = Path(selected_image).name

                        # Search for matching inference entry
                        if "prompts" in info and "inference" in info["prompts"]:
                            for entry in info["prompts"]["inference"]:
                                if entry.get("generated_file") == image_filename:
                                    # Found matching entry - get camera params
                                    elevation = entry.get("elevation", 15.0)
                                    azimuth_start = entry.get("azimuth_start", 90.0)
                                    elevation_list_str = ""
                                    azimuth_list_str = ""

                                    # Get per-view lists as comma-separated strings
                                    if "elevation_list" in entry and entry["elevation_list"]:
                                        elevation = entry["elevation_list"][0]
                                        elevation_list_str = ",".join(str(e) for e in entry["elevation_list"])
                                    if "azimuth_list" in entry and entry["azimuth_list"]:
                                        azimuth_start = entry["azimuth_list"][0]
                                        azimuth_list_str = ",".join(str(a) for a in entry["azimuth_list"])

                                    print(f"📷 Loaded camera params for {image_filename}:")
                                    print(f"   elevation={elevation}, azimuth_start={azimuth_start}")
                                    print(f"   elevation_list={elevation_list_str}")
                                    print(f"   azimuth_list={azimuth_list_str}")
                                    return float(elevation), float(azimuth_start), elevation_list_str, azimuth_list_str

                        return 15.0, 90.0, "", ""  # defaults if not found
                    except Exception as e:
                        print(f"Error loading camera params: {e}")
                        return 15.0, 90.0, "", ""

                # Auto-fill camera params when image is selected
                gtr_image_dropdown.change(
                    fn=get_camera_params_for_image,
                    inputs=[current_project, gtr_image_dropdown],
                    outputs=[gtr_elevation, gtr_azimuth_start, gtr_elevation_list, gtr_azimuth_list]
                )

                # Run pipeline - only use selected image from project
                def run_gtr_full_pipeline(checkpoint, selected_image, project, cam_radius, fov, elevation, azimuth_start, padding, render_cam_distance, render_fov, render_elevation, render_num_frames, enable_mesh_export, export_glb, enable_mesh_gif, transparent_bg, elevation_list_str, azimuth_list_str):
                    """Wrapper to handle input selection. Loads params from project_info.json if available."""
                    # Load per-image camera parameters from project_info.json
                    img_elevation, img_azimuth_start, img_elevation_list_str, img_azimuth_list_str = get_camera_params_for_image(project, selected_image)

                    # Use loaded params if available, otherwise fall back to UI values
                    use_elevation = img_elevation if img_elevation_list_str else elevation
                    use_azimuth = img_azimuth_start if img_azimuth_list_str else azimuth_start
                    use_elev_list = img_elevation_list_str if img_elevation_list_str else (elevation_list_str if elevation_list_str else None)
                    use_azim_list = img_azimuth_list_str if img_azimuth_list_str else (azimuth_list_str if azimuth_list_str else None)

                    return full_gtr_pipeline(None, checkpoint, None, selected_image, project, cam_radius, fov, use_elevation, use_azimuth, padding, enable_mesh_export, export_glb, enable_mesh_gif, render_cam_distance, render_fov, render_elevation, render_num_frames, use_elev_list, use_azim_list, transparent_bg)

                def run_gtr_all_images(checkpoint, project, cam_radius, fov, elevation, azimuth_start, padding, render_cam_distance, render_fov, render_elevation, render_num_frames, enable_mesh_export, export_glb, enable_mesh_gif, transparent_bg, progress=gr.Progress()):
                    """Run GTR pipeline for all generated images in the project. Loads per-image camera parameters from project_info.json."""
                    if not project:
                        return None, None, None, None, "❌ Please select a project first"

                    # Get all images from 03_multiview_images
                    images_dir = Path(project) / "03_multiview_images"
                    if not images_dir.exists():
                        return None, None, None, None, f"❌ No multiview images folder found: {images_dir}"

                    # Find all generated images
                    all_images = []
                    for ext in ['*.jpg', '*.jpeg', '*.png']:
                        all_images.extend(images_dir.glob(ext))
                    all_images = sorted(all_images)

                    if not all_images:
                        return None, None, None, None, "❌ No images found in 03_multiview_images"

                    logs = []
                    logs.append(f"Found {len(all_images)} images to process:\n")
                    for img in all_images:
                        logs.append(f"  - {img.name}")
                    logs.append("\n" + "="*50 + "\n")

                    last_mesh = None
                    last_glb = None
                    last_mesh_gif = None
                    last_nerf_gif = None
                    success_count = 0
                    fail_count = 0

                    for i, img_path in enumerate(all_images):
                        progress((i + 1) / len(all_images), desc=f"Processing {img_path.name} ({i+1}/{len(all_images)})")

                        logs.append(f"\n[{i+1}/{len(all_images)}] Processing: {img_path.name}")
                        logs.append("-" * 40)

                        # Load per-image camera parameters from project_info.json
                        img_elevation, img_azimuth_start, img_elevation_list_str, img_azimuth_list_str = get_camera_params_for_image(project, str(img_path))

                        # Use loaded params if available, otherwise fall back to UI values
                        use_elevation = img_elevation if img_elevation_list_str else elevation
                        use_azimuth = img_azimuth_start if img_azimuth_list_str else azimuth_start
                        use_elev_list = img_elevation_list_str if img_elevation_list_str else None
                        use_azim_list = img_azimuth_list_str if img_azimuth_list_str else None

                        if img_elevation_list_str or img_azimuth_list_str:
                            logs.append(f"  📷 Using saved params: elev=[{img_elevation_list_str}], azim=[{img_azimuth_list_str}]")
                        else:
                            logs.append(f"  📷 Using UI params: elev={elevation}, azim={azimuth_start}")

                        try:
                            mesh_file, glb_file, mesh_gif, nerf_gif, log = full_gtr_pipeline(
                                None, checkpoint, None, str(img_path), project, cam_radius, fov, use_elevation, use_azimuth, padding, enable_mesh_export, export_glb, enable_mesh_gif, render_cam_distance, render_fov, render_elevation, render_num_frames, use_elev_list, use_azim_list, transparent_bg
                            )

                            if mesh_file or nerf_gif:
                                success_count += 1
                                last_mesh = mesh_file
                                last_glb = glb_file
                                last_mesh_gif = mesh_gif
                                last_nerf_gif = nerf_gif
                                logs.append(f"✅ Success: {Path(mesh_file).parent.name}/mesh.obj" if mesh_file else "✅ Success: NeRF only")
                            else:
                                fail_count += 1
                                logs.append(f"❌ Failed: {log[:200] if log else 'Unknown error'}")

                        except Exception as e:
                            fail_count += 1
                            logs.append(f"❌ Error: {str(e)[:200]}")

                    # Summary
                    logs.append("\n" + "="*50)
                    logs.append(f"\n📊 SUMMARY: {success_count} succeeded, {fail_count} failed out of {len(all_images)} total")

                    return last_mesh, last_glb, last_mesh_gif, last_nerf_gif, "\n".join(logs)

                gtr_run_btn.click(
                    run_gtr_full_pipeline,
                    inputs=[gtr_checkpoint, gtr_image_dropdown, current_project, gtr_cam_radius, gtr_fov, gtr_elevation, gtr_azimuth_start, gtr_padding, gtr_render_cam_distance, gtr_render_fov, gtr_render_elevation, gtr_render_num_frames, gtr_enable_mesh_export, gtr_export_glb, gtr_enable_mesh_gif, gtr_transparent_bg, gtr_elevation_list, gtr_azimuth_list],
                    outputs=[gtr_output_mesh, gtr_output_glb, gtr_output_mesh_gif, gtr_output_nerf_gif, gtr_logs]
                )

                gtr_run_all_btn.click(
                    run_gtr_all_images,
                    inputs=[gtr_checkpoint, current_project, gtr_cam_radius, gtr_fov, gtr_elevation, gtr_azimuth_start, gtr_padding, gtr_render_cam_distance, gtr_render_fov, gtr_render_elevation, gtr_render_num_frames, gtr_enable_mesh_export, gtr_export_glb, gtr_enable_mesh_gif, gtr_transparent_bg],
                    outputs=[gtr_output_mesh, gtr_output_glb, gtr_output_mesh_gif, gtr_output_nerf_gif, gtr_logs]
                )

            with gr.Tab("🔧 Step-by-Step"):
                gr.Markdown("""
                ### 🎯 Step 1: Prepare Multi-view
                Convert your input into the GTR-compatible format.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Add image selector for step-by-step (filtered by current project)
                        with gr.Group():
                            gr.Markdown("**🎨 Multi-view Images**")
                            with gr.Row():
                                gtr_prep_image_dropdown = gr.Dropdown(
                                    choices=[],  # Will be populated when project is selected
                                    label="Select Image",
                                    value=None,
                                    info="Generated multiview images from selected project",
                                    container=False
                                )
                                gtr_prep_refresh_images_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)


                        gtr_prep_btn = gr.Button("▶️ Prepare Multi-view", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Preparation Results")
                        with gr.Group():
                            gtr_prep_output_dir = gr.Textbox(
                                label="✅ Prepared Directory",
                                interactive=False,
                                info="Use this path for inference"
                            )

                        with gr.Accordion("📝 Preparation Logs", open=False):
                            gtr_prep_logs = gr.Textbox(label="Logs", lines=5, show_label=False)

                # Refresh images for step-by-step (filter by current project)
                gtr_prep_refresh_images_btn.click(
                    fn=lambda proj: gr.Dropdown(choices=get_gtr_input_images(proj)),
                    inputs=[current_project],
                    outputs=[gtr_prep_image_dropdown]
                )

                # Prepare with input selection - only use selected image from project
                # Also pass camera parameters from Configuration section
                gtr_prep_btn.click(
                    prepare_multiview_gtr,
                    inputs=[gr.State(None), gr.State(None), gtr_prep_image_dropdown, current_project, gtr_cam_radius, gtr_fov, gtr_elevation, gtr_azimuth_start, gtr_padding],
                    outputs=[gtr_prep_output_dir, gr.State(), gtr_prep_logs]  # Add dummy output for image_name
                )

                gr.Markdown("""
                ---
                ### 🚀 Step 2: Run Inference
                Generate 3D mesh from prepared multi-view data.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("**📂 Input Directory**")
                            gtr_inf_prepared_dir = gr.Textbox(
                                label="Prepared Directory",
                                placeholder="Paste path from Step 1 or select below",
                                info="From Step 1 preparation"
                            )

                            # Add prepared directory selector
                            with gr.Row():
                                gtr_prepared_dir_dropdown = gr.Dropdown(
                                    choices=[],
                                    label="Select from Project",
                                    value=None,
                                    info="Prepared directories from Step 1",
                                    container=False
                                )
                                gtr_prepared_dir_refresh_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

                        with gr.Group():
                            gr.Markdown("**⚙️ Configuration**")
                            gtr_inf_checkpoint = gr.Textbox(
                                value=GTR_CKPT_PATH,
                                label="Checkpoint Path",
                                info="Path to GTR model checkpoint"
                            )

                        gtr_inf_btn = gr.Button("🎯 Run Inference", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Inference Results")
                        with gr.Group():
                            gtr_inf_mesh = gr.File(label="💎 3D Mesh (.obj)")
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("**🎬 Mesh Rendering**")
                                    gtr_inf_mesh_gif = gr.HTML(label="Mesh GIF")
                                with gr.Column():
                                    gr.Markdown("**🌟 NeRF Rendering**")
                                    gtr_inf_nerf_gif = gr.HTML(label="NeRF GIF")

                with gr.Accordion("📝 Inference Logs", open=False):
                    gtr_inf_logs = gr.Textbox(label="Logs", lines=5, show_label=False)

                # Event handlers for Step 2
                def load_gtr_prepared_dir(dir_path):
                    """Load selected prepared directory path into textbox"""
                    return dir_path if dir_path else ""

                gtr_prepared_dir_dropdown.change(
                    load_gtr_prepared_dir,
                    inputs=[gtr_prepared_dir_dropdown],
                    outputs=[gtr_inf_prepared_dir]
                )

                # Refresh prepared directories dropdown
                gtr_prepared_dir_refresh_btn.click(
                    fn=lambda proj: gr.Dropdown(choices=get_gtr_prepared_directories(proj)),
                    inputs=[current_project],
                    outputs=[gtr_prepared_dir_dropdown]
                )

                # Auto-populate prepared directory from Step 1 output
                gtr_prep_output_dir.change(
                    lambda path: path,
                    inputs=[gtr_prep_output_dir],
                    outputs=[gtr_inf_prepared_dir]
                )

                gtr_inf_btn.click(
                    run_gtr_inference,
                    inputs=[gtr_inf_prepared_dir, gtr_inf_checkpoint, current_project],
                    outputs=[gtr_inf_mesh, gtr_inf_mesh_gif, gtr_inf_nerf_gif, gtr_inf_logs]
                )

            with gr.Accordion("ℹ️ Usage Guide & Supported Formats", open=False):
                gr.Markdown("""
                ## 📖 Usage

                ### ⚡ Full Pipeline Tab
                - **Best for**: Quick end-to-end processing
                - **How to use**: Select/upload input → Click "Run Full Pipeline"
                - **Output**: 3D mesh (.obj) + rendering GIFs

                ### 🔧 Step-by-Step Tab
                - **Best for**: Fine control over each step
                - **How to use**:
                  1. Step 1: Prepare multi-view data
                  2. Step 2: Run inference with prepared data

                ## 📝 Supported Input Formats (Intelligent Auto-Detection)

                | Format | Layout | Views | Detection | Camera Setup |
                |--------|--------|-------|-----------|--------------|
                | **Horizontal Strip** | 1×N | 4-6 | Aspect ≥3.5 | Evenly distributed around object |
                | **Zero123++ Grid** | 3×2 | 6 | Aspect ~1.5 | Multi-elevation (70°-100°) |
                | **Square Grids** | 2×2, 3×3 | 4, 9 | Aspect ~1.0 | Horizontal ring |
                | **Custom Grids** | Any | Auto | Smart detect | Auto-generated |

                ### 🔍 Intelligent Grid Detection
                The system automatically:
                1. **Analyzes image dimensions** to detect grid size (rows × columns)
                2. **Splits into individual views** and saves them separately
                3. **Generates appropriate camera parameters** for each view
                4. **Supports:** 1×4, 1×5, 1×6, 2×2, 2×3, 3×2, 3×3, 4×2, and more!

                ### 📂 Output Structure
                For each multiview image processed, you get:
                ```
                projects/{project}/04_gtr_3d/{image_name}/
                ├── prepared_mv/          # GTR input data
                │   ├── rgb_000.png       # Processed views (512×512)
                │   ├── cam_000.txt       # Camera parameters
                │   └── ...
                ├── separated_views/      # 🆕 Individual view files
                │   ├── view_000_original.png    # Original split views
                │   ├── view_000_processed.png   # Background removed
                │   └── ...
                ├── mesh.obj             # Vertex colors
                ├── mesh_textured.obj    # UV mapped mesh
                ├── mesh_texture.png     # 2048×2048 baked texture
                ├── mesh.gif             # Mesh preview
                └── nerf.gif             # NeRF preview
                ```

                ## 💡 Tips
                - Upload any grid layout - it will be **automatically detected**!
                - Individual views are saved in `separated_views/` folder
                - Both original and processed (bg-removed) views are saved
                - Check **logs** to see detected grid size and view count
                """)

        # Auto-refresh GTR image dropdowns when project changes
        def refresh_gtr_dropdowns(project_path):
            """Refresh both GTR dropdowns when project changes"""
            if not project_path:
                return gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None)
            choices = get_gtr_input_images(project_path)
            # Set first choice as default value if available
            default_value = choices[0] if choices else None

            # Get prepared directories
            prepared_choices = get_gtr_prepared_directories(project_path)
            prepared_default = prepared_choices[0] if prepared_choices else None

            return (
                gr.Dropdown(choices=choices, value=default_value),
                gr.Dropdown(choices=choices, value=default_value),
                gr.Dropdown(choices=prepared_choices, value=prepared_default)
            )

        # Update dropdowns when project changes
        project_dropdown.change(
            fn=refresh_gtr_dropdowns,
            inputs=[project_dropdown],
            outputs=[gtr_image_dropdown, gtr_prep_image_dropdown, gtr_prepared_dir_dropdown]
        )

        with gr.Tab("🔮 3D Mesh Viewer"):
            gr.Markdown("""
            ## Interactive 3D Mesh Visualization

            View and interact with generated 3D meshes using Google model-viewer.
            **Supports**: .obj, .ply, .glb files from GTR output
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Select Mesh from Current Project")

                    # Mesh file selector (uses current_project from main dropdown)
                    mesh_file_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Mesh File",
                        value=None,
                        interactive=True
                    )

                    with gr.Row():
                        load_mesh_btn = gr.Button("Load Mesh", variant="primary")
                        refresh_mesh_btn = gr.Button("🔄 Refresh", scale=0, min_width=50)

                    mesh_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        max_lines=2
                    )

                with gr.Column(scale=3):
                    mesh_viewer_display = gr.HTML(
                        label="3D Mesh Viewer",
                        value="<div style='text-align:center; padding:100px; color:#888; background:#f5f5f5; border-radius:8px;'>Select a mesh file to visualize</div>"
                    )

            gr.Markdown("""
            ### 🎮 Interaction Controls:
            - **Rotate**: Click and drag
            - **Zoom**: Scroll wheel or pinch
            - **Pan**: Right-click and drag (or two-finger drag)
            - **Auto-rotate**: Enabled by default

            ### 📍 Where to Find Meshes:
            - Generated meshes are in: `projects/{project}/04_gtr_3d/{image_name}/mesh.obj`
            - After running GTR inference, use the project dropdown to browse available meshes
            """)

            # Mesh visualization functions
            def update_mesh_list(project_path):
                """Update mesh file dropdown based on selected project."""
                if not project_path:
                    return gr.Dropdown(choices=[], value=None)

                meshes = mesh_visualizer.get_available_meshes(project_path)
                mesh_choices = [str(Path(m).relative_to(project_path)) for m in meshes]
                return gr.Dropdown(choices=mesh_choices, value=mesh_choices[0] if mesh_choices else None)

            def load_mesh_visualization(project_path, mesh_file, uploaded_file):
                """Load and visualize the selected mesh."""
                if uploaded_file:
                    # Use uploaded file
                    mesh_path = uploaded_file
                elif project_path and mesh_file:
                    # Use project mesh
                    mesh_path = str(Path(project_path) / mesh_file)
                else:
                    return "<div style='color:orange; text-align:center; padding:50px;'>⚠️ No mesh selected</div>", "No mesh selected"

                if not os.path.exists(mesh_path):
                    return "<div style='color:red; text-align:center; padding:50px;'>⚠️ Mesh file not found</div>", "File not found"

                return mesh_visualizer.create_interactive_visualization(mesh_path)

            def refresh_mesh_projects():
                """Refresh project list."""
                if project_manager:
                    projects = project_manager.get_all_projects()
                    return gr.Dropdown(choices=projects)
                return gr.Dropdown()

            # Event handlers
            # Update mesh list when main project dropdown changes
            project_dropdown.change(
                fn=update_mesh_list,
                inputs=[project_dropdown],
                outputs=[mesh_file_dropdown]
            )

            load_mesh_btn.click(
                fn=load_mesh_visualization,
                inputs=[current_project, mesh_file_dropdown, gr.State(None)],
                outputs=[mesh_viewer_display, mesh_status]
            )

            # Refresh mesh list for current project
            refresh_mesh_btn.click(
                fn=update_mesh_list,
                inputs=[current_project],
                outputs=[mesh_file_dropdown]
            )

        # ===== EVALUATION TAB =====
        with gr.Tab("📊 Evaluation"):
            gr.Markdown("## Evaluate 3D Models\nSelect a mesh, render views, then run metrics.")

            with gr.Row():
                # LEFT PANEL: Mesh Selection + Render Controls
                with gr.Column(scale=1):
                    gr.Markdown("### Mesh Selection")

                    eval_mesh_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select Mesh to Evaluate",
                        value=None,
                        interactive=True
                    )

                    eval_refresh_btn = gr.Button("🔄 Refresh Mesh List", size="sm")

                    eval_render_status = gr.Textbox(
                        label="Render Status",
                        value="No renders available",
                        interactive=False,
                        lines=2
                    )

                    with gr.Row():
                        render_views_btn = gr.Button("🎬 Render Selected", variant="secondary")
                        render_all_btn = gr.Button("🎬 Render All Meshes", variant="secondary")

                    with gr.Accordion("Render Settings", open=False):
                        eval_num_views = gr.Slider(
                            minimum=12,
                            maximum=120,
                            value=70,
                            step=1,
                            label="Number of Views"
                        )

                        with gr.Row():
                            eval_normalize = gr.Checkbox(
                                value=True, label="Normalize Mesh",
                                info="Center and scale to standard size"
                            )
                            eval_object_scale = gr.Slider(
                                minimum=0.5, maximum=3.0, value=1.0, step=0.1,
                                label="Object Scale",
                                info="1.0 = default, >1 = bigger"
                            )

                    eval_rendered_gallery = gr.Gallery(
                        label="Rendered Views",
                        columns=3,
                        rows=2,
                        height=250
                    )

                # RIGHT PANEL: Evaluation Sub-tabs
                with gr.Column(scale=2):
                    with gr.Tabs():
                        # --- CLIP Score Sub-Tab ---
                        with gr.Tab("CLIP Score"):
                            gr.Markdown("_Compute CLIP similarity between rendered views and a text prompt._")

                            clip_score_text = gr.Textbox(
                                label="Text Prompt",
                                placeholder="e.g., a human head smiling",
                                value=""
                            )

                            run_clip_score_btn = gr.Button("▶ Run CLIP Score", variant="primary")

                            clip_score_results = gr.Markdown(
                                value="_Run CLIP Score to see results_"
                            )

                        # --- CLIP Metrics Sub-Tab ---
                        with gr.Tab("CLIP Metrics"):
                            gr.Markdown("_Directional CLIP metrics comparing original vs edited models._")

                            clip_text_input = gr.Textbox(
                                label="Original Text Prompt",
                                placeholder="e.g., a human head",
                                value=""
                            )

                            clip_text_edit = gr.Textbox(
                                label="Edited Text Prompt",
                                placeholder="e.g., a human head smiling",
                                value=""
                            )

                            with gr.Accordion("Advanced", open=False):
                                clip_edited_word = gr.Textbox(
                                    label="Edited Word (for CLIP_diff-edit)",
                                    placeholder="e.g., smiling",
                                    value=""
                                )

                                clip_generic_text = gr.Textbox(
                                    label="Generic Text (for CLIP_diff-noedit)",
                                    placeholder="e.g., a head",
                                    value=""
                                )

                                clip_input_dir = gr.Textbox(
                                    label="Input Renders Directory",
                                    placeholder="Path to original model renders",
                                    value=""
                                )

                                clip_edit_dir = gr.Textbox(
                                    label="Edited Renders Directory (optional)",
                                    placeholder="Leave empty to use same mesh renders",
                                    value=""
                                )

                            run_clip_metrics_btn = gr.Button("▶ Run CLIP Metrics", variant="primary")

                            clip_metrics_results = gr.Markdown(
                                value="_Run CLIP Metrics to see results_"
                            )

                        # --- CLIP-IQA Sub-Tab ---
                        with gr.Tab("CLIP-IQA"):
                            gr.Markdown("_No-reference image quality assessment (no text prompts needed)._")

                            clip_iqa_prompts = gr.CheckboxGroup(
                                choices=["quality", "sharpness", "noisiness", "aesthetic",
                                        "brightness", "contrast", "colorfulness",
                                        "render_quality", "geometry", "texture", "lighting"],
                                value=["quality", "sharpness", "aesthetic", "render_quality"],
                                label="Quality Dimensions to Evaluate"
                            )

                            run_clip_iqa_btn = gr.Button("▶ Run CLIP-IQA", variant="primary")

                            clip_iqa_results_display = gr.Markdown(
                                value="_Run CLIP-IQA to see quality scores_"
                            )

                        # --- GPT-4V Sub-Tab ---
                        with gr.Tab("GPT-4V"):
                            gr.Markdown("_AI-powered quality assessment (text alignment, visual quality, 3D consistency)._")

                            gpt_prompt = gr.Textbox(
                                label="Generation Prompt",
                                placeholder="Describe what the 3D model should be",
                                value=""
                            )

                            gr.Markdown("_Requires OPENAI_API_KEY environment variable_")

                            run_gpt_eval_btn = gr.Button("▶ Run GPT-4V Eval", variant="primary")

                            gpt_results_display = gr.Markdown(
                                value="_Run GPT-4V evaluation to see results_"
                            )

                        # --- Run All Sub-Tab ---
                        with gr.Tab("Run All"):
                            gr.Markdown("_Run multiple metrics at once on the selected mesh._")

                            eval_mode_checks = gr.CheckboxGroup(
                                choices=["CLIP Score", "CLIP Metrics", "CLIP-IQA", "GPT-4V Eval"],
                                value=["CLIP-IQA"],
                                label="Metrics to Run"
                            )

                            run_all_eval_btn = gr.Button("▶ Run All Evaluations", variant="primary")

                            eval_progress = gr.Textbox(
                                label="Progress",
                                value="Ready to evaluate",
                                interactive=False,
                                lines=3
                            )

                            # Combined results display
                            with gr.Tabs():
                                with gr.Tab("CLIP Results"):
                                    clip_results_display = gr.Markdown(
                                        value="_Run evaluation to see CLIP results_"
                                    )

                                with gr.Tab("CLIP-IQA Results"):
                                    clip_iqa_all_results = gr.Markdown(
                                        value="_Run evaluation to see CLIP-IQA results_"
                                    )

                                with gr.Tab("GPT-4V Results"):
                                    gpt_all_results = gr.Markdown(
                                        value="_Run evaluation to see GPT-4V results_"
                                    )

                        # --- Report Sub-Tab ---
                        with gr.Tab("Report"):
                            avg_scores_display = gr.Markdown(
                                value="_Run evaluations first to see average scores_"
                            )

                            with gr.Row():
                                generate_report_btn = gr.Button("📄 Generate PDF Report", variant="secondary")
                                refresh_avg_btn = gr.Button("🔄 Refresh Averages", variant="secondary", scale=0)

                            report_download = gr.File(
                                label="Download Report",
                                visible=True
                            )

            # Benchmark Tools (collapsed)
            with gr.Accordion("Benchmark Tools", open=False):
                gr.Markdown("Renders views and computes metrics for all methods. "
                            "Cases are auto-discovered from `evaluation/benchmark/cases/` folders. "
                            "Each case folder has `meta.json`, `original.glb/.obj`, and `{method}.glb/.obj`.")

                _cases_dir = Path("/home/ai/gr/DreamEdit3D/evaluation/benchmark/cases")
                _initial_cases = sorted([
                    d.name for d in _cases_dir.iterdir()
                    if d.is_dir() and (d / "meta.json").exists()
                ]) if _cases_dir.exists() else []
                benchmark_case_checks = gr.CheckboxGroup(
                    choices=_initial_cases,
                    value=_initial_cases,
                    label="Cases to Evaluate",
                    info="Case folders from evaluation/benchmark/cases/ containing meta.json"
                )
                refresh_cases_btn = gr.Button("Refresh Cases", size="sm")

                benchmark_method_checks = gr.CheckboxGroup(
                    choices=["mvedit", "ours", "vox-e", "preditor3d"],
                    value=["mvedit"],
                    label="Methods to Evaluate",
                    info="Methods with GLB/OBJ meshes in case folders"
                )
                refresh_methods_btn = gr.Button("Refresh Methods", size="sm")

                benchmark_metric_checks = gr.CheckboxGroup(
                    choices=["CLIP Score", "CLIP Directional", "CLIP-IQA"],
                    value=["CLIP Score", "CLIP-IQA"],
                    label="Metrics",
                    info="CLIP Score/Directional use per-case text prompts from meta.json"
                )

                benchmark_iqa_dims = gr.CheckboxGroup(
                    choices=["quality", "render_quality", "geometry", "texture",
                            "aesthetic", "sharpness", "noisiness",
                            "brightness", "contrast", "colorfulness"],
                    value=["quality", "render_quality", "geometry", "texture"],
                    label="IQA Quality Dimensions"
                )

                with gr.Row():
                    bench_full_num_views = gr.Number(
                        value=70, label="Views", info="Paper uses 70"
                    )
                    bench_full_normalize = gr.Checkbox(
                        value=True, label="Normalize"
                    )
                    bench_full_object_scale = gr.Slider(
                        minimum=0.5, maximum=3.0, value=1.0, step=0.1,
                        label="Scale"
                    )
                    bench_full_start_azimuth = gr.Slider(
                        minimum=0, maximum=360, value=0, step=5,
                        label="Start Azimuth"
                    )
                    bench_full_force_rerender = gr.Checkbox(
                        value=False, label="Force Re-render",
                        info="Delete existing renders and re-render from scratch"
                    )
                    bench_full_force_recompute = gr.Checkbox(
                        value=False, label="Force Recompute",
                        info="Recompute metrics from existing renders (ignore cached results)"
                    )

                run_full_benchmark_btn = gr.Button("Run Full Benchmark", variant="primary")

                benchmark_progress_log = gr.Textbox(
                    label="Progress",
                    value="Ready. Select methods and click Run.",
                    lines=8,
                    interactive=False
                )

                benchmark_results_md = gr.Markdown(
                    value="_Results will appear here after running the benchmark._"
                )

                # GIF Rendering (secondary, collapsed)
                with gr.Accordion("GIF Rendering", open=False):
                    gr.Markdown("Render GLB files to GIFs for benchmark comparison.")

                    benchmark_folder = gr.Dropdown(
                        choices=["mvedit", "ours", "vox-e", "preditor3d"],
                        value="mvedit",
                        label="Benchmark Folder",
                        info="Select benchmark to render"
                    )

                    with gr.Row():
                        bench_cam_distance = gr.Number(
                            value=3.5, label="Distance", info="Camera distance"
                        )
                        bench_cam_fov = gr.Number(
                            value=50.0, label="FOV", info="Field of view"
                        )
                        bench_cam_elevation = gr.Number(
                            value=20.0, label="Elevation", info="Camera elevation"
                        )

                    with gr.Row():
                        bench_num_frames = gr.Number(
                            value=50, label="Frames", info="Number of frames"
                        )
                        bench_start_azimuth_gif = gr.Number(
                            value=0, label="Start Azimuth", info="Starting angle"
                        )
                        bench_resolution = gr.Number(
                            value=512, label="Resolution", info="Image resolution"
                        )

                    bench_background = gr.Dropdown(
                        choices=["white", "black", "gray", "transparent"],
                        value="white",
                        label="Background Color"
                    )

                    with gr.Row():
                        render_benchmark_btn = gr.Button("🎬 Render All GLBs to GIFs", variant="primary")
                        refresh_benchmark_btn = gr.Button("🔄 Refresh", size="sm")

                    benchmark_status = gr.Textbox(
                        label="Status",
                        value="Ready",
                        lines=4
                    )

            # Evaluation helper functions
            def update_eval_mesh_list(project_path):
                """Update mesh dropdown for evaluation"""
                if not project_path:
                    return gr.Dropdown(choices=[], value=None), "No project selected"

                meshes = mesh_visualizer.get_available_meshes(project_path)
                mesh_choices = [str(Path(m).relative_to(project_path)) for m in meshes]

                status = f"Found {len(mesh_choices)} mesh(es)" if mesh_choices else "No meshes found"
                return gr.Dropdown(choices=mesh_choices, value=mesh_choices[0] if mesh_choices else None), status

            def load_saved_prompts(project_path):
                """Load saved prompts from project_info.json and auto-populate evaluation fields

                - Original Text (CLIP): from training prompts (e.g., "viking")
                - Edited Text (CLIP): from inference prompts with <asset0> replaced (e.g., "viking smile")
                - GPT Prompt: from inference prompts with <asset0> replaced
                """
                if not project_path or not project_manager:
                    return "", "", "", ""

                training_prompts, inference_prompts = project_manager.get_latest_prompts(project_path)

                # Get training prompt (original concept name)
                original_text = ""
                if training_prompts:
                    original_text = training_prompts[0] if isinstance(training_prompts, list) else training_prompts

                # Get inference prompt and replace <asset0> with training prompt
                edited_text = ""
                gpt_prompt = ""
                if inference_prompts:
                    inference_prompt = inference_prompts[0] if isinstance(inference_prompts, list) else inference_prompts
                    # Replace <asset0> with the training prompt
                    if original_text:
                        edited_text = inference_prompt.replace("<asset0>", original_text)
                        gpt_prompt = edited_text
                    else:
                        edited_text = inference_prompt
                        gpt_prompt = inference_prompt

                # If no inference prompt, use training prompt for both
                if not edited_text and original_text:
                    edited_text = original_text

                # clip_score_text gets the edited text (most useful for CLIP Score)
                return edited_text, original_text, edited_text, gpt_prompt

            def get_mesh_eval_folder(project_path, mesh_file):
                """Extract mesh folder name and create evaluation subfolder path"""
                # mesh_file looks like: 04_gtr_3d/generated_multiview_1769078295/mesh.obj
                # We want: generated_multiview_1769078295
                mesh_path = Path(mesh_file)
                mesh_folder_name = mesh_path.parent.name  # e.g., generated_multiview_1769078295
                eval_dir = Path(project_path) / "05_evaluation" / mesh_folder_name
                return eval_dir, mesh_folder_name

            def check_renders_status(project_path, mesh_file):
                """Check if renders exist for the selected mesh"""
                if not project_path or not mesh_file:
                    return "Select a mesh first", []

                # Get mesh-specific evaluation folder
                eval_dir, mesh_name = get_mesh_eval_folder(project_path, mesh_file)
                render_dir = eval_dir / "renders"

                if render_dir.exists():
                    rgb_files = sorted(render_dir.glob("rgb_*.png"))
                    if len(rgb_files) > 0:
                        gallery_images = [str(f) for f in rgb_files[:12]]
                        return f"Found {len(rgb_files)} renders for {mesh_name}:\n{render_dir}", gallery_images

                return f"No renders found for {mesh_name}. Click 'Render Views' to generate.", []

            def render_views_for_eval(project_path, mesh_file, num_views, do_normalize, object_scale,
                                       progress=gr.Progress()):
                """Render views for evaluation"""
                if not project_path or not mesh_file:
                    return "Please select a project and mesh first", [], ""

                mesh_path = str(Path(project_path) / mesh_file)
                if not os.path.exists(mesh_path):
                    return f"Mesh not found: {mesh_path}", [], ""

                # Create mesh-specific evaluation folder: 05_evaluation/{mesh_name}/renders/
                eval_dir, mesh_name = get_mesh_eval_folder(project_path, mesh_file)
                render_dir = eval_dir / "renders"

                progress(0.1, desc=f"Rendering {mesh_name}...")

                def progress_cb(msg):
                    progress(0.5, desc=msg)

                success, msg = eval_manager.render_mesh_for_eval(
                    mesh_path=mesh_path,
                    output_dir=str(render_dir),
                    num_views=int(num_views),
                    progress_callback=progress_cb,
                    do_normalize=do_normalize,
                    object_scale=float(object_scale)
                )

                progress(1.0, desc="Complete")

                if success:
                    # Load gallery images
                    rgb_files = sorted(render_dir.glob("rgb_*.png"))[:12]
                    gallery_images = [str(f) for f in rgb_files]
                    return f"✅ {msg}", gallery_images, str(render_dir)
                else:
                    return f"❌ {msg}", [], ""

            def run_evaluation(project_path, mesh_file, eval_mode,
                             clip_input, clip_edit, clip_word, clip_generic,
                             clip_input_dir, clip_edit_dir, clip_iqa_prompts, gpt_prompt,
                             progress=gr.Progress()):
                """Run the selected evaluation"""
                if not project_path or not mesh_file:
                    return "Select a mesh first", "_No results_", "_No results_", "_No results_"

                # Use mesh-specific evaluation folder: 05_evaluation/{mesh_name}/
                eval_dir, mesh_name = get_mesh_eval_folder(project_path, mesh_file)
                render_dir = eval_dir / "renders"

                # Check for renders
                if not render_dir.exists() or len(list(render_dir.glob("rgb_*.png"))) == 0:
                    return f"No renders found for {mesh_name}. Click 'Render Views' first.", "_No results_", "_No results_", "_No results_"

                clip_results_md = "_Not run_"
                clip_iqa_results_md = "_Not run_"
                gpt_results_md = "_Not run_"
                status_lines = []

                # Determine input directory for CLIP
                input_dir = clip_input_dir if clip_input_dir else str(render_dir)
                edit_dir = clip_edit_dir if clip_edit_dir else str(render_dir)

                # Run CLIP Score (standalone, only needs edited text)
                if eval_mode in ["CLIP Score", "All"]:
                    progress(0.1, desc="Running CLIP Score...")

                    text_prompt = clip_edit if clip_edit else clip_input
                    if not text_prompt:
                        status_lines.append("⚠️ CLIP Score: Need a text prompt (use Edited Text Prompt)")
                        clip_results_md = "_Please provide a text prompt in Edited Text Prompt field_"
                    else:
                        try:
                            success, msg = eval_manager.init_clip_evaluator()
                            if success:
                                renders_path = edit_dir if clip_edit_dir else str(render_dir)
                                images = eval_manager.clip_evaluator.load_images_from_dir(renders_path, pattern="rgb_*.png")
                                score_result = eval_manager.clip_evaluator.clip_score(images, text_prompt)

                                clip_results_md = f"## CLIP Score Results\n\n"
                                clip_results_md += f"**Text prompt**: {text_prompt}\n\n"
                                clip_results_md += f"- **Mean**: {score_result['CLIP_score_mean']:.2f}\n"
                                clip_results_md += f"- **Std**: {score_result['CLIP_score_std']:.2f}\n"
                                clip_results_md += f"- **Min**: {score_result['CLIP_score_min']:.2f}\n"
                                clip_results_md += f"- **Max**: {score_result['CLIP_score_max']:.2f}\n"
                                clip_results_md += f"\n_Scored {len(images)} images_"
                                status_lines.append(f"✅ CLIP Score: {score_result['CLIP_score_mean']:.2f}")

                                # Save results
                                import json
                                results_file = eval_dir / "clip_score_results.json"
                                with open(results_file, 'w') as f:
                                    json.dump({"text_prompt": text_prompt, **score_result}, f, indent=2)
                                status_lines.append(f"   Saved to: {results_file}")
                            else:
                                clip_results_md = f"_Error: {msg}_"
                                status_lines.append(f"❌ CLIP Score: {msg}")
                        except Exception as e:
                            clip_results_md = f"_Error: {e}_"
                            status_lines.append(f"❌ CLIP Score: {e}")

                # Run CLIP evaluation
                if eval_mode in ["CLIP Metrics", "All"]:
                    progress(0.15, desc="Running CLIP evaluation...")

                    if not clip_input or not clip_edit:
                        status_lines.append("⚠️ CLIP: Need both text prompts")
                        clip_results_md = "_Please provide original and edited text prompts_"
                    else:
                        results, msg = eval_manager.run_clip_evaluation(
                            input_renders_dir=input_dir,
                            edit_renders_dir=edit_dir,
                            text_input=clip_input,
                            text_edit=clip_edit,
                            text_edited_word=clip_word if clip_word else None,
                            text_generic=clip_generic if clip_generic else None
                        )

                        if results:
                            clip_results_md = eval_manager.format_clip_results(results)
                            status_lines.append(f"✅ CLIP: {msg}")

                            # Save results to 05_evaluation folder
                            results_file = eval_dir / "clip_results.json"
                            import json
                            with open(results_file, 'w') as f:
                                json.dump(results, f, indent=2)
                            status_lines.append(f"   Saved to: {results_file}")
                        else:
                            clip_results_md = f"_Error: {msg}_"
                            status_lines.append(f"❌ CLIP: {msg}")

                # Run CLIP-IQA evaluation
                if eval_mode in ["CLIP-IQA", "All"]:
                    progress(0.4, desc="Running CLIP-IQA evaluation...")

                    prompt_names = clip_iqa_prompts if clip_iqa_prompts else ["quality", "sharpness", "aesthetic", "render_quality"]
                    results, msg = eval_manager.run_clip_iqa_evaluation(
                        renders_dir=str(render_dir),
                        prompt_names=prompt_names
                    )

                    if results:
                        clip_iqa_results_md = eval_manager.format_clip_iqa_results(results)
                        status_lines.append(f"✅ CLIP-IQA: {msg}")

                        # Save results
                        results_file = eval_dir / "clip_iqa_results.json"
                        import json
                        with open(results_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        status_lines.append(f"   Saved to: {results_file}")
                    else:
                        clip_iqa_results_md = f"_Error: {msg}_"
                        status_lines.append(f"❌ CLIP-IQA: {msg}")

                # Run GPT-4V evaluation
                if eval_mode in ["GPT-4V Eval", "All"]:
                    progress(0.7, desc="Running GPT-4V evaluation...")

                    if not gpt_prompt:
                        status_lines.append("⚠️ GPT-4V: Need generation prompt")
                        gpt_results_md = "_Please provide the generation prompt_"
                    else:
                        output_file = str(eval_dir / "gpteval_results.json")
                        results, msg = eval_manager.run_gpteval(
                            renders_dir=str(render_dir),
                            prompt_text=gpt_prompt,
                            output_file=output_file
                        )

                        if results:
                            gpt_results_md = eval_manager.format_gpteval_results(results)
                            status_lines.append(f"✅ GPT-4V: {msg}")
                            status_lines.append(f"   Saved to: {output_file}")
                        else:
                            gpt_results_md = f"_Error: {msg}_"
                            status_lines.append(f"❌ GPT-4V: {msg}")

                progress(1.0, desc="Complete")

                # Save eval_summary.json linking everything together
                try:
                    from datetime import datetime
                    import json

                    summary = {
                        "mesh_name": mesh_name,
                        "mesh_file": mesh_file,
                        "mesh_path": str(Path(project_path) / mesh_file),
                        "evaluation_timestamp": datetime.now().isoformat(),
                        "prompts": {
                            "clip_input": clip_input,
                            "clip_edit": clip_edit,
                            "gpt_prompt": gpt_prompt
                        },
                        "results_files": {
                            "renders_dir": str(render_dir),
                            "clip_results": str(eval_dir / "clip_results.json") if (eval_dir / "clip_results.json").exists() else None,
                            "clip_iqa_results": str(eval_dir / "clip_iqa_results.json") if (eval_dir / "clip_iqa_results.json").exists() else None,
                            "gpteval_results": str(eval_dir / "gpteval_results.json") if (eval_dir / "gpteval_results.json").exists() else None
                        }
                    }

                    summary_file = eval_dir / "eval_summary.json"
                    with open(summary_file, 'w') as f:
                        json.dump(summary, f, indent=2)
                    status_lines.append(f"📋 Summary: {summary_file}")
                except Exception as e:
                    status_lines.append(f"⚠️ Could not save summary: {e}")

                status_text = "\n".join(status_lines) if status_lines else "Evaluation complete"
                return status_text, clip_results_md, clip_iqa_results_md, gpt_results_md

            # Benchmark GIF rendering function
            def render_benchmark_glbs_to_gifs(benchmark_name, cam_distance, cam_fov, cam_elevation,
                                              num_frames, start_azimuth, resolution, bg_color,
                                              progress=gr.Progress()):
                """Render all GLB files in a benchmark folder to GIFs using pyrender for proper PBR texture support."""
                import os as _os
                _os.environ['PYOPENGL_PLATFORM'] = 'egl'
                import imageio.v2 as imageio
                import pyrender

                benchmark_base = Path("/home/ai/gr/DreamEdit3D/evaluation/benchmark")
                glbs_dir = benchmark_base / benchmark_name / "glbs"
                gifs_dir = benchmark_base / benchmark_name / "gifs"

                if not glbs_dir.exists():
                    return f"GLBs folder not found: {glbs_dir}"

                gifs_dir.mkdir(exist_ok=True, parents=True)

                # Find all GLB files
                glb_files = sorted(list(glbs_dir.glob("*.glb")))
                if not glb_files:
                    return f"No GLB files found in {glbs_dir}"

                # Parse parameters
                is_transparent = (bg_color == "transparent")
                bg_colors_map = {
                    "white": [1.0, 1.0, 1.0, 1.0],
                    "black": [0.0, 0.0, 0.0, 1.0],
                    "gray": [0.5, 0.5, 0.5, 1.0],
                    "transparent": [0.0, 0.0, 0.0, 0.0],
                }
                bg_rgba = bg_colors_map.get(bg_color, [1.0, 1.0, 1.0, 1.0])

                resolution = int(resolution)
                num_frames = int(num_frames)
                start_azimuth = float(start_azimuth)
                cam_distance = float(cam_distance)
                cam_fov = float(cam_fov)
                cam_elevation = float(cam_elevation)

                # Pre-compute camera poses
                azimuths_deg = np.linspace(start_azimuth, start_azimuth + 360, num=num_frames, endpoint=False)
                elev_rad = np.radians(cam_elevation)

                def make_camera_pose(az_deg):
                    az_rad = np.radians(az_deg)
                    cam_pos = np.array([
                        cam_distance * np.cos(elev_rad) * np.cos(az_rad),
                        cam_distance * np.sin(elev_rad),
                        cam_distance * np.cos(elev_rad) * np.sin(az_rad),
                    ])
                    forward = -cam_pos / np.linalg.norm(cam_pos)
                    world_up = np.array([0.0, 1.0, 0.0])
                    right = np.cross(forward, world_up)
                    norm = np.linalg.norm(right)
                    if norm < 1e-6:
                        right = np.array([1.0, 0.0, 0.0])
                    else:
                        right = right / norm
                    up = np.cross(right, forward)
                    up = up / np.linalg.norm(up)
                    pose = np.eye(4)
                    pose[:3, 0] = right
                    pose[:3, 1] = up
                    pose[:3, 2] = -forward
                    pose[:3, 3] = cam_pos
                    return pose

                logs = [f"Rendering {len(glb_files)} GLB files from {benchmark_name} (pyrender)"]
                logs.append(f"Camera: distance={cam_distance}, FOV={cam_fov}, elevation={cam_elevation}")
                logs.append(f"Frames: {num_frames}, Start: {start_azimuth}, Background: {bg_color}")
                logs.append("")

                renderer = pyrender.OffscreenRenderer(resolution, resolution)
                render_flags = pyrender.constants.RenderFlags.RGBA if is_transparent else pyrender.constants.RenderFlags.NONE

                success_count = 0
                fail_count = 0

                for i, glb_path in enumerate(glb_files):
                    progress((i + 1) / len(glb_files), desc=f"Rendering {glb_path.name}")

                    try:
                        # Load as trimesh scene (preserves PBR materials and multi-geometry)
                        scene_trimesh = trimesh.load(str(glb_path))
                        if isinstance(scene_trimesh, trimesh.Trimesh):
                            scene_trimesh = trimesh.Scene(scene_trimesh)

                        # Normalize: center and scale to fit [-1, 1]
                        bounds = scene_trimesh.bounds  # (2, 3)
                        center = (bounds[0] + bounds[1]) / 2.0
                        extents = bounds[1] - bounds[0]
                        scale_factor = 2.0 / np.max(extents)
                        normalize_mtx = np.eye(4)
                        normalize_mtx[:3, :3] *= scale_factor
                        normalize_mtx[:3, 3] = -center * scale_factor
                        scene_trimesh.apply_transform(normalize_mtx)

                        # Convert to pyrender scene
                        scene = pyrender.Scene.from_trimesh_scene(
                            scene_trimesh,
                            bg_color=bg_rgba,
                            ambient_light=[0.3, 0.3, 0.3],
                        )

                        # Add camera
                        camera = pyrender.PerspectiveCamera(yfov=np.radians(cam_fov))
                        cam_node = scene.add(camera)

                        # Add directional light
                        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
                        light_node = scene.add(light)

                        # Render all frames
                        frames = []
                        for j in range(num_frames):
                            pose = make_camera_pose(azimuths_deg[j])
                            scene.set_pose(cam_node, pose)
                            scene.set_pose(light_node, pose)

                            color, depth = renderer.render(scene, flags=render_flags)

                            if is_transparent:
                                # color is RGBA
                                frames.append(color)
                            else:
                                # color is RGB
                                frames.append(color)

                        # Save GIF
                        gif_name = glb_path.stem + ".gif"
                        gif_path = gifs_dir / gif_name

                        if is_transparent:
                            with imageio.get_writer(str(gif_path), mode='I', fps=15, loop=0, disposal=2) as writer:
                                for frame in frames:
                                    writer.append_data(frame)
                        else:
                            with imageio.get_writer(str(gif_path), mode='I', fps=15, loop=0) as writer:
                                for frame in frames:
                                    writer.append_data(frame)

                        logs.append(f"OK {glb_path.name} -> {gif_name}")
                        success_count += 1

                    except Exception as e:
                        logs.append(f"FAIL {glb_path.name}: {str(e)[:80]}")
                        fail_count += 1

                renderer.delete()

                logs.append("")
                logs.append(f"Complete: {success_count} succeeded, {fail_count} failed")
                logs.append(f"Output: {gifs_dir}")

                return "\n".join(logs)

            def refresh_benchmark_folders():
                """Refresh list of available benchmark folders."""
                benchmark_base = Path("/home/ai/gr/DreamEdit3D/evaluation/benchmark")
                folders = []
                if benchmark_base.exists():
                    for folder in benchmark_base.iterdir():
                        if folder.is_dir() and (folder / "glbs").exists():
                            folders.append(folder.name)
                return gr.Dropdown(choices=folders if folders else ["mvedit", "ours", "vox-e", "preditor3d"])


            def discover_benchmark_methods():
                """Scan evaluation/benchmark/cases/ for available methods.
                Finds methods from mesh files ({method}.glb/.obj) and also from
                pre-existing render directories (renders/{method}/ with rgb_*.png)."""
                cases_dir = Path("/home/ai/gr/DreamEdit3D/evaluation/benchmark/cases")
                method_counts = {}  # method_name -> count of cases it appears in
                has_glbs = set()    # methods that have at least one mesh file
                skip_names = {"original", "meta"}
                if cases_dir.exists():
                    for case_dir in sorted(cases_dir.iterdir()):
                        if case_dir.is_dir() and (case_dir / "meta.json").exists():
                            # Methods from mesh files
                            for mesh_file in list(case_dir.glob("*.glb")) + list(case_dir.glob("*.obj")):
                                if mesh_file.stem in skip_names:
                                    continue
                                method_name = mesh_file.stem
                                method_counts[method_name] = method_counts.get(method_name, 0) + 1
                                has_glbs.add(method_name)
                            # Methods from pre-existing renders (no mesh file)
                            renders_dir = case_dir / "renders"
                            if renders_dir.exists():
                                for rdir in renders_dir.iterdir():
                                    if rdir.is_dir() and rdir.name not in skip_names and rdir.name != "original":
                                        if list(rdir.glob("rgb_*.png")):
                                            method_counts[rdir.name] = method_counts.get(rdir.name, 0) + 1
                methods = []
                for name in sorted(method_counts.keys()):
                    methods.append({"name": name, "count": method_counts[name],
                                    "has_glbs": name in has_glbs})
                return methods

            def discover_benchmark_cases():
                """Scan evaluation/benchmark/cases/ and return folder names that contain meta.json."""
                cases_dir = Path("/home/ai/gr/DreamEdit3D/evaluation/benchmark/cases")
                case_names = []
                if cases_dir.exists():
                    for case_dir in sorted(cases_dir.iterdir()):
                        if case_dir.is_dir() and (case_dir / "meta.json").exists():
                            case_names.append(case_dir.name)
                return case_names

            def refresh_benchmark_cases_ui():
                """Update case checkboxes based on what directories exist."""
                case_names = discover_benchmark_cases()
                return gr.CheckboxGroup(choices=case_names, value=case_names)

            def refresh_benchmark_methods_ui():
                """Update method checkboxes based on what directories exist."""
                methods_info = discover_benchmark_methods()
                all_names = [m["name"] for m in methods_info]
                checked = [m["name"] for m in methods_info if m["has_glbs"]]
                return gr.CheckboxGroup(choices=all_names if all_names else ["mvedit", "ours", "vox-e", "preditor3d"],
                                        value=checked)

            def run_full_benchmark(selected_cases, selected_methods, selected_metrics, iqa_dimensions,
                                   num_views, do_normalize, object_scale, start_azimuth,
                                   force_rerender, force_recompute, progress=gr.Progress()):
                """One-click benchmark: render views + score all cases from cases/ folders."""
                import json as _json

                cases_dir = Path("/home/ai/gr/DreamEdit3D/evaluation/benchmark/cases")
                benchmark_base = Path("/home/ai/gr/DreamEdit3D/evaluation/benchmark")
                project_root = os.path.dirname(__file__)
                render_script = os.path.join(project_root, "evaluation", "rendering", "render_views.py")

                logs = []
                def log(msg):
                    logs.append(msg)

                if not selected_methods:
                    return "No methods selected.", "_Select at least one method._"
                if not selected_metrics:
                    return "No metrics selected.", "_Select at least one metric._"
                if not cases_dir.exists():
                    return f"Cases dir not found: {cases_dir}", "_Missing evaluation/benchmark/cases/_"

                # ========== Discover cases ==========
                cases = []
                for case_dir in sorted(cases_dir.iterdir()):
                    if not case_dir.is_dir():
                        continue
                    # Filter by selected cases (if provided)
                    if selected_cases and case_dir.name not in selected_cases:
                        continue
                    meta_path = case_dir / "meta.json"
                    if not meta_path.exists():
                        continue
                    with open(meta_path) as f:
                        meta = _json.load(f)
                    # Find original mesh (.glb or .obj)
                    original_path = None
                    for ext in (".glb", ".obj"):
                        p = case_dir / f"original{ext}"
                        if p.exists():
                            original_path = p
                            break

                    case_info = {
                        "name": case_dir.name,
                        "dir": case_dir,
                        "text_input": meta.get("text_input", ""),
                        "text_edit": meta.get("text_edit", ""),
                        "has_original": original_path is not None,
                        "original_path": original_path,
                        "methods": {},
                    }
                    # Find method meshes (*.glb or *.obj, excluding original.*)
                    skip_stems = {"original", "meta"}
                    for mesh_file in list(case_dir.glob("*.glb")) + list(case_dir.glob("*.obj")):
                        if mesh_file.stem in skip_stems:
                            continue
                        method_name = mesh_file.stem
                        if method_name in selected_methods:
                            case_info["methods"][method_name] = mesh_file

                    # Also include selected methods that have pre-rendered images
                    # but no mesh file (e.g. vox-e with external renders)
                    renders_dir = case_dir / "renders"
                    if renders_dir.exists():
                        for method_name in selected_methods:
                            if method_name in case_info["methods"]:
                                continue  # already found via mesh file
                            method_render_dir = renders_dir / method_name
                            if method_render_dir.exists() and list(method_render_dir.glob("rgb_*.png")):
                                # None signals: has renders but no mesh to render from
                                case_info["methods"][method_name] = None

                    cases.append(case_info)

                if not cases:
                    return "No cases found.", "_No case folders with meta.json in evaluation/benchmark/cases/_"

                log(f"Found {len(cases)} cases, methods: {selected_methods}")
                num_views = int(num_views)

                # Count total work for progress
                render_count = 0
                for case in cases:
                    if case["has_original"]:
                        render_count += 1  # original
                    render_count += len(case["methods"])  # methods
                score_count = sum(len(c["methods"]) for c in cases)
                total_steps = render_count + score_count
                step = 0

                def _render_glb(glb_path, output_dir, azimuth):
                    """Render a GLB file if not already rendered."""
                    existing = list(output_dir.glob("rgb_*.png")) if output_dir.exists() else []
                    if len(existing) >= num_views and not force_rerender:
                        return "skip", len(existing)
                    if force_rerender and existing:
                        import shutil
                        shutil.rmtree(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        cmd = [
                            sys.executable, render_script,
                            "--mesh_path", str(glb_path),
                            "--output_dir", str(output_dir),
                            "--num_views", str(num_views),
                            "--resolution", "512",
                            "--object_scale", str(float(object_scale)),
                            "--azimuth_start", str(azimuth),
                        ]
                        if not do_normalize:
                            cmd.append("--no_normalize")
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=project_root)
                        if result.returncode == 0:
                            return "ok", num_views
                        else:
                            err = result.stderr.strip().split('\n')[-1] if result.stderr else "Unknown"
                            return "error", err[:80]
                    except subprocess.TimeoutExpired:
                        return "timeout", None
                    except Exception as e:
                        return "error", str(e)

                # ========== PHASE 1: Render ==========
                log("=== Phase 1: Rendering views ===")
                for case in cases:
                    case_name = case["name"]
                    renders_dir = case["dir"] / "renders"

                    # Render original
                    if case["has_original"]:
                        step += 1
                        progress(step / max(total_steps, 1), desc=f"Render {case_name}/original")
                        orig_render_dir = renders_dir / "original"
                        status, info = _render_glb(
                            case["original_path"], orig_render_dir, 0.0
                        )
                        if status == "skip":
                            log(f"  {case_name}/original: skip ({info} exist)")
                        elif status == "ok":
                            log(f"  {case_name}/original: rendered {info} views")
                        else:
                            log(f"  {case_name}/original: {status.upper()}: {info}")

                    # Render each method
                    for method_name, glb_path in case["methods"].items():
                        step += 1
                        progress(step / max(total_steps, 1), desc=f"Render {case_name}/{method_name}")
                        if glb_path is None:
                            log(f"  {case_name}/{method_name}: using pre-existing renders")
                            continue
                        method_render_dir = renders_dir / method_name
                        method_azimuth = float(start_azimuth)
                        status, info = _render_glb(glb_path, method_render_dir, method_azimuth)
                        if status == "skip":
                            log(f"  {case_name}/{method_name}: skip ({info} exist)")
                        elif status == "ok":
                            log(f"  {case_name}/{method_name}: rendered {info} views")
                        else:
                            log(f"  {case_name}/{method_name}: {status.upper()}: {info}")

                # ========== PHASE 1b: Generate GIFs from renders ==========
                log("\n=== Phase 1b: Generating GIFs from renders ===")
                try:
                    import imageio.v2 as imageio
                except ImportError:
                    import imageio
                for case in cases:
                    case_name = case["name"]
                    renders_dir = case["dir"] / "renders"
                    gifs_dir = case["dir"] / "gifs"

                    # GIF for original
                    if case["has_original"]:
                        orig_render_dir = renders_dir / "original"
                        orig_images = sorted(orig_render_dir.glob("rgb_*.png")) if orig_render_dir.exists() else []
                        if orig_images:
                            gifs_dir.mkdir(exist_ok=True, parents=True)
                            gif_path = gifs_dir / "original.gif"
                            if force_rerender and gif_path.exists():
                                gif_path.unlink()
                            if not gif_path.exists():
                                try:
                                    frames = [imageio.imread(str(p)) for p in orig_images]
                                    imageio.mimsave(str(gif_path), frames, fps=15, loop=0)
                                    log(f"  {case_name}/gifs/original.gif: {len(frames)} frames")
                                except Exception as e:
                                    log(f"  {case_name}/gifs/original.gif: ERROR: {e}")

                    # GIF for each method
                    for method_name in case["methods"]:
                        method_render_dir = renders_dir / method_name
                        method_images = sorted(method_render_dir.glob("rgb_*.png")) if method_render_dir.exists() else []
                        if method_images:
                            gifs_dir.mkdir(exist_ok=True, parents=True)
                            gif_path = gifs_dir / f"{method_name}.gif"
                            if force_rerender and gif_path.exists():
                                gif_path.unlink()
                            if not gif_path.exists():
                                try:
                                    frames = [imageio.imread(str(p)) for p in method_images]
                                    imageio.mimsave(str(gif_path), frames, fps=15, loop=0)
                                    log(f"  {case_name}/gifs/{method_name}.gif: {len(frames)} frames")
                                except Exception as e:
                                    log(f"  {case_name}/gifs/{method_name}.gif: ERROR: {e}")

                # ========== PHASE 2: Init evaluators ==========
                log("\n=== Phase 2: Initializing evaluators ===")
                clip_evaluator = None
                iqa_evaluator = None
                need_clip = "CLIP Score" in selected_metrics or "CLIP Directional" in selected_metrics

                if need_clip:
                    if CLIP_EVAL_AVAILABLE:
                        success, msg = eval_manager.init_clip_evaluator()
                        if success:
                            clip_evaluator = eval_manager.clip_evaluator
                            log("  CLIP evaluator: ready")
                        else:
                            log(f"  CLIP evaluator: FAILED - {msg}")
                    else:
                        log("  CLIP evaluator: not available (missing dependencies)")

                if "CLIP-IQA" in selected_metrics:
                    if CLIP_IQA_AVAILABLE:
                        success, msg = eval_manager.init_clip_iqa_evaluator()
                        if success:
                            iqa_evaluator = eval_manager.clip_iqa_evaluator
                            log("  CLIP-IQA evaluator: ready")
                        else:
                            log(f"  CLIP-IQA evaluator: FAILED - {msg}")
                    else:
                        log("  CLIP-IQA evaluator: not available (missing dependencies)")

                if not iqa_dimensions:
                    iqa_dimensions = ["quality", "render_quality", "geometry", "texture"]

                do_clip_score = "CLIP Score" in selected_metrics and clip_evaluator is not None
                do_clip_dir = "CLIP Directional" in selected_metrics and clip_evaluator is not None
                do_iqa = "CLIP-IQA" in selected_metrics and iqa_evaluator is not None

                # ========== PHASE 3: Score per case ==========
                log("\n=== Phase 3: Scoring benchmark cases ===")
                all_results = {}

                for case in cases:
                    case_name = case["name"]
                    text_input = case["text_input"]
                    text_edit = case["text_edit"]
                    renders_dir = case["dir"] / "renders"

                    all_results[case_name] = {}

                    # Load original renders once per case (for CLIP directional)
                    original_images = None
                    if do_clip_dir and case["has_original"]:
                        orig_render_dir = renders_dir / "original"
                        if orig_render_dir.exists() and list(orig_render_dir.glob("rgb_*.png")):
                            try:
                                original_images = clip_evaluator.load_images_from_dir(
                                    str(orig_render_dir), pattern="rgb_*.png")
                            except Exception as e:
                                log(f"  {case_name}: failed to load original renders: {e}")

                    for method_name in case["methods"]:
                        step += 1
                        progress(step / max(total_steps, 1), desc=f"Score {case_name}/{method_name}")

                        # Try loading existing results first (skip recalculation)
                        result_path = case["dir"] / f"results_{method_name}.json"
                        if not force_rerender and not force_recompute and result_path.exists():
                            try:
                                with open(result_path) as f:
                                    saved = _json.load(f)
                                metrics = saved.get("metrics", {})
                                all_results[case_name][method_name] = metrics
                                parts = []
                                if "CLIP_score" in metrics:
                                    parts.append(f"CLIP={metrics['CLIP_score']:.2f}")
                                if "CLIP_dir-cos" in metrics:
                                    parts.append(f"dir-cos={metrics['CLIP_dir-cos']:.4f}")
                                for dim in iqa_dimensions[:2]:
                                    k = f"IQA_{dim}"
                                    if k in metrics:
                                        parts.append(f"{dim}={metrics[k]:.3f}")
                                log(f"  {case_name}/{method_name}: loaded ({', '.join(parts)})")
                                continue
                            except Exception:
                                pass  # fall through to recalculate

                        edit_dir = renders_dir / method_name
                        if not edit_dir.exists() or not list(edit_dir.glob("rgb_*.png")):
                            log(f"  SKIP {case_name}/{method_name}: no renders")
                            continue

                        metrics = {}

                        # CLIP Score with per-case text_edit prompt
                        if do_clip_score:
                            try:
                                images = clip_evaluator.load_images_from_dir(str(edit_dir), pattern="rgb_*.png")
                                result = clip_evaluator.clip_score(images, text_edit)
                                metrics["CLIP_score"] = result["CLIP_score_mean"]
                            except Exception as e:
                                log(f"    {case_name}/{method_name} CLIP error: {e}")

                        # CLIP Directional (requires original renders + both text prompts)
                        if do_clip_dir and original_images and text_input and text_edit:
                            try:
                                edit_images = clip_evaluator.load_images_from_dir(str(edit_dir), pattern="rgb_*.png")
                                # Match view counts (use min of both)
                                n = min(len(original_images), len(edit_images))
                                dir_results = clip_evaluator.clip_dir_variants(
                                    text_input, text_edit,
                                    original_images[:n], edit_images[:n]
                                )
                                for k, v in dir_results.items():
                                    metrics[k] = v
                            except Exception as e:
                                log(f"    {case_name}/{method_name} CLIP-dir error: {e}")

                        # CLIP-IQA
                        if do_iqa:
                            try:
                                per_image = iqa_evaluator.score_directory(str(edit_dir), prompt_names=iqa_dimensions)
                                agg = iqa_evaluator.aggregate_scores(per_image)
                                for dim in iqa_dimensions:
                                    key = f"{dim}_mean"
                                    if key in agg:
                                        metrics[f"IQA_{dim}"] = agg[key]
                            except Exception as e:
                                log(f"    {case_name}/{method_name} IQA error: {e}")

                        all_results[case_name][method_name] = metrics

                        # Brief log
                        parts = []
                        if "CLIP_score" in metrics:
                            parts.append(f"CLIP={metrics['CLIP_score']:.2f}")
                        if "CLIP_dir-cos" in metrics:
                            parts.append(f"dir-cos={metrics['CLIP_dir-cos']:.4f}")
                        for dim in iqa_dimensions[:2]:
                            k = f"IQA_{dim}"
                            if k in metrics:
                                parts.append(f"{dim}={metrics[k]:.3f}")
                        log(f"  {case_name}/{method_name}: {', '.join(parts) if parts else 'no metrics'}")

                        # Save per-case results
                        try:
                            with open(result_path, "w") as f:
                                _json.dump({"case": case_name, "method": method_name,
                                           "text_input": text_input, "text_edit": text_edit,
                                           "metrics": metrics}, f, indent=2)
                        except Exception:
                            pass

                progress(1.0, desc="Done")

                # ========== PHASE 4: Build markdown table ==========
                md_lines = ["## Benchmark Results\n"]

                # Determine which metric columns exist
                dir_metric_keys = ["CLIP_dir", "CLIP_dir-cos", "CLIP_dir-avg", "CLIP_dir-avg-cos"]
                all_possible_keys = ["CLIP_score"] + dir_metric_keys + [f"IQA_{d}" for d in iqa_dimensions]
                metric_keys = []
                for mk in all_possible_keys:
                    for case_r in all_results.values():
                        for method_r in case_r.values():
                            if mk in method_r and mk not in metric_keys:
                                metric_keys.append(mk)

                if not metric_keys:
                    md_lines.append("_No metrics were computed. Check that evaluators loaded correctly._")
                else:
                    # Header
                    md_lines.append("| Case | Method | " + " | ".join(metric_keys) + " |")
                    md_lines.append("|------|--------|" + "|".join(["--------"] * len(metric_keys)) + "|")

                    # Rows
                    method_totals = {}  # method -> {metric -> [values]}
                    for case_name, methods_dict in all_results.items():
                        for method_name, metrics in methods_dict.items():
                            row = f"| {case_name} | {method_name} |"
                            for k in metric_keys:
                                val = metrics.get(k)
                                if val is not None:
                                    row += f" {val:.4f} |"
                                    method_totals.setdefault(method_name, {}).setdefault(k, []).append(val)
                                else:
                                    row += " -- |"
                            md_lines.append(row)

                    # Method averages (ours last, best value per column bolded)
                    if method_totals:
                        md_lines.append("")
                        md_lines.append("### Method Averages\n")
                        md_lines.append("| Method | " + " | ".join(metric_keys) + " |")
                        md_lines.append("|--------|" + "|".join(["--------"] * len(metric_keys)) + "|")

                        # Compute mean per method per metric
                        method_means = {}
                        for method_name, totals in method_totals.items():
                            method_means[method_name] = {}
                            for k in metric_keys:
                                vals = totals.get(k, [])
                                method_means[method_name][k] = np.mean(vals) if vals else None

                        # Find best (highest) value per metric
                        best_per_metric = {}
                        for k in metric_keys:
                            vals = [method_means[m][k] for m in method_means if method_means[m][k] is not None]
                            best_per_metric[k] = max(vals) if vals else None

                        # Sort methods: "ours" last, rest alphabetical
                        sorted_methods = sorted(
                            method_totals.keys(),
                            key=lambda m: (1 if m == "ours" else 0, m)
                        )

                        for method_name in sorted_methods:
                            row = f"| **{method_name}** |"
                            for k in metric_keys:
                                val = method_means[method_name][k]
                                if val is not None:
                                    if best_per_metric[k] is not None and val >= best_per_metric[k] - 1e-8:
                                        row += f" **{val:.4f}** |"
                                    else:
                                        row += f" {val:.4f} |"
                                else:
                                    row += " -- |"
                            md_lines.append(row)

                results_md = "\n".join(md_lines)

                # ========== PHASE 5: Save ==========
                output_path = benchmark_base / "benchmark_results.json"
                combined = {"cases": all_results}
                try:
                    with open(output_path, "w") as f:
                        _json.dump(combined, f, indent=2)
                    log(f"\nResults saved to: {output_path}")
                except Exception as e:
                    log(f"\nFailed to save results: {e}")

                log(f"\nBenchmark complete: {len(cases)} cases scored.")
                return "\n".join(logs), results_md

            # Event handlers for Evaluation tab
            project_dropdown.change(
                fn=update_eval_mesh_list,
                inputs=[project_dropdown],
                outputs=[eval_mesh_dropdown, eval_render_status]
            )

            # Auto-populate prompts when project changes
            project_dropdown.change(
                fn=load_saved_prompts,
                inputs=[project_dropdown],
                outputs=[clip_score_text, clip_text_input, clip_text_edit, gpt_prompt]
            )

            eval_refresh_btn.click(
                fn=update_eval_mesh_list,
                inputs=[current_project],
                outputs=[eval_mesh_dropdown, eval_render_status]
            )

            eval_refresh_btn.click(
                fn=load_saved_prompts,
                inputs=[current_project],
                outputs=[clip_score_text, clip_text_input, clip_text_edit, gpt_prompt]
            )

            def load_prompts_for_mesh(project_path, mesh_file):
                """Load prompts for a specific mesh based on its inference entry in project_info.json"""
                if not project_path or not mesh_file or not project_manager:
                    return "", "", "", ""

                # Get training prompt (original concept)
                training_prompts, _ = project_manager.get_latest_prompts(project_path)
                original_text = ""
                if training_prompts:
                    original_text = training_prompts[0] if isinstance(training_prompts, list) else training_prompts

                # Get mesh folder name (e.g., generated_multiview_1769078295)
                mesh_name = Path(mesh_file).parent.name

                # Find the inference entry for this specific mesh
                info = project_manager.get_project_info(project_path)
                inference_list = info.get("prompts", {}).get("inference", []) if info else []

                edited_text = original_text
                gpt_prompt_text = original_text

                for entry in inference_list:
                    gen_file = entry.get("generated_file", "")
                    # Match mesh folder name
                    if mesh_name in gen_file.replace(".jpg", "").replace(".png", ""):
                        prompts = entry.get("prompts", [])
                        if prompts:
                            raw_prompt = prompts[0] if isinstance(prompts, list) else prompts
                            # Replace <asset0> with training prompt
                            if original_text:
                                edited_text = raw_prompt.replace("<asset0>", original_text)
                            else:
                                edited_text = raw_prompt
                            gpt_prompt_text = edited_text
                        break

                # clip_score_text gets the edited text (most useful for CLIP Score)
                return edited_text, original_text, edited_text, gpt_prompt_text

            eval_mesh_dropdown.change(
                fn=check_renders_status,
                inputs=[current_project, eval_mesh_dropdown],
                outputs=[eval_render_status, eval_rendered_gallery]
            )

            # Update prompts when mesh selection changes
            eval_mesh_dropdown.change(
                fn=load_prompts_for_mesh,
                inputs=[current_project, eval_mesh_dropdown],
                outputs=[clip_score_text, clip_text_input, clip_text_edit, gpt_prompt]
            )

            def render_all_meshes_for_eval(project_path, num_views, do_normalize, object_scale, progress=gr.Progress()):
                """Render views for all meshes in the project"""
                if not project_path:
                    return "Please select a project first", []

                # Get all meshes from 04_gtr_3d
                meshes = mesh_visualizer.get_available_meshes(project_path)

                if not meshes:
                    return "No meshes found in project", []

                logs = []
                logs.append(f"Found {len(meshes)} mesh(es) to render:\n")
                for m in meshes:
                    logs.append(f"  - {Path(m).parent.name}")
                logs.append("\n" + "="*50 + "\n")

                success_count = 0
                fail_count = 0
                last_gallery = []

                for i, mesh_path in enumerate(meshes):
                    # Convert to relative path for get_mesh_eval_folder
                    mesh_file = str(Path(mesh_path).relative_to(project_path))
                    mesh_name = Path(mesh_path).parent.name

                    progress((i + 1) / len(meshes), desc=f"Rendering {mesh_name} ({i+1}/{len(meshes)})")

                    logs.append(f"\n[{i+1}/{len(meshes)}] Rendering: {mesh_name}")
                    logs.append("-" * 40)

                    try:
                        # Get mesh-specific evaluation folder
                        eval_dir, _ = get_mesh_eval_folder(project_path, mesh_file)
                        render_dir = eval_dir / "renders"

                        success, msg = eval_manager.render_mesh_for_eval(
                            mesh_path=mesh_path,
                            output_dir=str(render_dir),
                            num_views=int(num_views),
                            progress_callback=None,
                            do_normalize=do_normalize,
                            object_scale=float(object_scale)
                        )

                        if success:
                            success_count += 1
                            logs.append(f"✅ Success: {render_dir}")
                            # Load gallery from last successful render
                            rgb_files = sorted(render_dir.glob("rgb_*.png"))[:12]
                            last_gallery = [str(f) for f in rgb_files]
                        else:
                            fail_count += 1
                            logs.append(f"❌ Failed: {msg[:200]}")

                    except Exception as e:
                        fail_count += 1
                        logs.append(f"❌ Error: {str(e)[:200]}")

                # Summary
                logs.append("\n" + "="*50)
                logs.append(f"\n📊 SUMMARY: {success_count} succeeded, {fail_count} failed out of {len(meshes)} total")

                return "\n".join(logs), last_gallery

            render_views_btn.click(
                fn=render_views_for_eval,
                inputs=[current_project, eval_mesh_dropdown, eval_num_views, eval_normalize, eval_object_scale],
                outputs=[eval_render_status, eval_rendered_gallery, clip_input_dir]
            )

            render_all_btn.click(
                fn=render_all_meshes_for_eval,
                inputs=[current_project, eval_num_views, eval_normalize, eval_object_scale],
                outputs=[eval_render_status, eval_rendered_gallery]
            )

            # Per-tab evaluation buttons
            def run_clip_score_only(project_path, mesh_file, clip_score_text_val):
                """Run CLIP Score evaluation only"""
                status, clip_md, _, _ = run_evaluation(
                    project_path, mesh_file, "CLIP Score",
                    "", clip_score_text_val, "", "",
                    "", "", [], ""
                )
                return clip_md

            def run_clip_metrics_only(project_path, mesh_file,
                                      clip_input, clip_edit, clip_word, clip_generic,
                                      input_dir, edit_dir):
                """Run CLIP Metrics evaluation only"""
                status, clip_md, _, _ = run_evaluation(
                    project_path, mesh_file, "CLIP Metrics",
                    clip_input, clip_edit, clip_word, clip_generic,
                    input_dir, edit_dir, [], ""
                )
                return clip_md

            def run_clip_iqa_only(project_path, mesh_file, iqa_prompts):
                """Run CLIP-IQA evaluation only"""
                status, _, iqa_md, _ = run_evaluation(
                    project_path, mesh_file, "CLIP-IQA",
                    "", "", "", "",
                    "", "", iqa_prompts, ""
                )
                return iqa_md

            def run_gpt_eval_only(project_path, mesh_file, gpt_prompt_text):
                """Run GPT-4V evaluation only"""
                status, _, _, gpt_md = run_evaluation(
                    project_path, mesh_file, "GPT-4V Eval",
                    "", "", "", "",
                    "", "", [], gpt_prompt_text
                )
                return gpt_md

            run_clip_score_btn.click(
                fn=run_clip_score_only,
                inputs=[current_project, eval_mesh_dropdown, clip_score_text],
                outputs=[clip_score_results]
            )

            run_clip_metrics_btn.click(
                fn=run_clip_metrics_only,
                inputs=[current_project, eval_mesh_dropdown,
                        clip_text_input, clip_text_edit, clip_edited_word, clip_generic_text,
                        clip_input_dir, clip_edit_dir],
                outputs=[clip_metrics_results]
            )

            run_clip_iqa_btn.click(
                fn=run_clip_iqa_only,
                inputs=[current_project, eval_mesh_dropdown, clip_iqa_prompts],
                outputs=[clip_iqa_results_display]
            )

            run_gpt_eval_btn.click(
                fn=run_gpt_eval_only,
                inputs=[current_project, eval_mesh_dropdown, gpt_prompt],
                outputs=[gpt_results_display]
            )

            def run_all_evaluations(project_path, eval_mode_list, clip_word, clip_generic, clip_iqa_prompts,
                                   progress=gr.Progress()):
                """Run evaluation for all meshes in the project

                For CLIP metrics:
                - Find the mesh with prompt "<asset0>" as the ORIGINAL/BASELINE
                - Compare other meshes (edited) against the original
                """
                # Convert checkbox list to "All" if multiple selected, or single mode
                if not eval_mode_list:
                    eval_mode_list = ["CLIP-IQA"]
                eval_mode = "All" if len(eval_mode_list) > 1 else eval_mode_list[0]
                if not project_path:
                    return "Please select a project first", "_No results_", "_No results_", "_No results_"

                # Get all meshes
                meshes = mesh_visualizer.get_available_meshes(project_path)
                if not meshes:
                    return "No meshes found in project", "_No results_", "_No results_", "_No results_"

                # Get prompts from project_info.json
                training_prompts, _ = project_manager.get_latest_prompts(project_path)

                # Get training prompt (original concept name)
                original_text = ""
                if training_prompts:
                    original_text = training_prompts[0] if isinstance(training_prompts, list) else training_prompts

                # Get project info to match inference prompts to meshes
                info = project_manager.get_project_info(project_path)
                inference_list = info.get("prompts", {}).get("inference", []) if info else []

                # Build mapping: mesh_folder -> (raw_prompt, resolved_prompt)
                file_to_prompt = {}
                baseline_mesh = None  # Mesh with just "<asset0>" prompt

                for entry in inference_list:
                    gen_file = entry.get("generated_file", "")
                    prompts = entry.get("prompts", [])
                    if gen_file and prompts:
                        mesh_folder = gen_file.replace(".jpg", "").replace(".png", "")
                        raw_prompt = prompts[0] if isinstance(prompts, list) else prompts

                        # Check if this is the baseline (just "<asset0>" with no edits)
                        if raw_prompt.strip() == "<asset0>":
                            baseline_mesh = mesh_folder

                        # Replace <asset0> with training prompt for display
                        resolved_prompt = raw_prompt.replace("<asset0>", original_text) if original_text else raw_prompt
                        file_to_prompt[mesh_folder] = {
                            "raw": raw_prompt,
                            "resolved": resolved_prompt,
                            "is_baseline": raw_prompt.strip() == "<asset0>"
                        }

                logs = []
                logs.append(f"Found {len(meshes)} mesh(es) to evaluate")
                logs.append(f"Training concept: {original_text}")

                # Find baseline renders directory
                baseline_render_dir = None
                if baseline_mesh:
                    for mesh_path in meshes:
                        if baseline_mesh in str(mesh_path):
                            mesh_file = str(Path(mesh_path).relative_to(project_path))
                            eval_dir, _ = get_mesh_eval_folder(project_path, mesh_file)
                            baseline_render_dir = eval_dir / "renders"
                            if baseline_render_dir.exists() and len(list(baseline_render_dir.glob("rgb_*.png"))) > 0:
                                logs.append(f"Baseline mesh (original): {baseline_mesh}")
                                logs.append(f"Baseline renders: {baseline_render_dir}")
                            else:
                                baseline_render_dir = None
                                logs.append(f"⚠️ Baseline mesh found ({baseline_mesh}) but no renders - render it first!")
                            break

                if not baseline_render_dir:
                    logs.append("⚠️ No baseline mesh found (mesh with just '<asset0>' prompt)")
                    logs.append("   CLIP will compare each mesh to itself (less meaningful)")

                logs.append("="*50 + "\n")

                success_count = 0
                fail_count = 0
                skip_count = 0
                all_clip_results = []
                all_clip_iqa_results = []
                all_gpt_results = []

                for i, mesh_path in enumerate(meshes):
                    mesh_file = str(Path(mesh_path).relative_to(project_path))
                    mesh_name = Path(mesh_path).parent.name

                    progress((i + 1) / len(meshes), desc=f"Evaluating {mesh_name} ({i+1}/{len(meshes)})")

                    # Get prompt info for this mesh
                    prompt_info = file_to_prompt.get(mesh_name, {"raw": "", "resolved": original_text, "is_baseline": False})
                    resolved_prompt = prompt_info["resolved"]
                    is_baseline = prompt_info["is_baseline"]

                    logs.append(f"\n[{i+1}/{len(meshes)}] {mesh_name}")
                    logs.append("-" * 40)
                    logs.append(f"  Prompt: {resolved_prompt}")

                    # Skip baseline mesh for CLIP (it's used as reference, not evaluated)
                    if is_baseline and baseline_render_dir:
                        logs.append(f"  ⏭️ Skipping (this is the baseline/original)")
                        skip_count += 1
                        continue

                    # Check if renders exist for this mesh
                    eval_dir, _ = get_mesh_eval_folder(project_path, mesh_file)
                    render_dir = eval_dir / "renders"

                    if not render_dir.exists() or len(list(render_dir.glob("rgb_*.png"))) == 0:
                        logs.append(f"  ⚠️ No renders found - skipping (run 'Render All Meshes' first)")
                        fail_count += 1
                        continue

                    # Determine input (original) and edit renders directories
                    input_render_dir = str(baseline_render_dir) if baseline_render_dir else str(render_dir)
                    edit_render_dir = str(render_dir)

                    if baseline_render_dir:
                        logs.append(f"  Input (original): {baseline_mesh}")
                        logs.append(f"  Edit (this mesh): {mesh_name}")

                    try:
                        # Run evaluation for this mesh
                        status, clip_md, clip_iqa_md, gpt_md = run_evaluation(
                            project_path, mesh_file, eval_mode,
                            original_text, resolved_prompt, clip_word, clip_generic,
                            input_render_dir, edit_render_dir, clip_iqa_prompts, resolved_prompt
                        )

                        if "✅" in status:
                            success_count += 1
                            logs.append(f"  ✅ Evaluation complete")
                            all_clip_results.append(f"### {mesh_name}\n**Prompt:** {resolved_prompt}\n{clip_md}")
                            all_clip_iqa_results.append(f"### {mesh_name}\n**Prompt:** {resolved_prompt}\n{clip_iqa_md}")
                            all_gpt_results.append(f"### {mesh_name}\n**Prompt:** {resolved_prompt}\n{gpt_md}")
                        else:
                            fail_count += 1
                            logs.append(f"  ❌ {status[:100]}")

                    except Exception as e:
                        fail_count += 1
                        logs.append(f"  ❌ Error: {str(e)[:100]}")

                # Summary
                logs.append("\n" + "="*50)
                logs.append(f"\n📊 SUMMARY:")
                logs.append(f"   ✅ Evaluated: {success_count}")
                logs.append(f"   ⏭️ Skipped (baseline): {skip_count}")
                logs.append(f"   ❌ Failed: {fail_count}")
                logs.append(f"   Total meshes: {len(meshes)}")

                # Combine results
                combined_clip = "\n\n".join(all_clip_results) if all_clip_results else "_No CLIP results_"
                combined_clip_iqa = "\n\n".join(all_clip_iqa_results) if all_clip_iqa_results else "_No CLIP-IQA results_"
                combined_gpt = "\n\n".join(all_gpt_results) if all_gpt_results else "_No GPT results_"

                return "\n".join(logs), combined_clip, combined_clip_iqa, combined_gpt

            run_all_eval_btn.click(
                fn=run_all_evaluations,
                inputs=[
                    current_project, eval_mode_checks,
                    clip_edited_word, clip_generic_text, clip_iqa_prompts
                ],
                outputs=[eval_progress, clip_results_display, clip_iqa_all_results, gpt_all_results]
            )

            def generate_report(project_path):
                """Generate PDF report and update averages display"""
                if not project_path:
                    return None, "Please select a project first"

                pdf_path, msg = eval_manager.generate_gpteval_report(project_path)

                if pdf_path:
                    avg_display = eval_manager.format_average_scores(project_path)
                    return pdf_path, f"✅ {msg}\n\n{avg_display}"
                else:
                    return None, f"❌ {msg}"

            def refresh_averages(project_path):
                """Refresh average scores display"""
                if not project_path:
                    return "_Please select a project first_"
                return eval_manager.format_average_scores(project_path)

            generate_report_btn.click(
                fn=generate_report,
                inputs=[current_project],
                outputs=[report_download, avg_scores_display]
            )

            refresh_avg_btn.click(
                fn=refresh_averages,
                inputs=[current_project],
                outputs=[avg_scores_display]
            )

            # Also refresh averages when project changes
            project_dropdown.change(
                fn=refresh_averages,
                inputs=[project_dropdown],
                outputs=[avg_scores_display]
            )

            # Benchmark GIF rendering event handlers
            render_benchmark_btn.click(
                fn=render_benchmark_glbs_to_gifs,
                inputs=[benchmark_folder, bench_cam_distance, bench_cam_fov, bench_cam_elevation,
                       bench_num_frames, bench_start_azimuth_gif, bench_resolution, bench_background],
                outputs=[benchmark_status]
            )

            refresh_benchmark_btn.click(
                fn=refresh_benchmark_folders,
                outputs=[benchmark_folder]
            )

            # One-click benchmark event handlers
            refresh_cases_btn.click(
                fn=refresh_benchmark_cases_ui,
                outputs=[benchmark_case_checks]
            )

            refresh_methods_btn.click(
                fn=refresh_benchmark_methods_ui,
                outputs=[benchmark_method_checks]
            )

            run_full_benchmark_btn.click(
                fn=run_full_benchmark,
                inputs=[
                    benchmark_case_checks,
                    benchmark_method_checks,
                    benchmark_metric_checks,
                    benchmark_iqa_dims,
                    bench_full_num_views,
                    bench_full_normalize,
                    bench_full_object_scale,
                    bench_full_start_azimuth,
                    bench_full_force_rerender,
                    bench_full_force_recompute,
                ],
                outputs=[benchmark_progress_log, benchmark_results_md]
            )


    return demo

if __name__ == "__main__":
    print("🎯 Starting DreamEdit3D Complete Pipeline...")
    if not SAM_AVAILABLE:
        print("⚠️  Warning: Segment Anything not available. Please install:")
        print("   pip install segment-anything")

    # Initialize project manager
    base_dir = Path(__file__).parent
    project_manager = ProjectManager(base_dir)
    print(f"📁 Project directory: {project_manager.projects_dir}")

    demo = create_interface()

    # Create FastAPI app with CORS and static file serving
    app = FastAPI(title="DreamEdit3D Complete Pipeline")

    # Enable CORS for external access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Mount static file server for mesh outputs (serves from projects directory)
    from fastapi.staticfiles import StaticFiles
    app.mount("/mesh_static", StaticFiles(directory=str(base_dir / "projects"), html=True), name="mesh_static")

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "service": "DreamEdit3D"}

    # Mount Gradio app
    app = gr.mount_gradio_app(app, demo, path="/")

    # Launch with Uvicorn
    print("🚀 Launching server with FastAPI + Gradio...")
    print(f"📁 Project directory: {project_manager.projects_dir}")
    print("📍 Mesh GLB/HTML files saved to: projects/*/04_gtr_3d/*/")
    print("🌐 Access at: http://0.0.0.0:7860")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        reload=False,
        access_log=True,
        ws_ping_interval=None,  # Disable ping interval (no timeout)
        ws_ping_timeout=None,   # Disable ping timeout (no timeout)
        timeout_keep_alive=7200,  # 2 hours keep-alive
        limit_concurrency=1000,
        limit_max_requests=1000,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )