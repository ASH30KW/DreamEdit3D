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

# Add parent directory to path to import DreamEdit3DApp
sys.path.insert(0, str(Path(__file__).parent.parent))
from gradio_app import DreamEdit3DApp

# SAM imports
try:
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: segment_anything not installed. Install with: pip install segment-anything")
    SAM_AVAILABLE = False

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
            # Define checkpoint paths for different model sizes
            checkpoints = {
                'vit_h': 'checkpoints/sam_vit_h_4b8939.pth',  # Largest, most accurate
                'vit_l': 'checkpoints/sam_vit_l_0b3195.pth',  # Medium
                'vit_b': 'checkpoints/sam_vit_b_01ec64.pth'   # Smallest, fastest
            }
            
            checkpoint_path = checkpoints.get(model_type, checkpoints['vit_h'])
            
            # Try to load the model
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
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
current_box = None

# Load default multi-view images
DEFAULT_IMAGE_DIR = "/home/ai/gr/DreamEdit3D/mask/race-chicken-mv"

def load_default_images():
    """Load default multi-view images"""
    default_images = {}
    if os.path.exists(DEFAULT_IMAGE_DIR):
        for view_dir in sorted(glob.glob(os.path.join(DEFAULT_IMAGE_DIR, "view_*"))):
            view_name = os.path.basename(view_dir)
            img_path = os.path.join(view_dir, "img.jpg")
            if os.path.exists(img_path):
                default_images[view_name] = img_path
    return default_images

DEFAULT_IMAGES = load_default_images()

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

def handle_point_click(image, evt: gr.SelectData, point_type):
    """Handle point clicks on image"""
    global current_points
    
    if image is None:
        return image, "Please upload an image first"
    
    # Get click coordinates
    x, y = evt.index
    
    # Add point to list
    label = 1 if point_type == "Positive (include)" else 0
    current_points.append({"point": [x, y], "label": label})
    
    # Draw points on image
    points = [p["point"] for p in current_points]
    labels = [p["label"] for p in current_points]
    
    image_with_points = draw_points_on_image(np.array(image), points, labels)
    
    return image_with_points, f"Added {'positive' if label == 1 else 'negative'} point at ({x}, {y}). Total points: {len(current_points)}"

def clear_points():
    """Clear all points"""
    global current_points
    current_points = []
    sam_instance.clear_points()
    return None, "Points cleared"

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

def load_all_views():
    """Load all 4 views at once"""
    views = []
    for view_name in sorted(DEFAULT_IMAGES.keys()):
        img = select_default_image(view_name)
        if img:
            views.append(img)

    # Return 4 images (or None if not available)
    while len(views) < 4:
        views.append(None)

    return views[0], views[1], views[2], views[3]

def segment_all_views(view1, view2, view3, view4, max_objects=10):
    """Segment all 4 views and return individual masks for each"""
    results = []
    all_masks = []
    statuses = []

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

    # Combine all masks from all views into one gallery
    combined_masks = []
    for i, masks in enumerate(all_masks, 1):
        combined_masks.extend(masks)

    status_text = "\n".join(statuses)

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

def process_all_views_with_brush(view1_mask, view2_mask, view3_mask, view4_mask, save_folder_name):
    """Process all 4 views with brush + SAM segmentation and save to a new folder"""
    results = []
    masks = []
    statuses = []

    # Create output folder structure
    if not save_folder_name:
        import time
        save_folder_name = f"masked_output_{int(time.time())}"

    output_base = Path(__file__).parent / "masked_output" / save_folder_name

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
                # Create view folder
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

                    # Save original image as img.jpg
                    img_path = view_folder / "img.jpg"
                    Image.fromarray(img_array).save(img_path)

                # Save binary mask as mask0.png
                mask_path = view_folder / "mask0.png"
                mask_image = Image.fromarray(mask)
                mask_image.save(mask_path)

                statuses[-1] += f" | Saved to {mask_path}"
                saved_folder_path = str(output_base)

    status_text = "\n".join(statuses)
    if saved_folder_path:
        status_text += f"\n\n✅ All data saved to: {saved_folder_path}"

    return results[0], results[1], results[2], results[3], masks[0], masks[1], masks[2], masks[3], status_text, saved_folder_path

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

# Set paths for GTR pipeline
GTR_CKPT_PATH = "../snap_gtr/ckpts/full_checkpoint.pth"
GTR_TEMP_DIR = "../temp_gtr_gradio"
GTR_EXAMPLES_DIR = "../examples"
GTR_SCRIPTS_DIR = "../snap_gtr/scripts"

def get_gtr_input_folders():
    """Get list of available input folders for GTR"""
    input_folders = []

    # Check generated_multiview folder (main output from DreamEdit3D)
    generated_multiview_dir = Path(__file__).parent.parent / "generated_multiview"
    if generated_multiview_dir.exists():
        for folder in generated_multiview_dir.iterdir():
            if folder.is_dir():
                input_folders.append(str(folder))

    # Check mask/masked_output folders
    masked_output_dir = Path(__file__).parent / "masked_output"
    if masked_output_dir.exists():
        for folder in masked_output_dir.iterdir():
            if folder.is_dir():
                # Check if it has view folders
                view_folders = list(folder.glob("view_*"))
                if view_folders:
                    input_folders.append(str(folder))

    # Check examples directory
    examples_path = Path(GTR_EXAMPLES_DIR)
    if examples_path.exists():
        for folder in examples_path.iterdir():
            if folder.is_dir() and folder.name.endswith('-mv'):
                input_folders.append(str(folder))

    # Check default image directory
    if Path(DEFAULT_IMAGE_DIR).exists():
        input_folders.append(DEFAULT_IMAGE_DIR)

    return sorted(input_folders)

def prepare_multiview_gtr(input_file_or_dir, selected_example=None, selected_folder=None):
    """Run prepare_mv.py script for GTR"""
    try:
        # Create temp directory for outputs
        Path(GTR_TEMP_DIR).mkdir(exist_ok=True)

        # Determine input source - priority: folder > example > file
        if selected_folder:
            temp_input = selected_folder
        elif selected_example:
            temp_input = selected_example
        elif input_file_or_dir:
            # Handle uploaded file
            if hasattr(input_file_or_dir, 'name'):
                temp_input = input_file_or_dir.name
            else:
                temp_input = input_file_or_dir
        else:
            return None, "Please select a folder, example, or upload an image"

        # Create unique output directory
        out_dir = Path(GTR_TEMP_DIR) / "prepared_mv"
        out_dir.mkdir(exist_ok=True, parents=True)

        # Run prepare script
        prepare_script = Path(GTR_SCRIPTS_DIR) / "prepare_mv.py"
        cmd = [
            "python", str(prepare_script),
            "--in_dir", str(temp_input),
            "--out_dir", str(out_dir)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Get prepared images for preview
        prepared_images = sorted(list(out_dir.glob("rgb_*.png")))

        if not prepared_images:
            return None, f"Error: No images generated\n{result.stderr}"

        return str(out_dir), f"✓ Prepared {len(prepared_images)} views from {temp_input}\n{result.stdout}"

    except subprocess.CalledProcessError as e:
        return None, f"Error running prepare_mv.py:\n{e.stderr}"
    except Exception as e:
        return None, f"Error: {str(e)}"


def run_gtr_inference(prepared_dir, checkpoint_path=GTR_CKPT_PATH):
    """Run inference.py script for GTR"""
    try:
        if not prepared_dir:
            return None, None, None, "Please run preparation first"

        if not Path(checkpoint_path).exists():
            return None, None, None, f"Checkpoint not found: {checkpoint_path}"

        # Create output directory
        out_dir = Path(GTR_TEMP_DIR) / "inference_output"
        out_dir.mkdir(exist_ok=True, parents=True)

        # Run inference script from snap_gtr directory (so config paths work)
        inference_script = Path(GTR_SCRIPTS_DIR) / "inference.py"
        snap_gtr_dir = Path(__file__).parent.parent / "snap_gtr"

        # Convert paths to absolute
        abs_checkpoint = Path(checkpoint_path).resolve()
        abs_prepared_dir = Path(prepared_dir).resolve()
        abs_out_dir = out_dir.resolve()

        cmd = [
            "python", "scripts/inference.py",
            "--ckpt_path", str(abs_checkpoint),
            "--in_dir", str(abs_prepared_dir),
            "--out_dir", str(abs_out_dir),
            "--seed", "2025"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(snap_gtr_dir))

        # Get output files
        mesh_file = out_dir / "mesh.obj"
        mesh_gif = out_dir / "mesh.gif"
        nerf_gif = out_dir / "nerf.gif"

        if not mesh_file.exists():
            return None, None, None, f"Error: Mesh not generated\n{result.stderr}"

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


def full_gtr_pipeline(input_file_or_dir, checkpoint_path, selected_example=None, selected_folder=None):
    """Run both prepare and inference for GTR"""
    # Step 1: Prepare
    prepared_dir, prep_log = prepare_multiview_gtr(input_file_or_dir, selected_example, selected_folder)

    if not prepared_dir:
        return None, None, None, prep_log

    # Step 2: Inference
    mesh_file, mesh_gif, nerf_gif, inf_log = run_gtr_inference(prepared_dir, checkpoint_path)

    combined_log = f"=== PREPARATION ===\n{prep_log}\n\n=== INFERENCE ===\n{inf_log}"

    return mesh_file, mesh_gif, nerf_gif, combined_log


def get_gtr_example_images():
    """Get list of example images from examples directory"""
    example_files = []
    if Path(GTR_EXAMPLES_DIR).exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            example_files.extend(sorted(Path(GTR_EXAMPLES_DIR).glob(ext)))
    return [str(f) for f in example_files]

def get_generated_multiview_images():
    """Get list of generated multiview images from DreamEdit3D output"""
    generated_images = []
    generated_mv_dir = Path(__file__).parent.parent / "generated_multiview"
    if generated_mv_dir.exists():
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            generated_images.extend(sorted(generated_mv_dir.glob(ext), reverse=True))  # Most recent first
    return [str(f) for f in generated_images]

def get_gtr_prepared_directories():
    """Get list of prepared multi-view directories"""
    prepared_dirs = []
    examples_path = Path(GTR_EXAMPLES_DIR)

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
        """
    ) as demo:
        gr.Markdown("""
        # 🎯 DreamEdit3D: Complete 3D Generation Pipeline

        **End-to-end workflow**: Segment objects with SAM → Train 3D-aware models → Generate multi-view images → Create 3D meshes
        """)

        # State variables
        box_state = gr.State([])

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
                1. Upload an image or select a default multi-view image
                2. Select point type (Positive = include, Negative = exclude)
                3. Click on the image to add points
                4. Click "Generate Mask" to create the segmentation
                """)

                with gr.Row():
                    with gr.Column():
                        # Default image selector
                        if DEFAULT_IMAGES:
                            point_default_selector = gr.Dropdown(
                                choices=["None"] + list(DEFAULT_IMAGES.keys()),
                                value="None",
                                label="Select Default Multi-View Image"
                            )

                        point_image = gr.Image(label="Upload Image or Use Default", type="pil")
                        point_type = gr.Radio(
                            choices=["Positive (include)", "Negative (exclude)"],
                            value="Positive (include)",
                            label="Point Type"
                        )
                        point_display = gr.Image(label="Click to add points", interactive=False)

                        with gr.Row():
                            clear_points_btn = gr.Button("Clear Points", variant="secondary")
                            segment_points_btn = gr.Button("Generate Mask from Points", variant="primary")

                    with gr.Column():
                        point_result = gr.Image(label="Segmentation Result")
                        point_binary_mask = gr.Image(label="Binary Mask (White=Object, Black=Background)")
                        point_status = gr.Textbox(label="Status")

                # Load default image when selected
                if DEFAULT_IMAGES:
                    point_default_selector.change(
                        fn=select_default_image,
                        inputs=[point_default_selector],
                        outputs=[point_image]
                    )

                # Handle point clicks
                point_image.select(
                    fn=handle_point_click,
                    inputs=[point_image, point_type],
                    outputs=[point_display, point_status]
                )

                clear_points_btn.click(
                    fn=clear_points,
                    inputs=[],
                    outputs=[point_display, point_status]
                )

                segment_points_btn.click(
                    fn=segment_with_points,
                    inputs=[point_image],
                    outputs=[point_result, point_binary_mask, gr.State(), point_status]
                )

            with gr.Tab("📦 Box-based Segmentation"):
                gr.Markdown("""
                **Instructions:**
                1. Upload an image or select a default multi-view image
                2. Click twice on the image to draw a bounding box (first click = top-left, second click = bottom-right)
                3. Click "Generate Mask" to create the segmentation
                """)

                with gr.Row():
                    with gr.Column():
                        # Default image selector
                        if DEFAULT_IMAGES:
                            box_default_selector = gr.Dropdown(
                                choices=["None"] + list(DEFAULT_IMAGES.keys()),
                                value="None",
                                label="Select Default Multi-View Image"
                            )

                        box_image = gr.Image(label="Upload Image or Use Default", type="pil")
                        box_display = gr.Image(label="Click twice to draw box", interactive=False)
                    
                        with gr.Row():
                            clear_box_btn = gr.Button("Clear Box", variant="secondary")
                            segment_box_btn = gr.Button("Generate Mask from Box", variant="primary")
                
                    with gr.Column():
                        box_result = gr.Image(label="Segmentation Result")
                        box_binary_mask = gr.Image(label="Binary Mask (White=Object, Black=Background)")
                        box_status = gr.Textbox(label="Status")
            
                # Load default image when selected
                if DEFAULT_IMAGES:
                    box_default_selector.change(
                        fn=select_default_image,
                        inputs=[box_default_selector],
                        outputs=[box_image]
                    )

                # Handle box selection
                box_image.select(
                    fn=handle_box_select,
                    inputs=[box_image, box_state],
                    outputs=[box_display, box_status, box_state]
                )

                clear_box_btn.click(
                    fn=clear_box,
                    inputs=[],
                    outputs=[box_display, box_status, box_state]
                )

                segment_box_btn.click(
                    fn=segment_with_box,
                    inputs=[box_image],
                    outputs=[box_result, box_binary_mask, gr.State(), box_status]
                )
        
            with gr.Tab("🔍 Segment Everything"):
                gr.Markdown("**Automatically find and segment all objects**")

                with gr.Row():
                    with gr.Column():
                        # Default image selector
                        if DEFAULT_IMAGES:
                            auto_default_selector = gr.Dropdown(
                                choices=["None"] + list(DEFAULT_IMAGES.keys()),
                                value="None",
                                label="Select Default Multi-View Image"
                            )

                        auto_image = gr.Image(label="Upload Image or Use Default", type="pil")
                        segment_all_btn = gr.Button("Segment All Objects", variant="primary")

                    with gr.Column():
                        auto_result = gr.Image(label="All Objects Segmented")
                        auto_binary_mask = gr.Image(label="Combined Binary Mask")
                        auto_status = gr.Textbox(label="Status")

                # Load default image when selected
                if DEFAULT_IMAGES:
                    auto_default_selector.change(
                        fn=select_default_image,
                        inputs=[auto_default_selector],
                        outputs=[auto_image]
                    )

                segment_all_btn.click(
                    fn=segment_everything,
                    inputs=[auto_image],
                    outputs=[auto_result, auto_binary_mask, auto_status]
                )

            with gr.Tab("🎨 Auto-Segment Individual Objects"):
                gr.Markdown("""
                **Automatically segment all objects and show individual binary masks**

                **Instructions:**
                1. Click "Load All 4 Views" to load all multi-view images at once
                2. Choose how many objects to detect per view (max 20)
                3. Click "Auto-Segment All Views" to generate individual masks for each object in all 4 views
                4. Each detected object from all views gets its own binary mask displayed in the gallery
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        max_objects_slider = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Maximum Number of Objects to Detect (per view)"
                        )

                        with gr.Row():
                            if DEFAULT_IMAGES:
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

                # Load all views at once
                if DEFAULT_IMAGES:
                    load_all_views_btn.click(
                        fn=load_all_views,
                        inputs=[],
                        outputs=[view1_image, view2_image, view3_image, view4_image]
                    )

                # Segment all views
                auto_segment_all_btn.click(
                    fn=segment_all_views,
                    inputs=[view1_image, view2_image, view3_image, view4_image, max_objects_slider],
                    outputs=[view1_result, view2_result, view3_result, view4_result, individual_masks_gallery, auto_seg_status]
                )
        
            with gr.Tab("✏️ Brush Masking with SAM"):
                gr.Markdown("""
                **Draw on images + SAM automatic segmentation to find and merge overlapping segments**

                **Instructions:**
                1. Enter a folder name to save your masked data (or leave empty for auto-generated name)
                2. Click "Load All 4 Views" to load all multi-view images
                3. Draw on each view to roughly mark the area of interest
                4. Click "Process All Views with SAM" - SAM will automatically segment and merge all segments that overlap with your brushed areas
                5. Get refined segmentation masks saved in a folder structure ready for training
                6. Use the saved folder path in the "DreamEdit3D Training" tab
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        brush_save_folder = gr.Textbox(
                            label="Save Folder Name",
                            placeholder="e.g., my_masked_data (leave empty for auto-generated name)",
                            info="Folder will be created in mask/ directory"
                        )
                        if DEFAULT_IMAGES:
                            load_brush_views_btn = gr.Button("Load All 4 Views", variant="secondary")
                        process_brush_btn = gr.Button("Process All Views with SAM", variant="primary")

                gr.Markdown("### Draw on Images (Brush Areas of Interest)")
                with gr.Row():
                    view1_brush_input = gr.ImageMask(
                        label="View 1 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"]),
                        sources=["upload", "clipboard"]
                    )
                    view2_brush_input = gr.ImageMask(
                        label="View 2 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"]),
                        sources=["upload", "clipboard"]
                    )
                    view3_brush_input = gr.ImageMask(
                        label="View 3 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"]),
                        sources=["upload", "clipboard"]
                    )
                    view4_brush_input = gr.ImageMask(
                        label="View 4 - Draw Mask",
                        brush=gr.Brush(default_size=20, colors=["#00FF00"]),
                        sources=["upload", "clipboard"]
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

                # Load default images for brushing
                if DEFAULT_IMAGES:
                    def load_all_views_for_brush():
                        views = []
                        for view_name in sorted(DEFAULT_IMAGES.keys()):
                            img = select_default_image(view_name)
                            if img:
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
                        inputs=[],
                        outputs=[view1_brush_input, view2_brush_input, view3_brush_input, view4_brush_input]
                    )

                # Process all views with brush + SAM
                process_brush_btn.click(
                    fn=process_all_views_with_brush,
                    inputs=[view1_brush_input, view2_brush_input, view3_brush_input, view4_brush_input, brush_save_folder],
                    outputs=[view1_brush_result, view2_brush_result, view3_brush_result, view4_brush_result,
                            view1_brush_mask, view2_brush_mask, view3_brush_mask, view4_brush_mask, brush_status, brush_saved_path]
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

            def load_masked_folder(folder_path, view_index=1):
                """Load image and masks from a saved folder structure"""
                if not folder_path or not os.path.exists(folder_path):
                    return None, [], "Folder not found"

                folder = Path(folder_path)

                # Check if it's a multi-view structure
                view_dirs = sorted([d for d in folder.iterdir() if d.is_dir() and d.name.startswith('view_')])

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

                # Find mask files
                mask_files = sorted([str(f) for f in view_dir.glob("mask*.png")])

                if main_image and mask_files:
                    return main_image, mask_files, f"Loaded {len(mask_files)} mask(s) from {folder_path}"
                else:
                    return None, [], "No valid image/masks found in folder"

            # Initialize DreamEdit3D app
            d3d_app = DreamEdit3DApp()

            def get_masked_output_folders():
                """Get all folders from masked_output directory"""
                masked_output_dir = Path(__file__).parent / "masked_output"
                if not masked_output_dir.exists():
                    return []
                folders = [f.name for f in masked_output_dir.iterdir() if f.is_dir()]
                return sorted(folders, reverse=True)  # Most recent first

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Step 1: Prepare Your Data")

                    # Load from saved masked folder
                    gr.Markdown("**Option 1: Load from SAM-masked folder**")
                    with gr.Row():
                        d3d_folder_dropdown = gr.Dropdown(
                            label="Select Masked Output Folder",
                            choices=get_masked_output_folders(),
                            interactive=True,
                            scale=2
                        )
                        d3d_refresh_folders_btn = gr.Button("🔄", scale=0, min_width=40)
                        d3d_view_selector = gr.Dropdown(
                            label="View",
                            choices=[1, 2, 3, 4],
                            value=1,
                            scale=1
                        )
                        d3d_load_folder_btn = gr.Button("Load Folder", scale=1)

                    with gr.Row():
                        d3d_folder_path = gr.Textbox(
                            label="Or enter custom path",
                            placeholder="e.g., mask/my_custom_data",
                            scale=3
                        )

                    d3d_load_status = gr.Textbox(label="Load Status", max_lines=2)

                    # Example selector
                    gr.Markdown("**Option 2: Load from examples**")
                    if d3d_app.examples:
                        with gr.Row():
                            d3d_example_dropdown = gr.Dropdown(
                                label="Load Example (optional)",
                                choices=list(d3d_app.examples.keys()),
                                value=None
                            )
                            d3d_load_example_btn = gr.Button("Load Example")

                    gr.Markdown("**Option 3: Manual upload**")

                    d3d_main_image = gr.Image(
                        label="Main Image",
                        type="filepath",
                        height=300
                    )

                    d3d_concept_names = gr.Textbox(
                        label="Concept Names (comma-separated)",
                        placeholder="e.g., creature, bowl, stone",
                        info="One name per mask"
                    )

                    d3d_mask_files = gr.File(
                        label="Mask Files",
                        file_count="multiple",
                        file_types=["image"],
                        height=200
                    )

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

            d3d_train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

            with gr.Row():
                d3d_training_status = gr.Textbox(label="Training Status", max_lines=5)
                d3d_training_output = gr.Textbox(label="Training Output", max_lines=10)

            d3d_trained_model_path = gr.Textbox(label="Trained Model Path", visible=False)

            # Connect folder loading
            def load_from_folder_or_state(folder_dropdown, folder_path, view_index, saved_path_state):
                """Load from folder path or use saved path from brush tab"""
                # Priority: dropdown > custom path > saved state
                if folder_dropdown:
                    # Construct full path from dropdown selection
                    masked_output_dir = Path(__file__).parent / "masked_output"
                    path_to_use = str(masked_output_dir / folder_dropdown)
                elif folder_path:
                    path_to_use = folder_path
                else:
                    path_to_use = saved_path_state
                return load_masked_folder(path_to_use, view_index)

            d3d_load_folder_btn.click(
                fn=load_from_folder_or_state,
                inputs=[d3d_folder_dropdown, d3d_folder_path, d3d_view_selector, brush_saved_path],
                outputs=[d3d_main_image, d3d_mask_files, d3d_load_status]
            )

            # Refresh folders button
            d3d_refresh_folders_btn.click(
                fn=lambda: gr.Dropdown(choices=get_masked_output_folders()),
                outputs=[d3d_folder_dropdown]
            )

            # Connect example loading
            if d3d_app.examples:
                d3d_load_example_btn.click(
                    fn=d3d_app.load_example,
                    inputs=[d3d_example_dropdown],
                    outputs=[d3d_main_image, d3d_mask_files, d3d_concept_names, gr.State()]
                )

            # Connect training
            d3d_train_btn.click(
                fn=d3d_app.train_model,
                inputs=[d3d_main_image, d3d_mask_files, d3d_concept_names, d3d_phase1_steps, d3d_phase2_steps],
                outputs=[d3d_training_status, d3d_training_output, d3d_trained_model_path]
            )

            gr.Markdown("### Step 3: Generate Images")
            with gr.Row():
                with gr.Column():
                    d3d_model_dropdown = gr.Dropdown(
                        label="Select Trained Model",
                        choices=d3d_app.get_available_models(),
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

                    d3d_generate_btn = gr.Button("🎨 Generate Image", variant="primary")

                with gr.Column():
                    d3d_generated_image = gr.Image(label="Generated Multi-View Image", height=400)
                    d3d_generation_status = gr.Textbox(label="Generation Status")

            d3d_refresh_btn.click(
                fn=d3d_app.refresh_models,
                outputs=[d3d_model_dropdown]
            )

            d3d_generate_btn.click(
                fn=d3d_app.generate_image,
                inputs=[d3d_model_dropdown, d3d_prompt_input, d3d_output_filename],
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
                        # Add folder selector for multi-view folders
                        gtr_input_folders = get_gtr_input_folders()
                        if gtr_input_folders:
                            with gr.Group():
                                gr.Markdown("**📁 Multi-view Folders**")
                                with gr.Row():
                                    gtr_folder_dropdown = gr.Dropdown(
                                        choices=gtr_input_folders,
                                        label="Select Folder",
                                        value=None,
                                        info="Folders with view_* subdirectories",
                                        container=False
                                    )
                                    gtr_refresh_folders_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

                        # Add generated multiview image selector
                        gtr_generated_images = get_generated_multiview_images()
                        if gtr_generated_images:
                            with gr.Group():
                                gr.Markdown("**🎨 Generated Multi-view Images**")
                                with gr.Row():
                                    gtr_generated_dropdown = gr.Dropdown(
                                        choices=gtr_generated_images,
                                        label="Select Generated Image",
                                        value=None,
                                        info="From DreamEdit3D training",
                                        container=False
                                    )
                                    gtr_refresh_generated_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

                        with gr.Group():
                            gr.Markdown("**📤 Upload or Select Example**")
                            gtr_input_image = gr.File(label="Upload Image", file_types=["image"])

                            # Add example image selector
                            gtr_example_images = get_gtr_example_images()
                            if gtr_example_images:
                                gtr_example_dropdown = gr.Dropdown(
                                    choices=gtr_example_images,
                                    label="Example Images",
                                    value=None,
                                    container=False
                                )

                        with gr.Group():
                            gr.Markdown("**⚙️ Configuration**")
                            gtr_checkpoint = gr.Textbox(
                                value=GTR_CKPT_PATH,
                                label="Checkpoint Path",
                                info="Path to GTR model checkpoint"
                            )

                        gtr_run_btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 Results")
                        with gr.Group():
                            gtr_output_mesh = gr.File(label="💎 3D Mesh (.obj)")
                            gtr_output_mesh_gif = gr.Image(label="🎬 Mesh Rendering", height=300)
                            gtr_output_nerf_gif = gr.Image(label="🌟 NeRF Rendering", height=300)

                with gr.Accordion("📝 Execution Logs", open=False):
                    gtr_logs = gr.Textbox(label="Logs", lines=10, show_label=False)

                # Refresh folders functionality
                if gtr_input_folders:
                    gtr_refresh_folders_btn.click(
                        fn=lambda: gr.Dropdown(choices=get_gtr_input_folders()),
                        outputs=[gtr_folder_dropdown]
                    )

                # Refresh generated images functionality
                if gtr_generated_images:
                    gtr_refresh_generated_btn.click(
                        fn=lambda: gr.Dropdown(choices=get_generated_multiview_images()),
                        outputs=[gtr_generated_dropdown]
                    )

                # Run pipeline - need to handle generated image selection
                def run_with_generated_image(uploaded_file, checkpoint, example, folder, generated_image):
                    """Wrapper to handle generated image selection"""
                    # Priority: folder > generated_image > example > uploaded
                    if generated_image and not folder:
                        # Use generated image as the uploaded file
                        return full_gtr_pipeline(generated_image, checkpoint, example, folder)
                    return full_gtr_pipeline(uploaded_file, checkpoint, example, folder)

                gtr_run_inputs = [gtr_input_image, gtr_checkpoint]
                if gtr_example_images:
                    gtr_run_inputs.append(gtr_example_dropdown)
                else:
                    gtr_run_inputs.append(gr.State(None))
                if gtr_input_folders:
                    gtr_run_inputs.append(gtr_folder_dropdown)
                else:
                    gtr_run_inputs.append(gr.State(None))
                if gtr_generated_images:
                    gtr_run_inputs.append(gtr_generated_dropdown)
                else:
                    gtr_run_inputs.append(gr.State(None))

                gtr_run_btn.click(
                    run_with_generated_image,
                    inputs=gtr_run_inputs,
                    outputs=[gtr_output_mesh, gtr_output_mesh_gif, gtr_output_nerf_gif, gtr_logs]
                )

            with gr.Tab("🔧 Step-by-Step"):
                gr.Markdown("""
                ### 🎯 Step 1: Prepare Multi-view
                Convert your input into the GTR-compatible format.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        # Add folder selector for step-by-step
                        if gtr_input_folders:
                            with gr.Group():
                                gr.Markdown("**📁 Multi-view Folders**")
                                with gr.Row():
                                    gtr_prep_folder_dropdown = gr.Dropdown(
                                        choices=gtr_input_folders,
                                        label="Select Folder",
                                        value=None,
                                        info="Folders with view_* subdirectories",
                                        container=False
                                    )
                                    gtr_prep_refresh_folders_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

                        # Add generated multiview image selector for step-by-step
                        if gtr_generated_images:
                            with gr.Group():
                                gr.Markdown("**🎨 Generated Multi-view Images**")
                                with gr.Row():
                                    gtr_prep_generated_dropdown = gr.Dropdown(
                                        choices=gtr_generated_images,
                                        label="Select Generated Image",
                                        value=None,
                                        info="From DreamEdit3D training",
                                        container=False
                                    )
                                    gtr_prep_refresh_generated_btn = gr.Button("🔄", size="sm", scale=0, min_width=50)

                        with gr.Group():
                            gr.Markdown("**📤 Upload or Select Example**")
                            gtr_prep_input = gr.File(label="Upload Image", file_types=["image"])

                            # Add example selector for step-by-step
                            if gtr_example_images:
                                gtr_prep_example_dropdown = gr.Dropdown(
                                    choices=gtr_example_images,
                                    label="Example Images",
                                    value=None,
                                    container=False
                                )

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

                # Refresh folders for step-by-step
                if gtr_input_folders:
                    gtr_prep_refresh_folders_btn.click(
                        fn=lambda: gr.Dropdown(choices=get_gtr_input_folders()),
                        outputs=[gtr_prep_folder_dropdown]
                    )

                # Refresh generated images for step-by-step
                if gtr_generated_images:
                    gtr_prep_refresh_generated_btn.click(
                        fn=lambda: gr.Dropdown(choices=get_generated_multiview_images()),
                        outputs=[gtr_prep_generated_dropdown]
                    )

                # Prepare with all input options including generated images
                def prep_with_generated_image(uploaded_file, example, folder, generated_image):
                    """Wrapper to handle generated image selection for preparation"""
                    # Priority: folder > generated_image > example > uploaded
                    if generated_image and not folder:
                        return prepare_multiview_gtr(generated_image, example, folder)
                    return prepare_multiview_gtr(uploaded_file, example, folder)

                gtr_prep_inputs = [gtr_prep_input]
                if gtr_example_images:
                    gtr_prep_inputs.append(gtr_prep_example_dropdown)
                else:
                    gtr_prep_inputs.append(gr.State(None))
                if gtr_input_folders:
                    gtr_prep_inputs.append(gtr_prep_folder_dropdown)
                else:
                    gtr_prep_inputs.append(gr.State(None))
                if gtr_generated_images:
                    gtr_prep_inputs.append(gtr_prep_generated_dropdown)
                else:
                    gtr_prep_inputs.append(gr.State(None))

                gtr_prep_btn.click(
                    prep_with_generated_image,
                    inputs=gtr_prep_inputs,
                    outputs=[gtr_prep_output_dir, gtr_prep_logs]
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
                            gtr_prepared_dirs = get_gtr_prepared_directories()
                            if gtr_prepared_dirs:
                                gtr_prepared_dir_dropdown = gr.Dropdown(
                                    choices=gtr_prepared_dirs,
                                    label="Or Select Prepared Example",
                                    value=None,
                                    container=False
                                )

                                def load_gtr_prepared_dir(dir_path):
                                    if dir_path:
                                        return dir_path
                                    return None

                                gtr_prepared_dir_dropdown.change(
                                    load_gtr_prepared_dir,
                                    inputs=[gtr_prepared_dir_dropdown],
                                    outputs=[gtr_inf_prepared_dir]
                                )

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
                            gtr_inf_mesh_gif = gr.Image(label="🎬 Mesh Rendering", height=300)
                            gtr_inf_nerf_gif = gr.Image(label="🌟 NeRF Rendering", height=300)

                with gr.Accordion("📝 Inference Logs", open=False):
                    gtr_inf_logs = gr.Textbox(label="Logs", lines=5, show_label=False)

                gtr_inf_btn.click(
                    run_gtr_inference,
                    inputs=[gtr_inf_prepared_dir, gtr_inf_checkpoint],
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

                ## 📝 Supported Input Formats

                | Format | Description | Example |
                |--------|-------------|---------|
                | **Zero123++ Grid** | 3×2 grid (6 views) | Multi-view image grid |
                | **Horizontal Grid** | 1×4 strip (4 views) | Side-by-side views |
                | **Folder Structure** | Directories with `view_*` subdirs | Generated from DreamEdit3D |
                | **Generated Images** | Output from training tab | `generated_multiview/*.jpg` |

                ## 💡 Tips
                - Use **folders** for masked multi-view data
                - Use **generated images** for training outputs
                - **Refresh buttons** (🔄) update available options
                - Check **logs** for detailed execution info
                """)

        with gr.Tab("📚 Setup Instructions"):
            gr.Markdown("""
            ## Setup Instructions:
            
            ### 1. Install Dependencies:
            ```bash
            pip install segment-anything torch torchvision
            pip install opencv-python pillow numpy
            ```
            
            ### 2. Download SAM Checkpoints:
            ```bash
            # Choose one based on your needs:
            
            # Highest quality (recommended):
            wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
            
            # Medium quality:
            wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
            
            # Fastest:
            wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
            ```
            
            ### 3. Usage Tips:
            - **Point Segmentation**: 
              - Green points = include in mask
              - Red points = exclude from mask
              - Click multiple points for better accuracy
            - **Box Segmentation**: 
              - Click twice: first for top-left corner, second for bottom-right corner
              - Draw tight boxes around objects for best results
            - **Segment Everything**: Good for exploring what SAM can detect
            - **Binary Masks**: White pixels (255) = masked object, Black pixels (0) = background
            
            ### 4. Model Sizes:
            - **vit_h**: Largest, most accurate, slower (~2.4GB)
            - **vit_l**: Medium size and accuracy (~1.2GB)  
            - **vit_b**: Fastest, good for real-time use (~375MB)
            
            ### 5. Output Format:
            All masks are returned as binary images where:
            - **White (255)** = Selected/masked area
            - **Black (0)** = Background/unselected area
            """)
    
    return demo

if __name__ == "__main__":
    print("🎯 Starting SAM Image Masking Interface...")
    if not SAM_AVAILABLE:
        print("⚠️  Warning: Segment Anything not available. Please install:")
        print("   pip install segment-anything")

    demo = create_interface()

    # Add allowed paths for Gradio to access files
    parent_dir = Path(__file__).parent.parent
    allowed_paths = [
        str(parent_dir / "temp_workspace"),
        str(parent_dir / "trained_models"),
        str(parent_dir / "generated_multiview"),
        str(Path(__file__).parent / "masked_output"),
        str(Path(__file__).parent),
        str(parent_dir / "temp_gtr_gradio"),
        str(parent_dir / "examples"),
        str(parent_dir / "ckpts")
    ]

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=allowed_paths,
        inbrowser=True,
        max_threads=10
    )