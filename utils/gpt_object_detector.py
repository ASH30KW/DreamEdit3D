"""
GPT-4V based object detection for automatic concept naming
"""
import os
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai package not installed. Install with: pip install openai")


def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 for GPT-4V API"""
    import io
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def apply_mask_to_image(image_path: str, mask_path: str) -> Image.Image:
    """
    Apply mask to image to isolate the object.

    Args:
        image_path: Path to original image
        mask_path: Path to mask image

    Returns:
        PIL Image with mask applied (transparent background)
    """
    img = Image.open(image_path).convert('RGBA')
    mask = Image.open(mask_path).convert('L')

    # Resize mask to match image if needed
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

    # Create RGBA image with transparent background
    img_array = np.array(img)
    mask_array = np.array(mask)

    # Apply mask to alpha channel
    img_array[:, :, 3] = mask_array

    # Create white background version for better GPT-4V detection
    white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    masked_img = Image.alpha_composite(white_bg, Image.fromarray(img_array))

    return masked_img.convert('RGB')


def detect_object_name_gpt4v(
    image_path: str,
    mask_path: str,
    api_key: Optional[str] = None,
    fallback_name: str = "object"
) -> str:
    """
    Detect object name from masked image using GPT-4V.

    Args:
        image_path: Path to original image
        mask_path: Path to mask image
        api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        fallback_name: Name to use if detection fails

    Returns:
        Detected object name (e.g., "tomato", "human", "car")
    """
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        print(f"  ⚠ OpenAI not available, using fallback: {fallback_name}")
        return fallback_name

    # Get API key
    api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print(f"  ⚠ No OpenAI API key found, using fallback: {fallback_name}")
        return fallback_name

    try:
        # Apply mask to isolate object
        masked_img = apply_mask_to_image(image_path, mask_path)

        # Encode image
        base64_image = encode_image_base64(masked_img)

        # Create OpenAI client
        client = OpenAI(api_key=api_key)

        # Call GPT-4V
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Look at this masked object image. What is the main object shown?

Reply with ONLY a single word or very short phrase (1-3 words) that describes the object.
Use simple, common words suitable for text-to-3D prompts.

Examples of good answers:
- tomato
- human
- car
- panda
- bowl
- creature
- character

Your answer (one word only):"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "low"  # Use low detail for faster/cheaper processing
                            }
                        }
                    ]
                }
            ],
            max_tokens=20,
            temperature=0.3  # Lower temperature for more consistent naming
        )

        # Extract detected name
        detected = response.choices[0].message.content.strip()
        detected = detected.strip('"\'.,!?').lower()  # Clean up response

        # Validate it's not too long or weird
        if len(detected.split()) > 3 or len(detected) > 30:
            print(f"  ⚠ GPT-4V returned unusual name: '{detected}', using fallback")
            return fallback_name

        print(f"  ✓ GPT-4V detected: '{detected}'")
        return detected

    except Exception as e:
        print(f"  ⚠ GPT-4V detection failed ({e}), using fallback: {fallback_name}")
        return fallback_name


def detect_concept_names_batch(
    image_path: str,
    mask_paths: list,
    api_key: Optional[str] = None
) -> list:
    """
    Detect concept names for multiple masks using GPT-4V.

    Args:
        image_path: Path to original image
        mask_paths: List of paths to mask images
        api_key: OpenAI API key (optional)

    Returns:
        List of detected concept names
    """
    concept_names = []

    for i, mask_path in enumerate(mask_paths):
        print(f"  Detecting object {i+1}/{len(mask_paths)}...")
        name = detect_object_name_gpt4v(
            image_path,
            mask_path,
            api_key=api_key,
            fallback_name=f"object{i}" if i > 0 else "object"
        )
        concept_names.append(name)

    return concept_names
