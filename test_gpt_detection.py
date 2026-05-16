#!/usr/bin/env python3
"""
Quick test script to verify GPT-4V object detection is working.
"""
import os
import sys
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from utils.gpt_object_detector import detect_object_name_gpt4v, detect_concept_names_batch
    print("✓ GPT object detector imported successfully")
except ImportError as e:
    print(f"✗ Failed to import GPT object detector: {e}")
    sys.exit(1)

try:
    from openai import OpenAI
    print("✓ OpenAI package installed")
except ImportError:
    print("✗ OpenAI package not installed")
    print("  Install with: pip install openai")
    sys.exit(1)

# Check API key
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    print("✗ OPENAI_API_KEY environment variable not set")
    print("  Set with: export OPENAI_API_KEY=sk-your-key-here")
    sys.exit(1)
else:
    print(f"✓ OPENAI_API_KEY found (starts with: {api_key[:10]}...)")

# Find a test project
print("\nLooking for test project...")
projects_dir = Path(__file__).parent / "projects"

test_project = None
for project in projects_dir.iterdir():
    if project.is_dir():
        sam_masks = project / "01_sam_masks"
        if sam_masks.exists():
            view_dirs = list(sam_masks.glob("view_*"))
            if view_dirs:
                test_project = project
                break

if not test_project:
    print("✗ No suitable test project found")
    print("  Need a project with 01_sam_masks/view_*/img.jpg and mask files")
    sys.exit(1)

print(f"✓ Found test project: {test_project.name}")

# Find test data
view_dir = list((test_project / "01_sam_masks").glob("view_*"))[0]
img_path = view_dir / "img.jpg"
if not img_path.exists():
    img_path = view_dir / "img.png"

mask_files = sorted([f for f in view_dir.glob("mask*.png")
                     if f.stem.replace("mask", "").isdigit()])

if not img_path.exists() or not mask_files:
    print(f"✗ Test data not found in {view_dir}")
    sys.exit(1)

print(f"  Image: {img_path.name}")
print(f"  Masks: {len(mask_files)} found")

# Test detection
print("\n" + "="*60)
print("Testing GPT-4V object detection...")
print("="*60)

try:
    detected_names = detect_concept_names_batch(
        image_path=str(img_path),
        mask_paths=[str(m) for m in mask_files],
        api_key=api_key
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for i, name in enumerate(detected_names):
        print(f"  Mask {i}: '{name}'")

    print("\n✓ GPT-4V detection is working!")
    print(f"  Auto-detected concept names: {', '.join(detected_names)}")
    print("\nYou can now use this in the DreamEdit3D training interface!")

except Exception as e:
    print(f"\n✗ Detection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
