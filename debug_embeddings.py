"""
Debug script to check if custom tokens are properly loaded
"""

import os
import torch
from diffusers import DiffusionPipeline
from mvdream.model_zoo import build_model

def debug_custom_model(custom_model_path):
    """Debug what's in the custom model directory"""
    print(f"=== Debugging Custom Model: {custom_model_path} ===")
    
    if not os.path.exists(custom_model_path):
        print("ERROR: Custom model path doesn't exist!")
        return
    
    print("Files in custom model directory:")
    for root, dirs, files in os.walk(custom_model_path):
        level = root.replace(custom_model_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{subindent}{file} ({file_size} bytes)")

def test_break_a_scene_pipeline(custom_model_path):
    """Test if Break-A-Scene pipeline can generate with <asset0>"""
    print(f"\n=== Testing Break-A-Scene Pipeline ===")
    
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            custom_model_path,
            torch_dtype=torch.float16,
        )
        pipeline.to("cuda")
        
        print("Pipeline loaded successfully!")
        
        # Test tokenizer
        if hasattr(pipeline, 'tokenizer'):
            tokenizer = pipeline.tokenizer
            print(f"Tokenizer vocab size: {len(tokenizer)}")
            
            # Test tokenizing <asset0>
            test_prompts = [
                "a photo of <asset0>",
                "a photo of <asset0> at the beach", 
                "<asset0>",
            ]
            
            for prompt in test_prompts:
                tokens = tokenizer(prompt)
                token_ids = tokens['input_ids']
                print(f"Prompt: '{prompt}'")
                print(f"Token IDs: {token_ids}")
                
                # Decode back to check
                decoded = tokenizer.decode(token_ids)
                print(f"Decoded: '{decoded}'")
                print()
        
        # Test text encoder
        if hasattr(pipeline, 'text_encoder'):
            text_encoder = pipeline.text_encoder
            print(f"Text encoder: {type(text_encoder)}")
            
            # Get embeddings for <asset0>
            test_input = pipeline.tokenizer(
                "a photo of <asset0>", 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to("cuda")
            
            with torch.no_grad():
                embeddings = text_encoder(**test_input)
                print(f"Text embeddings shape: {embeddings.last_hidden_state.shape}")
        
        # Try generating a test image
        print("Attempting to generate test image...")
        with torch.no_grad():
            result = pipeline(
                "a photo of <asset0>",
                num_inference_steps=20,
                guidance_scale=7.5,
            )
            print(f"Generated {len(result.images)} image(s)")
            result.images[0].save("debug_asset0_test.png")
            print("Test image saved as: debug_asset0_test.png")
        
        return True
        
    except Exception as e:
        print(f"Error testing Break-A-Scene pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_mvdream_tokenizer():
    """Debug MVDream's tokenizer"""
    print(f"\n=== Testing MVDream Tokenizer ===")
    
    try:
        model = build_model("sd-v2.1-base-4view")
        model.to("cuda")
        
        # Test if MVDream can understand <asset0> by default
        test_prompts = [
            "a photo of <asset0>",
            "a photo of a cat",  # baseline
        ]
        
        for prompt in test_prompts:
            print(f"\nTesting prompt: '{prompt}'")
            try:
                conditioning = model.get_learned_conditioning([prompt])
                print(f"Conditioning shape: {conditioning.shape}")
                print(f"Conditioning mean: {conditioning.mean().item():.6f}")
                print(f"Conditioning std: {conditioning.std().item():.6f}")
            except Exception as e:
                print(f"Error getting conditioning: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error testing MVDream: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_model_path", type=str, required=True)
    args = parser.parse_args()
    
    # Debug custom model
    debug_custom_model(args.custom_model_path)
    
    # Test Break-A-Scene pipeline
    breakascene_works = test_break_a_scene_pipeline(args.custom_model_path)
    
    # Test MVDream tokenizer
    mvdream_works = debug_mvdream_tokenizer()
    
    print(f"\n=== Summary ===")
    print(f"Break-A-Scene pipeline works: {breakascene_works}")
    print(f"MVDream base model works: {mvdream_works}")
    
    if breakascene_works and not mvdream_works:
        print("\nSUGGESTION: The issue is that MVDream doesn't know about <asset0>.")
        print("We need to properly transfer the custom embeddings from Break-A-Scene to MVDream.")

if __name__ == "__main__":
    main()