"""
MVDream with Break-A-Scene text encoder replacement
Simple approach: use Break-A-Scene's text encoder directly in MVDream
"""

import argparse
import torch
import numpy as np
from PIL import Image
import os
import random
from diffusers import DiffusionPipeline

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from omegaconf import OmegaConf


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def replace_text_encoder(mvdream_model, break_a_scene_path, device):
    """
    Replace MVDream's text encoder with Break-A-Scene's custom text encoder
    """
    print(f"=== Replacing text encoder with Break-A-Scene ===")
    
    try:
        # Load Break-A-Scene pipeline
        bas_pipeline = DiffusionPipeline.from_pretrained(
            break_a_scene_path,
            torch_dtype=torch.float16,
        )
        bas_pipeline.to(device)
        
        print(f"Break-A-Scene tokenizer vocab size: {len(bas_pipeline.tokenizer)}")
        print(f"Break-A-Scene text encoder: {type(bas_pipeline.text_encoder)}")
        
        # Test that Break-A-Scene understands <asset0>
        test_prompt = "a photo of <asset0>"
        test_tokens = bas_pipeline.tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            test_encoding = bas_pipeline.text_encoder(**test_tokens)
        print(f"Break-A-Scene encoding shape for '<asset0>': {test_encoding.last_hidden_state.shape}")
        
        # Create a custom conditioning function for MVDream
        class CustomConditioningWrapper:
            def __init__(self, bas_text_encoder, bas_tokenizer, device):
                self.text_encoder = bas_text_encoder
                self.tokenizer = bas_tokenizer
                self.device = device
            
            def encode(self, prompts):
                if isinstance(prompts, str):
                    prompts = [prompts]
                
                # Tokenize with Break-A-Scene tokenizer (which knows about <asset0>)
                tokens = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",  # Always pad to max length
                    truncation=True,
                    max_length=77  # Standard CLIP length
                ).to(self.device)
                
                # Encode with Break-A-Scene text encoder
                with torch.no_grad():
                    text_encodings = self.text_encoder(**tokens)
                
                # Ensure we always return 77 tokens (CLIP standard)
                hidden_states = text_encodings.last_hidden_state
                if hidden_states.shape[1] != 77:
                    print(f"Warning: Got {hidden_states.shape[1]} tokens, expected 77. Padding/truncating...")
                    if hidden_states.shape[1] < 77:
                        # Pad with zeros
                        padding = torch.zeros(
                            hidden_states.shape[0], 
                            77 - hidden_states.shape[1], 
                            hidden_states.shape[2],
                            device=hidden_states.device,
                            dtype=hidden_states.dtype
                        )
                        hidden_states = torch.cat([hidden_states, padding], dim=1)
                    else:
                        # Truncate
                        hidden_states = hidden_states[:, :77, :]
                
                return hidden_states
            
            def __call__(self, prompts):
                return self.encode(prompts)
        
        # Create the wrapper
        custom_encoder = CustomConditioningWrapper(
            bas_pipeline.text_encoder, 
            bas_pipeline.tokenizer, 
            device
        )
        
        # Replace MVDream's text conditioning function
        original_get_learned_conditioning = mvdream_model.get_learned_conditioning
        
        def custom_get_learned_conditioning(prompts):
            print(f"Using custom text encoding for: {prompts}")
            return custom_encoder.encode(prompts)
        
        mvdream_model.get_learned_conditioning = custom_get_learned_conditioning
        
        # Store reference to avoid garbage collection
        mvdream_model._custom_text_encoder = custom_encoder
        mvdream_model._bas_pipeline = bas_pipeline
        
        print("Successfully replaced MVDream's text encoder with Break-A-Scene!")
        return True
        
    except Exception as e:
        print(f"Error replacing text encoder: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_multiview_with_custom_encoder(mvdream_model, sampler, prompt, uc, camera, args):
    """
    Generate multi-view images using MVDream with Break-A-Scene text encoder
    """
    print(f"Generating multi-view images for: '{prompt}'")
    
    dtype = torch.float16 if args.fp16 else torch.float32
    batch_size = max(4, args.num_frames)
    device = args.device
    
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        try:
            # Get text conditioning using Break-A-Scene text encoder
            c = mvdream_model.get_learned_conditioning([prompt]).to(device)
            print(f"Text conditioning shape: {c.shape}")
            print(f"Text conditioning mean: {c.mean().item():.6f}")
            
            c_ = {"context": c.repeat(batch_size, 1, 1)}
            uc_ = {"context": uc.repeat(batch_size, 1, 1)}
            
            # Add camera conditioning for multi-view
            if camera is not None:
                c_["camera"] = uc_["camera"] = camera
                c_["num_frames"] = uc_["num_frames"] = args.num_frames
                print(f"Added camera conditioning. Camera shape: {camera.shape}")
                print(f"Number of frames: {args.num_frames}")
            
            # Generate latents
            shape = [4, args.image_size // 8, args.image_size // 8]
            print(f"Sampling with shape: {shape}, steps: {args.num_inference_steps}")
            
            samples_ddim, _ = sampler.sample(
                S=args.num_inference_steps,
                conditioning=c_,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=args.guidance_scale,
                unconditional_conditioning=uc_,
                eta=0.0,
                x_T=None
            )
            
            print("Decoding latents to images...")
            x_sample = mvdream_model.decode_first_stage(samples_ddim)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()
            
            images = [img.astype(np.uint8) for img in x_sample]
            print(f"Generated {len(images)} images")
            
            return images
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    parser = argparse.ArgumentParser()
    
    # Model paths
    parser.add_argument("--custom_model_path", type=str, required=True,
                       help="Path to Break-A-Scene model")
    parser.add_argument("--mvdream_model", type=str, default="sd-v2.1-base-4view")
    parser.add_argument("--mvdream_config", type=str, default=None)
    parser.add_argument("--mvdream_ckpt", type=str, default=None)
    
    # Generation parameters
    parser.add_argument("--prompt", type=str, default="a photo of <asset0> at the beach")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--output_path", type=str, default="outputs/multiview_custom.png")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    # Multi-view settings
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--elevation", type=int, default=15)
    parser.add_argument("--azimuth_start", type=int, default=90)
    parser.add_argument("--azimuth_span", type=int, default=360)
    
    # Sampling settings
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    print("=== Loading MVDream Model ===")
    if args.mvdream_config is None:
        mvdream_model = build_model(args.mvdream_model, ckpt_path=args.mvdream_ckpt)
    else:
        config = OmegaConf.load(args.mvdream_config)
        mvdream_model = instantiate_from_config(config.model)
        if args.mvdream_ckpt:
            mvdream_model.load_state_dict(torch.load(args.mvdream_ckpt, map_location='cpu'))
    
    mvdream_model.device = args.device
    mvdream_model.to(args.device)
    mvdream_model.eval()
    
    print("=== Replacing Text Encoder ===")
    # Replace text encoder with Break-A-Scene's
    success = replace_text_encoder(mvdream_model, args.custom_model_path, args.device)
    
    if not success:
        print("ERROR: Failed to replace text encoder!")
        return
    
    print("=== Setting up Generation ===")
    # Setup sampler and conditioning
    sampler = DDIMSampler(mvdream_model)
    
    # Get unconditional conditioning using the custom encoder
    uc = mvdream_model.get_learned_conditioning([""]).to(args.device)
    print(f"Unconditional conditioning shape: {uc.shape}")
    
    # Setup camera for multi-view
    camera = get_camera(
        args.num_frames,
        elevation=args.elevation,
        azimuth_start=args.azimuth_start,
        azimuth_span=args.azimuth_span
    )
    batch_size = max(4, args.num_frames)
    camera = camera.repeat(batch_size // args.num_frames, 1).to(args.device)
    
    print("=== Generating Multi-View Images ===")
    # Generate images
    full_prompt = args.prompt + args.suffix
    print(f"Full prompt: '{full_prompt}'")
    
    images = generate_multiview_with_custom_encoder(
        mvdream_model, sampler, full_prompt, uc, camera, args
    )
    
    if images is not None:
        print("=== Saving Results ===")
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else ".", exist_ok=True)
        
        # Save individual views
        for i, img in enumerate(images[:args.num_frames]):
            individual_path = args.output_path.replace(".png", f"_view_{i}.png")
            Image.fromarray(img).save(individual_path)
            print(f"Saved view {i} to: {individual_path}")
        
        # Save combined image
        if len(images) >= args.num_frames:
            combined_image = np.concatenate(images[:args.num_frames], axis=1)
            Image.fromarray(combined_image).save(args.output_path)
            print(f"Saved combined multi-view to: {args.output_path}")
        
        # Also save comparison with Break-A-Scene single view
        print("=== Generating Break-A-Scene comparison ===")
        bas_pipeline = mvdream_model._bas_pipeline
        comparison_image = bas_pipeline(full_prompt, num_inference_steps=20).images[0]
        comparison_path = args.output_path.replace(".png", "_breakascene_comparison.png")
        comparison_image.save(comparison_path)
        print(f"Saved Break-A-Scene comparison to: {comparison_path}")
        
        print("\nSUCCESS! 🎉")
        print("Check the outputs:")
        print(f"- Multi-view: {args.output_path}")
        print(f"- Individual views: *_view_*.png")
        print(f"- Break-A-Scene comparison: {comparison_path}")
        
    else:
        print("Generation failed!")


if __name__ == "__main__":
    main()




# python scripts/t2i_test.py     --custom_model_path "/home/ai/gr/DreamEdit3D/outputs/creature"     --prompt "a photo of <asset0> at the beach"     --output_path "outputs/multiview_creature.png"     --num_frames 4     --elevation 15     --azimuth_span 360