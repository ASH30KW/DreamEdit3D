import os
import sys
import random
import argparse
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if type(prompt) != list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size, 1, 1)}
        uc_ = {"context": uc.repeat(batch_size, 1, 1)}
        if camera is not None:
            c_["camera"] = uc_["camera"] = camera
            c_["num_frames"] = uc_["num_frames"] = num_frames

        shape = [4, image_size // 8, image_size // 8]
        samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
                                        batch_size=batch_size, shape=shape,
                                        verbose=False, 
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc_,
                                        eta=ddim_eta, x_T=None)
        x_sample = model.decode_first_stage(samples_ddim)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
        x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


class SimplifiedBreakSceneMVDream:
    def __init__(self):
        self._parse_args()
        self._load_break_scene_pipeline()
        self._load_mvdream_model()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        
        # Break-A-Scene arguments
        parser.add_argument("--break_scene_model_path", type=str, required=True, 
                           help="Path to the trained Break-A-Scene model")
        parser.add_argument("--asset_prompt", type=str, default="a photo of <asset0>", 
                           help="Prompt containing the asset token")
        parser.add_argument("--description_prompt", type=str, required=True,
                           help="Descriptive prompt to use with MVDream (e.g., 'a red car')")
        
        # MVDream arguments
        parser.add_argument("--mvdream_model_name", type=str, default="sd-v2.1-base-4view", 
                           help="MVDream model name")
        parser.add_argument("--mvdream_config_path", type=str, default=None, 
                           help="MVDream config path (optional)")
        parser.add_argument("--mvdream_ckpt_path", type=str, default=None, 
                           help="MVDream checkpoint path (optional)")
        parser.add_argument("--suffix", type=str, default=", 3d asset", 
                           help="Suffix to add to the prompt for MVDream")
        
        # Common arguments
        parser.add_argument("--size", type=int, default=256, help="Image size")
        parser.add_argument("--num_frames", type=int, default=4, help="Number of views to generate")
        parser.add_argument("--camera_elev", type=int, default=15, help="Camera elevation")
        parser.add_argument("--camera_azim", type=int, default=90, help="Camera azimuth start")
        parser.add_argument("--camera_azim_span", type=int, default=360, help="Camera azimuth span")
        parser.add_argument("--seed", type=int, default=23, help="Random seed")
        parser.add_argument("--fp16", action="store_true", help="Use half precision")
        parser.add_argument("--device", type=str, default="cuda", help="Device to use")
        parser.add_argument("--output_path", type=str, default="outputs/", 
                           help="Output directory")
        
        self.args = parser.parse_args()

    def _load_break_scene_pipeline(self):
        """Load the Break-A-Scene pipeline with the trained asset token"""
        print("Loading Break-A-Scene pipeline...")
        self.break_scene_pipeline = DiffusionPipeline.from_pretrained(
            self.args.break_scene_model_path,
            torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
        )
        self.break_scene_pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.break_scene_pipeline.to(self.args.device)
        print("Break-A-Scene pipeline loaded.")

    def _load_mvdream_model(self):
        """Load the MVDream model"""
        print("Loading MVDream model...")
        
        dtype = torch.float16 if self.args.fp16 else torch.float32
        
        if self.args.mvdream_config_path is None:
            self.mvdream_model = build_model(self.args.mvdream_model_name, 
                                           ckpt_path=self.args.mvdream_ckpt_path)
        else:
            assert self.args.mvdream_ckpt_path is not None, "ckpt_path must be specified!"
            config = OmegaConf.load(self.args.mvdream_config_path)
            self.mvdream_model = instantiate_from_config(config.model)
            self.mvdream_model.load_state_dict(torch.load(self.args.mvdream_ckpt_path, map_location='cpu'))
        
        self.mvdream_model.device = self.args.device
        self.mvdream_model.to(self.args.device)
        self.mvdream_model.eval()

        self.sampler = DDIMSampler(self.mvdream_model)
        self.uc = self.mvdream_model.get_learned_conditioning([""]).to(self.args.device)
        
        print("MVDream model loaded.")

    def _setup_camera(self):
        """Setup camera matrices for multi-view generation"""
        batch_size = max(4, self.args.num_frames)
        camera = get_camera(self.args.num_frames, 
                           elevation=self.args.camera_elev,
                           azimuth_start=self.args.camera_azim, 
                           azimuth_span=self.args.camera_azim_span)
        camera = camera.repeat(batch_size // self.args.num_frames, 1).to(self.args.device)
        return camera, batch_size

    @torch.no_grad()
    def generate_reference_image(self):
        """Generate a reference image using Break-A-Scene"""
        print("Generating reference image with Break-A-Scene...")
        
        set_seed(self.args.seed)
        
        # Generate image with the asset token
        reference_image = self.break_scene_pipeline(
            prompt=self.args.asset_prompt,
            num_inference_steps=50,
            guidance_scale=7.5,
            height=self.args.size,
            width=self.args.size
        ).images[0]
        
        # Save reference image
        os.makedirs(self.args.output_path, exist_ok=True)
        ref_path = os.path.join(self.args.output_path, "reference_asset.png")
        reference_image.save(ref_path)
        print(f"Reference image saved to: {ref_path}")
        
        return reference_image

    def _extract_asset_embedding(self):
        """Extract the learned embedding for <asset0> from Break-A-Scene text encoder"""
        print("Extracting <asset0> embedding from Break-A-Scene...")
        
        # Get the text encoder from Break-A-Scene pipeline
        text_encoder = self.break_scene_pipeline.text_encoder
        tokenizer = self.break_scene_pipeline.tokenizer
        
        # Find the asset token in the tokenizer vocabulary
        asset_token = "<asset0>"
        if asset_token not in tokenizer.get_vocab():
            raise ValueError(f"Asset token '{asset_token}' not found in Break-A-Scene tokenizer vocabulary")
        
        # Get the token ID and extract its embedding
        token_id = tokenizer.convert_tokens_to_ids(asset_token)
        learned_embedding = text_encoder.get_input_embeddings().weight.data[token_id].clone()
        
        print(f"Extracted embedding for '{asset_token}' (token_id: {token_id})")
        return asset_token, learned_embedding

    def _transfer_embedding_to_mvdream(self, asset_token, learned_embedding):
        """Transfer the learned embedding to MVDream's text encoder"""
        print("Transferring embedding to MVDream...")
        
        # Get MVDream's text encoder
        text_encoder = self.mvdream_model.cond_stage_model
        
        # Handle different encoder types (CLIP vs OpenCLIP)
        if hasattr(text_encoder, 'tokenizer'):
            # CLIP-style tokenizer (FrozenCLIPEmbedder)
            tokenizer = text_encoder.tokenizer
            if asset_token not in tokenizer.get_vocab():
                tokenizer.add_tokens([asset_token])
            token_id = tokenizer.convert_tokens_to_ids(asset_token)
            
            # Resize embeddings if necessary
            embedding_layer = text_encoder.transformer.embeddings.token_embeddings
            if token_id >= embedding_layer.num_embeddings:
                # Resize embedding layer
                old_embeddings = embedding_layer.weight.data
                new_embedding_layer = torch.nn.Embedding(token_id + 1, embedding_layer.embedding_dim)
                new_embedding_layer.weight.data[:old_embeddings.shape[0]] = old_embeddings
                text_encoder.transformer.embeddings.token_embeddings = new_embedding_layer
                embedding_layer = new_embedding_layer
            
            # Copy the learned embedding
            embedding_layer.weight.data[token_id] = learned_embedding.to(embedding_layer.weight.device)
            
        else:
            # OpenCLIP-style (FrozenOpenCLIPEmbedder)
            # For OpenCLIP, we need to be more careful as it has a fixed vocab
            # We'll use a workaround by replacing a less common token's embedding
            import open_clip
            
            # Find a suitable token to replace (using a placeholder approach)
            test_tokens = ["!", "?", "#", "@", "%", "^", "&", "*"]
            chosen_token = None
            chosen_token_id = None
            
            for test_token in test_tokens:
                tokens = open_clip.tokenize([test_token])
                if len(tokens[0].nonzero()) > 0:
                    chosen_token_id = tokens[0][tokens[0].nonzero()[0]].item()
                    chosen_token = test_token
                    break
            
            if chosen_token_id is None:
                # Fallback to a default token ID
                chosen_token_id = 999  # Should be valid in most OpenCLIP vocabs
                chosen_token = "!"
            
            print(f"Using token '{chosen_token}' (ID: {chosen_token_id}) as placeholder for '{asset_token}' in OpenCLIP")
            
            # Copy the learned embedding to the chosen token position
            embedding_layer = text_encoder.model.token_embedding
            embedding_layer.weight.data[chosen_token_id] = learned_embedding.to(embedding_layer.weight.device)
            
            # Return the token we should use in prompts
            asset_token = chosen_token
        
        print(f"Successfully transferred embedding to MVDream")
        return asset_token

    @torch.no_grad()
    def generate_4view_images(self):
        """Generate 4-view images using transferred <asset0> embedding"""
        print("Generating 4-view images with MVDream using transferred embedding...")
        
        # Extract and transfer the learned embedding
        asset_token, learned_embedding = self._extract_asset_embedding()
        transferred_token = self._transfer_embedding_to_mvdream(asset_token, learned_embedding)
        
        # Setup camera and batch size
        camera, batch_size = self._setup_camera()
        
        # Create prompt with the transferred asset token
        prompt = f"a photo of {transferred_token}, {self.args.description_prompt}" + self.args.suffix
        print(f"Using MVDream prompt: {prompt}")
        
        # Set seed for reproducibility
        set_seed(self.args.seed)
        
        # Generate images
        dtype = torch.float16 if self.args.fp16 else torch.float32
        
        images = []
        for j in range(3):  # Generate 3 sets as in original code
            img = t2i(self.mvdream_model, self.args.size, prompt, self.uc, self.sampler,
                     step=50, scale=10, batch_size=batch_size, ddim_eta=0.0,
                     dtype=dtype, device=self.args.device, camera=camera, 
                     num_frames=self.args.num_frames)
            img = np.concatenate(img, 1)
            images.append(img)
        
        # Combine all images
        final_image = np.concatenate(images, 0)
        
        # Save the result
        output_path = os.path.join(self.args.output_path, "asset_4views.png")
        Image.fromarray(final_image).save(output_path)
        print(f"4-view images saved to: {output_path}")
        
        return final_image

    def run_pipeline(self):
        """Run the complete pipeline"""
        print("Starting Break-A-Scene + MVDream pipeline...")
        
        # Step 1: Generate reference image with Break-A-Scene
        reference_image = self.generate_reference_image()
        
        # Step 2: Generate 4-view images with MVDream
        multiview_images = self.generate_4view_images()
        
        print("Pipeline completed successfully!")
        return reference_image, multiview_images


if __name__ == "__main__":
    pipeline = SimplifiedBreakSceneMVDream()
    pipeline.run_pipeline()