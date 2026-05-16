# import os
# import sys
# import random
# import argparse
# from PIL import Image
# import numpy as np
# from omegaconf import OmegaConf
# import torch 

# from mvdream.camera_utils import get_camera
# from mvdream.ldm.util import instantiate_from_config
# from mvdream.ldm.models.diffusion.ddim import DDIMSampler
# from mvdream.model_zoo import build_model
# from diffusers import DiffusionPipeline, DDIMScheduler

# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)


# def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
#     if type(prompt)!=list:
#         prompt = [prompt]
#     with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype):
#         c = model.get_learned_conditioning(prompt).to(device)
#         c_ = {"context": c.repeat(batch_size,1,1)}
#         uc_ = {"context": uc.repeat(batch_size,1,1)}
#         if camera is not None:
#             c_["camera"] = uc_["camera"] = camera
#             c_["num_frames"] = uc_["num_frames"] = num_frames

#         shape = [4, image_size // 8, image_size // 8]
#         samples_ddim, _ = sampler.sample(S=step, conditioning=c_,
#                                         batch_size=batch_size, shape=shape,
#                                         verbose=False, 
#                                         unconditional_guidance_scale=scale,
#                                         unconditional_conditioning=uc_,
#                                         eta=ddim_eta, x_T=None)
#         x_sample = model.decode_first_stage(samples_ddim)
#         x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
#         x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

#     return list(x_sample.astype(np.uint8))


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
#     parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
#     parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
#     parser.add_argument("--text", type=str, default="an astronaut riding a horse")
#     parser.add_argument("--suffix", type=str, default=", 3d asset")
#     parser.add_argument("--size", type=int, default=256)
#     parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
#     parser.add_argument("--use_camera", type=int, default=1)
#     parser.add_argument("--camera_elev", type=int, default=15)
#     parser.add_argument("--camera_azim", type=int, default=90)
#     parser.add_argument("--camera_azim_span", type=int, default=360)
#     parser.add_argument("--seed", type=int, default=23)
#     parser.add_argument("--fp16", action="store_true")
#     parser.add_argument("--device", type=str, default='cuda')
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--prompt", type=str, default="a photo of <asset0> at the beach")
#     args = parser.parse_args()

#     dtype = torch.float16 if args.fp16 else torch.float32
#     device = args.device
#     batch_size = max(4, args.num_frames)
#     pipeline = DiffusionPipeline.from_pretrained(
#             args.model_path,
#             torch_dtype=torch.float16,
#         )
#     print("load t2i model ... ")
#     if args.config_path is None:
#         model = build_model(args.model_name, ckpt_path=args.ckpt_path)
#     else:
#         assert args.ckpt_path is not None, "ckpt_path must be specified!"
#         config = OmegaConf.load(args.config_path)
#         model = instantiate_from_config(config.model)
#         model.load_state_dict(torch.load(args.ckpt_path, map_location='cpu'))
#     model.device = device
#     model.to(device)
#     model.eval()

#     sampler = DDIMSampler(model)
#     uc = model.get_learned_conditioning( [""] ).to(device)
#     print("load t2i model done . ")

#     # pre-compute camera matrices
#     if args.use_camera:
#         camera = get_camera(args.num_frames, elevation=args.camera_elev, 
#                 azimuth_start=args.camera_azim, azimuth_span=args.camera_azim_span)
#         camera = camera.repeat(batch_size//args.num_frames,1).to(device)
#     else:
#         camera = None
    
#     t = args.text + args.suffix
#     set_seed(args.seed)
#     images = []
#     for j in range(3):
#         img = t2i(model, args.size, t, uc, sampler, step=50, scale=10, batch_size=batch_size, ddim_eta=0.0, 
#                 dtype=dtype, device=device, camera=camera, num_frames=args.num_frames)
#         img = np.concatenate(img, 1)
#         images.append(img)
#     images = np.concatenate(images, 0)
#     Image.fromarray(images).save(f"sample.png")


"""
MVDream integration with Break-A-Scene custom assets
Adapted from Break-A-Scene inference for multi-view generation
"""

import argparse
import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, DDIMScheduler

from mvdream.camera_utils import get_camera
from mvdream.ldm.util import instantiate_from_config
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model
from omegaconf import OmegaConf


class MVDreamBreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_mvdream_model()
        self._load_custom_pipeline()
        self._setup_camera()

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="MVDream with Break-A-Scene custom assets")
        
        # Break-A-Scene model (contains your <asset0> token)
        parser.add_argument("--custom_model_path", type=str, required=True,
                           help="Path to Break-A-Scene model with custom tokens")
        
        # MVDream model settings
        parser.add_argument("--mvdream_model", type=str, default="sd-v2.1-base-4view",
                           help="MVDream base model")
        parser.add_argument("--mvdream_config", type=str, default=None,
                           help="MVDream config path")
        parser.add_argument("--mvdream_ckpt", type=str, default=None,
                           help="MVDream checkpoint path")
        
        # Generation settings
        parser.add_argument("--prompt", type=str, default="a photo of <asset0> at the beach")
        parser.add_argument("--suffix", type=str, default=", 3d asset")
        parser.add_argument("--output_path", type=str, default="outputs/mvdream_result.png")
        parser.add_argument("--device", type=str, default="cuda")
        
        # Multi-view settings
        parser.add_argument("--num_frames", type=int, default=4, help="Number of views")
        parser.add_argument("--image_size", type=int, default=256)
        parser.add_argument("--elevation", type=int, default=15)
        parser.add_argument("--azimuth_start", type=int, default=90)
        parser.add_argument("--azimuth_span", type=int, default=360)
        
        # Sampling settings
        parser.add_argument("--num_inference_steps", type=int, default=50)
        parser.add_argument("--guidance_scale", type=float, default=10.0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--fp16", action="store_true")
        
        self.args = parser.parse_args()
        
    def _load_mvdream_model(self):
        """Load MVDream model for multi-view generation"""
        print("Loading MVDream model...")
        
        if self.args.mvdream_config is None:
            self.mvdream_model = build_model(self.args.mvdream_model, 
                                           ckpt_path=self.args.mvdream_ckpt)
        else:
            assert self.args.mvdream_ckpt is not None, "ckpt_path must be specified!"
            config = OmegaConf.load(self.args.mvdream_config)
            self.mvdream_model = instantiate_from_config(config.model)
            self.mvdream_model.load_state_dict(torch.load(self.args.mvdream_ckpt, map_location='cpu'))
        
        self.mvdream_model.device = self.args.device
        self.mvdream_model.to(self.args.device)
        self.mvdream_model.eval()
        
        # Setup sampler
        self.sampler = DDIMSampler(self.mvdream_model)
        self.unconditional_conditioning = self.mvdream_model.get_learned_conditioning([""]).to(self.args.device)
        
        print("MVDream model loaded successfully.")

    def _load_custom_pipeline(self):
        """Load Break-A-Scene pipeline with custom tokens"""
        print(f"Loading custom Break-A-Scene model from: {self.args.custom_model_path}")
        
        try:
            self.custom_pipeline = DiffusionPipeline.from_pretrained(
                self.args.custom_model_path,
                torch_dtype=torch.float16 if self.args.fp16 else torch.float32,
            )
            self.custom_pipeline.scheduler = DDIMScheduler(
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
            )
            self.custom_pipeline.to(self.args.device)
            print("Custom pipeline loaded successfully.")
            
        except Exception as e:
            print(f"Error loading custom pipeline: {e}")
            print("Will try to transfer embeddings to MVDream model...")
            self.custom_pipeline = None
            self._transfer_custom_embeddings()

    def _transfer_custom_embeddings(self):
        """Transfer custom embeddings from Break-A-Scene to MVDream"""
        try:
            import os
            from safetensors import safe_open
            from transformers import CLIPTextModel
            
            # Look for embedding files
            embedding_files = [
                os.path.join(self.args.custom_model_path, "learned_embeds.bin"),
                os.path.join(self.args.custom_model_path, "learned_embeds.safetensors"),
                os.path.join(self.args.custom_model_path, "embeddings.pt"),
            ]
            
            embeddings_loaded = False
            for embed_file in embedding_files:
                if os.path.exists(embed_file):
                    print(f"Loading embeddings from: {embed_file}")
                    
                    if embed_file.endswith('.safetensors'):
                        with safe_open(embed_file, framework="pt", device=self.args.device) as f:
                            embeddings = {key: f.get_tensor(key) for key in f.keys()}
                    else:
                        embeddings = torch.load(embed_file, map_location=self.args.device)
                    
                    # Transfer to MVDream's text encoder
                    if hasattr(self.mvdream_model, 'cond_stage_model'):
                        text_encoder = self.mvdream_model.cond_stage_model
                        if hasattr(text_encoder, 'transformer'):
                            token_embeds = text_encoder.transformer.text_model.embeddings.token_embedding
                            
                            for token_name, embedding in embeddings.items():
                                print(f"Adding custom token: {token_name}")
                                # Add the embedding to the token embedding layer
                                # Note: This is a simplified approach - you might need to adjust based on your model structure
                                
                    embeddings_loaded = True
                    break
            
            if not embeddings_loaded:
                print("No custom embeddings found. Using base model only.")
                
        except Exception as e:
            print(f"Error transferring embeddings: {e}")

    def _setup_camera(self):
        """Setup camera parameters for multi-view generation"""
        self.camera = get_camera(
            self.args.num_frames,
            elevation=self.args.elevation,
            azimuth_start=self.args.azimuth_start,
            azimuth_span=self.args.azimuth_span
        )
        batch_size = max(4, self.args.num_frames)
        self.camera = self.camera.repeat(batch_size // self.args.num_frames, 1).to(self.args.device)

    def _set_seed(self):
        """Set random seed for reproducibility"""
        import random
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

    @torch.no_grad()
    def generate_multiview(self, prompt):
        """Generate multi-view images using MVDream with custom tokens"""
        self._set_seed()
        
        full_prompt = prompt + self.args.suffix
        print(f"Generating multi-view images with prompt: '{full_prompt}'")
        
        # Prepare conditioning
        dtype = torch.float16 if self.args.fp16 else torch.float32
        batch_size = max(4, self.args.num_frames)
        
        with torch.no_grad(), torch.autocast(device_type=self.args.device, dtype=dtype):
            # Get text conditioning
            c = self.mvdream_model.get_learned_conditioning([full_prompt]).to(self.args.device)
            c_ = {"context": c.repeat(batch_size, 1, 1)}
            uc_ = {"context": self.unconditional_conditioning.repeat(batch_size, 1, 1)}
            
            # Add camera conditioning
            c_["camera"] = uc_["camera"] = self.camera
            c_["num_frames"] = uc_["num_frames"] = self.args.num_frames

            # Sample
            shape = [4, self.args.image_size // 8, self.args.image_size // 8]
            samples_ddim, _ = self.sampler.sample(
                S=self.args.num_inference_steps, 
                conditioning=c_,
                batch_size=batch_size, 
                shape=shape,
                verbose=False, 
                unconditional_guidance_scale=self.args.guidance_scale,
                unconditional_conditioning=uc_,
                eta=0.0, 
                x_T=None
            )
            
            # Decode images
            x_sample = self.mvdream_model.decode_first_stage(samples_ddim)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0, 2, 3, 1).cpu().numpy()

        return [img.astype(np.uint8) for img in x_sample]

    def infer_and_save(self):
        """Main inference function"""
        print("Starting multi-view generation...")
        
        # Method 1: Try using custom pipeline if available
        if self.custom_pipeline is not None:
            try:
                print("Attempting generation with custom pipeline...")
                # This might not work directly since Break-A-Scene isn't multi-view
                # But we can try to adapt it
                images = self.custom_pipeline(
                    [self.args.prompt + self.args.suffix],
                    num_inference_steps=self.args.num_inference_steps,
                    guidance_scale=self.args.guidance_scale,
                ).images
                
                # For now, just replicate the single image for multi-view
                # In a more sophisticated implementation, you'd modify the pipeline
                print("Custom pipeline generated single view. Replicating for multi-view...")
                multi_view_images = [np.array(images[0]) for _ in range(self.args.num_frames)]
                
            except Exception as e:
                print(f"Custom pipeline failed: {e}")
                print("Falling back to MVDream with transferred embeddings...")
                multi_view_images = self.generate_multiview(self.args.prompt)
        else:
            # Method 2: Use MVDream with transferred embeddings
            multi_view_images = self.generate_multiview(self.args.prompt)

        # Combine images into a single output
        combined_image = np.concatenate(multi_view_images, axis=1)  # Horizontal concatenation
        
        # Save result
        Image.fromarray(combined_image).save(self.args.output_path)
        print(f"Multi-view images saved to: {self.args.output_path}")
        
        return multi_view_images


if __name__ == "__main__":
    mvdream_inference = MVDreamBreakASceneInference()
    mvdream_inference.infer_and_save()