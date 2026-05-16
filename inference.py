"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
# MVDream imports
from mvdream.camera_utils import get_camera, get_camera_from_lists
from mvdream.ldm.models.diffusion.ddim import DDIMSampler
from mvdream.model_zoo import build_model


class BreakASceneInference:
    def __init__(self):
        self._parse_args()
        self._load_pipeline()

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument(
            "--prompt", type=str, default="a photo of <asset0> at the beach"
        )
        parser.add_argument("--output_path", type=str, default="outputs/result.jpg")
        parser.add_argument("--device", type=str, default="cuda")
        parser.add_argument("--size", type=int, default= 256, help="Image size for generation")
        parser.add_argument("--num_frames", type=int, default=4, help="Number of views to generate")
        parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
        parser.add_argument("--scale", type=float, default=7.5, help="Guidance scale")
        parser.add_argument("--camera_elev", type=int, default=15, help="Camera elevation (used if elevation_list not provided)")
        parser.add_argument("--camera_azim", type=int, default=90, help="Camera azimuth start")
        parser.add_argument("--camera_azim_span", type=int, default=360, help="Camera azimuth span")
        parser.add_argument("--elevation_list", type=str, default=None,
                          help="Comma-separated list of elevations (e.g., '20,-10' for 2 rows with different elevations)")
        parser.add_argument(
            "--negative_prompt",
            type=str,
            default="blurry, low quality, bad anatomy, distorted, ugly, noisy, artifacts, poorly rendered",
            help="Negative prompt to avoid unwanted features"
        )
        parser.add_argument(
            "--positive_prompt",
            type=str,
            default="high quality, detailed, sharp focus, professional lighting, good contrast, studio lighting, well-lit, crisp details",
            help="Positive prompt suffix to enhance quality (appended to your prompt)"
        )
        parser.add_argument(
            "--num_generations",
            type=int,
            default=1,
            help="Number of 1xN image strips to generate (model loaded once, faster for multiple)"
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=-1,
            help="Random seed (-1 for random)"
        )
        self.args = parser.parse_args()

    def _load_pipeline(self):
        # Load the trained MVDream model
        mvdream_model_path = os.path.join(self.args.model_path, "mvdream_model.pth")
        
        if os.path.exists(mvdream_model_path):
            print("Loading trained MVDream model...")
            # Load base MVDream model
            self.mvdream_model = build_model("sd-v2.1-base-4view")
            
            # Load trained weights
            state_dict = torch.load(mvdream_model_path, map_location='cpu')
            
            # Handle token embedding size mismatch
            current_embeddings = self.mvdream_model.cond_stage_model.model.token_embedding.weight
            saved_embeddings = state_dict['cond_stage_model.model.token_embedding.weight']
            
            if current_embeddings.shape != saved_embeddings.shape:
                print(f"Expanding token embeddings from {current_embeddings.shape} to {saved_embeddings.shape}")
                # Create new embedding layer with the correct size
                new_embedding = torch.nn.Embedding(saved_embeddings.shape[0], saved_embeddings.shape[1])
                new_embedding.weight.data = saved_embeddings
                self.mvdream_model.cond_stage_model.model.token_embedding = new_embedding
                
                # Remove the embedding weight from state_dict to avoid the mismatch error
                del state_dict['cond_stage_model.model.token_embedding.weight']
            
            # Load the rest of the state dict
            self.mvdream_model.load_state_dict(state_dict, strict=False)
            self.mvdream_model.to(self.args.device)
            print("Trained MVDream model loaded successfully.")
        else:
            print("No trained MVDream model found, using base model...")
            # Fallback to base model if trained model not found
            self.mvdream_model = build_model("sd-v2.1-base-4view")
            self.mvdream_model.to(self.args.device)
        
        # Create DDIM sampler
        self.sampler = DDIMSampler(self.mvdream_model)
        
        # Prepare unconditional embeddings (using negative prompt for better quality)
        self.uc = self.mvdream_model.get_learned_conditioning([self.args.negative_prompt]).to(self.args.device)
        print(f"Using negative prompt: {self.args.negative_prompt}")

    @torch.no_grad()
    def infer_and_save(self, prompts):
        # Combine user prompt with positive prompt suffix
        user_prompt = prompts[0]
        if self.args.positive_prompt:
            full_prompt = f"{user_prompt}, {self.args.positive_prompt}"
        else:
            full_prompt = user_prompt

        print(f"Generating with prompt: {full_prompt}")

        # Calculate azimuth angles for each view
        angle_gap = self.args.camera_azim_span / self.args.num_frames
        azimuth_list = [(self.args.camera_azim + i * angle_gap) % 360 for i in range(self.args.num_frames)]

        # Parse elevation list for per-view elevations (single batch, mixed elevations)
        if self.args.elevation_list:
            input_elevations = [float(e.strip()) for e in self.args.elevation_list.split(',')]
            # Cycle through elevations for each view (e.g., [20, -10] -> [20, -10, 20, -10] for 4 views)
            elevation_list = [input_elevations[i % len(input_elevations)] for i in range(self.args.num_frames)]
        else:
            elevation_list = [self.args.camera_elev] * self.args.num_frames

        print(f"Generating {self.args.num_frames} views in single batch:")
        for i, (elev, azim) in enumerate(zip(elevation_list, azimuth_list)):
            print(f"  View {i+1}: elevation {elev}°, azimuth {azim}°")

        # Generate camera matrices with per-view elevation/azimuth
        camera = get_camera_from_lists(elevation_list, azimuth_list)
        camera = camera.to(self.args.device)

        # Prepare output directory
        os.makedirs(os.path.dirname(self.args.output_path), exist_ok=True)

        # Get base path for multiple outputs
        base_path = self.args.output_path
        path_parts = os.path.splitext(base_path)

        # Set initial seed
        if self.args.seed >= 0:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)

        saved_paths = []

        # Generate multiple times (model already loaded, much faster!)
        for gen_idx in range(self.args.num_generations):
            if self.args.num_generations > 1:
                print(f"\n=== Generation {gen_idx + 1}/{self.args.num_generations} ===")

            # Generate all views in a single batch (MVDream cross-view attention ensures consistency)
            images = self.t2i(
                prompt=full_prompt,
                step=self.args.steps,
                scale=self.args.scale,
                camera=camera,
                num_frames=self.args.num_frames
            )

            # Concatenate views horizontally
            if len(images) > 1:
                combined_image = np.concatenate(images, axis=1)
            else:
                combined_image = images[0]

            # Save result with index if multiple generations
            if self.args.num_generations > 1:
                output_path = f"{path_parts[0]}_{gen_idx + 1:02d}{path_parts[1]}"
            else:
                output_path = base_path

            Image.fromarray(combined_image).save(output_path)
            saved_paths.append(output_path)
            print(f"Result saved to: {output_path}")

        if self.args.num_generations > 1:
            print(f"\n✅ Generated {self.args.num_generations} images total")
    
    def t2i(self, prompt, step=20, scale=7.5, camera=None, num_frames=1):
        """Text-to-image generation using MVDream"""
        with torch.no_grad(), torch.autocast(device_type=str(self.args.device).split(':')[0], dtype=torch.float16):
            # Repeat prompt for batch size
            prompts = [prompt] * num_frames
            c = self.mvdream_model.get_learned_conditioning(prompts).to(self.args.device)
            c_ = {"context": c}

            # Repeat unconditional conditioning for batch size
            uc_repeated = self.uc.repeat(num_frames, 1, 1)
            uc_ = {"context": uc_repeated}

            if camera is not None:
                c_["camera"] = uc_["camera"] = camera
                c_["num_frames"] = uc_["num_frames"] = num_frames

            shape = [4, self.args.size // 8, self.args.size // 8]
            samples_ddim, _ = self.sampler.sample(
                S=step,
                conditioning=c_,
                batch_size=num_frames,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_,
                eta=0.0,
                x_T=None
            )
            
            x_sample = self.mvdream_model.decode_first_stage(samples_ddim)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

        return list(x_sample.astype(np.uint8))


if __name__ == "__main__":
    break_a_scene_inference = BreakASceneInference()
    break_a_scene_inference.infer_and_save(
        prompts=[break_a_scene_inference.args.prompt]
    )