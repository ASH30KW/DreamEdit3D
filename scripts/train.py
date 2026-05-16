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
import hashlib
import itertools
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import List, Optional
import random
import time

# Make repo root importable when this script is invoked as
# `python scripts/train.py ...` so that mvdream/ and utils/ resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.utils.checkpoint
from torch.utils.data import Dataset
import numpy as np

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    PYNVML_AVAILABLE = True
except:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU metrics will be limited. Install with: pip install nvidia-ml-py3")

import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from utils import ptp_utils
from utils.ptp_utils import AttentionStore
from diffusers.models.cross_attention import CrossAttention

# MVDream imports
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

def get_gpu_metrics():
    """Get GPU metrics including memory and utilization."""
    metrics = {}

    if torch.cuda.is_available():
        # PyTorch built-in metrics
        device = torch.cuda.current_device()
        metrics['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated(device) / 1e9
        metrics['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved(device) / 1e9
        metrics['gpu_memory_allocated_pct'] = torch.cuda.memory_allocated(device) / torch.cuda.get_device_properties(device).total_memory * 100

        # NVML metrics if available
        if PYNVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)

                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics['gpu_utilization_pct'] = utilization.gpu
                metrics['gpu_memory_utilization_pct'] = utilization.memory

                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics['gpu_temperature_c'] = temp

                # Power
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                metrics['gpu_power_watts'] = power

            except Exception as e:
                pass  # Silently skip if metrics unavailable

    return metrics

check_min_version("0.12.0")

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import (
            RobertaSeriesModelWithTransformation,
        )

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="MVDream Spatial DreamBooth training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        required=True,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a photo at the beach",
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--no_prior_preservation",
        action="store_false",
        help="Flag to add prior preservation loss.",
        dest="with_prior_preservation"
    )
    parser.add_argument(
        "--prior_loss_weight",
        type=float,
        default=1.0,
        help="The weight of prior preservation loss.",
    )
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--no_train_text_encoder",
        action="store_false",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
        dest="train_text_encoder"
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--phase1_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the first phase.",
    )
    parser.add_argument(
        "--phase2_train_steps",
        type=int,
        default="400",
        help="Number of trainig steps for the second phase.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=5000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="DEPRECATED: Use actual batching instead. This is kept at 1 for compatibility.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--initial_learning_rate",
        type=float,
        default=5e-4,
        help="The LR for the Textual Inversion steps.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="dreamedit3d",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name. If not specified, will be auto-generated.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument("--lambda_attention", type=float, default=1e-2)
    parser.add_argument("--img_log_steps", type=int, default=200)
    parser.add_argument("--num_of_assets", type=int, default=1)
    parser.add_argument("--concept_names", type=str, nargs="+", default=[], help="Concept names for labeled masks (e.g., body head)")
    parser.add_argument("--initializer_tokens", type=str, nargs="+", default=[])
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<asset>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--do_not_apply_masked_loss",
        action="store_false",
        help="Use masked loss instead of standard epsilon prediciton loss",
        dest="apply_masked_loss"
    )
    parser.add_argument(
        "--log_checkpoints",
        action="store_true",
        help="Indicator to log intermediate model checkpoints",
    )

    # MVDream specific arguments
    parser.add_argument("--model_name", type=str, default="sd-v2.1-base-4view", help="load pre-trained model from hugginface")
    parser.add_argument("--config_path", type=str, default=None, help="load model from local config (override model_name)")
    parser.add_argument("--ckpt_path", type=str, default=None, help="path to local checkpoint")
    parser.add_argument("--text", type=str, default="an astronaut riding a horse")
    parser.add_argument("--suffix", type=str, default=", 3d asset")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=4, help="num of frames (views) to generate")
    parser.add_argument("--use_camera", type=int, default=1)
    parser.add_argument("--camera_elev", type=int, default=15)
    parser.add_argument("--camera_azim", type=int, default=90)
    parser.add_argument("--camera_azim_span", type=int, default=360)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", type=str, default='cuda')
    
    # MVDream training mode switches
    parser.add_argument("--use_3d_attention", action="store_true", help="Use 3D attention during training")
    parser.add_argument("--mvdream_training_mode", type=str, default="mixed", choices=["2d", "3d", "mixed"],
                       help="Training mode: 2d (standard), 3d (multi-view), or mixed")

    # Single-view vs multi-view training
    parser.add_argument("--view_mode", type=str, default="multi", choices=["single", "multi"],
                       help="View mode: 'single' for training on a specific view, 'multi' for all views together")
    parser.add_argument("--view_index", type=int, default=0,
                       help="Which view to use in single-view mode (0-indexed, e.g., 0 for 1st view, 1 for 2nd view)")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    assert len(args.initializer_tokens) == 0 or len(args.initializer_tokens) == args.num_of_assets
    args.max_train_steps = args.phase1_train_steps + args.phase2_train_steps

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        if args.class_data_dir is not None:
            warnings.warn(
                "You need not use --class_data_dir without --with_prior_preservation."
            )
        if args.class_prompt is not None:
            warnings.warn(
                "You need not use --class_prompt without --with_prior_preservation."
            )

    return args


class DreamBoothDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        placeholder_tokens,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        size=256,
        center_crop=False,
        num_of_assets=1,
        flip_p=0.5,
        num_frames=4,
        use_multiview=False,
        view_index=0,
        view_mode="multi",
        concept_names=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.flip_p = flip_p
        self.num_frames = num_frames
        self.use_multiview = use_multiview
        self.view_index = view_index  # Which view to use in single-view mode
        self.view_mode = view_mode  # "single" or "multi"
        self.current_view_index = 0  # For cycling through views
        self.concept_names = concept_names  # List of concept names for labeled masks

        # Helper function to find mask path (numbered or labeled)
        def find_mask_path(directory, mask_index):
            """Find mask file by index, trying both numbered (mask0.png) and labeled (mask_body.png) formats"""
            # Try numbered format first
            numbered_path = os.path.join(directory, f"mask{mask_index}.png")
            if os.path.exists(numbered_path):
                return numbered_path

            # Try labeled format using concept_names
            if self.concept_names and mask_index < len(self.concept_names):
                label = self.concept_names[mask_index].strip()
                labeled_path = os.path.join(directory, f"mask_{label}.png")
                if os.path.exists(labeled_path):
                    return labeled_path

            # Fallback: try to find any mask files and use by index
            import glob
            mask_files = sorted(glob.glob(os.path.join(directory, "mask*.png")))
            if mask_index < len(mask_files):
                return mask_files[mask_index]

            raise FileNotFoundError(f"Could not find mask{mask_index}.png or labeled mask in {directory}")

        self.find_mask_path = find_mask_path

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ]
        )

        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(
                f"Instance {self.instance_data_root} images root doesn't exists."
            )

        self.placeholder_tokens = placeholder_tokens

        # Check if multi-view data structure exists (view_1, view_2, etc.)
        view_dirs = [os.path.join(instance_data_root, f"view_{i+1}") for i in range(num_frames)]
        has_multiview_data = all(os.path.isdir(vd) for vd in view_dirs)

        # Handle multi-view data loading based on view_mode
        if has_multiview_data:
            if self.view_mode == "single":
                # Single-view mode: Load only the specified view
                selected_view_dir = view_dirs[self.view_index]

                # Load image for the selected view
                view_img_path = os.path.join(selected_view_dir, "img.jpg")
                self.instance_image = self.image_transforms(Image.open(view_img_path))

                # Load masks for the selected view
                self.instance_masks = []
                for i in range(num_of_assets):
                    view_mask_path = self.find_mask_path(selected_view_dir, i)
                    curr_mask = Image.open(view_mask_path)
                    curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                    self.instance_masks.append(curr_mask)
                self.instance_masks = torch.cat(self.instance_masks)

                self.instance_images = None  # Not used in single-view mode
                self.instance_masks_multiview = None  # Not used in single-view mode
            else:
                # Multi-view mode: Load ALL views from subdirectories (original behavior)
                self.instance_images = []
                self.instance_masks_multiview = []

                for view_dir in view_dirs:
                    # Load image for this view
                    view_img_path = os.path.join(view_dir, "img.jpg")
                    view_image = self.image_transforms(Image.open(view_img_path))
                    self.instance_images.append(view_image)

                    # Load masks for this view
                    view_masks = []
                    for i in range(num_of_assets):
                        view_mask_path = self.find_mask_path(view_dir, i)
                        curr_mask = Image.open(view_mask_path)
                        curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                        view_masks.append(curr_mask)
                    self.instance_masks_multiview.append(torch.cat(view_masks))

                # Stack all views: [num_frames, C, H, W] for images
                self.instance_images = torch.stack(self.instance_images)
                # Stack masks: [num_frames, num_assets, 1, 1, H, W]
                self.instance_masks_multiview = torch.stack(self.instance_masks_multiview)
                self.instance_image = None  # Not used in multi-view mode with real data
                self.instance_masks = None  # Not used in multi-view mode with real data
        else:
            # Original single-view loading
            instance_img_path = os.path.join(instance_data_root, "img.jpg")
            self.instance_image = self.image_transforms(Image.open(instance_img_path))

            self.instance_masks = []
            for i in range(num_of_assets):
                instance_mask_path = self.find_mask_path(instance_data_root, i)
                curr_mask = Image.open(instance_mask_path)
                curr_mask = self.mask_transforms(curr_mask)[0, None, None, ...]
                self.instance_masks.append(curr_mask)
            self.instance_masks = torch.cat(self.instance_masks)

            self.instance_images = None  # Not used in single-view mode
            self.instance_masks_multiview = None  # Not used in single-view mode

        self._length = 1

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self._length)
            self.class_prompt = class_prompt
        else:
            self.class_data_root = None

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}

        num_of_tokens = random.randrange(1, len(self.placeholder_tokens) + 1)
        tokens_ids_to_use = random.sample(
            range(len(self.placeholder_tokens)), k=num_of_tokens
        )
        tokens_to_use = [self.placeholder_tokens[tkn_i] for tkn_i in tokens_ids_to_use]
        prompt = "a photo of " + " and ".join(tokens_to_use)

        # Handle multi-view training - ALWAYS use all views together
        if self.instance_images is not None:
            # Multi-view data exists - always return all views for simultaneous processing
            instance_images = self.instance_images  # [num_frames, C, H, W]
            # Get masks for selected tokens across all views
            # instance_masks_multiview: [num_frames, num_assets, 1, 1, H, W]
            # Select assets based on tokens_ids_to_use
            instance_masks = self.instance_masks_multiview[:, tokens_ids_to_use]  # [num_frames, num_selected_tokens, 1, 1, H, W]
            # Keep shape as [num_frames, num_selected_tokens, 1, 1, H, W] for now
            # Will be reshaped in training loop if needed
        else:
            # Single-view data (original behavior)
            if self.use_multiview:
                # Generate multiple views by repeating the image for compatibility
                instance_images = self.instance_image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
                instance_masks = self.instance_masks[tokens_ids_to_use].unsqueeze(0).repeat(self.num_frames, 1, 1, 1, 1)
            else:
                # Single view mode
                instance_images = self.instance_image
                instance_masks = self.instance_masks[tokens_ids_to_use]

        example["instance_images"] = instance_images
        example["instance_masks"] = instance_masks
        example["token_ids"] = torch.tensor(tokens_ids_to_use)

        # Apply horizontal flip augmentation to all views if needed
        if random.random() > self.flip_p:
            if instance_images.dim() == 4:  # Multi-view: [num_frames, C, H, W]
                for i in range(instance_images.shape[0]):
                    example["instance_images"][i] = TF.hflip(example["instance_images"][i])
                # Flip masks - handle different shapes
                if instance_masks.dim() == 5:  # [num_frames, num_tokens, 1, 1, H, W]
                    for i in range(instance_masks.shape[0]):
                        for j in range(instance_masks.shape[1]):
                            example["instance_masks"][i, j] = TF.hflip(example["instance_masks"][i, j])
                else:
                    example["instance_masks"] = TF.hflip(example["instance_masks"])
            else:
                # Single view
                example["instance_images"] = TF.hflip(example["instance_images"])
                example["instance_masks"] = TF.hflip(example["instance_masks"])

        example["instance_prompt_ids"] = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images]
            )
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            
            if self.use_multiview:
                class_images = self.image_transforms(class_image).unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
                example["class_images"] = class_images
            else:
                example["class_images"] = self.image_transforms(class_image)
                
            example["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    masks = [example["instance_masks"] for example in examples]
    token_ids = [example["token_ids"] for example in examples]

    if with_prior_preservation:
        input_ids = [example["class_prompt_ids"] for example in examples] + input_ids
        pixel_values = [example["class_images"] for example in examples] + pixel_values

    # Stack pixel values - handles both [C, H, W] and [num_frames, C, H, W]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    # Result: [batch, C, H, W] for single-view or [batch, num_frames, C, H, W] for multi-view

    input_ids = torch.cat(input_ids, dim=0)

    # Stack masks - handles both [tokens, 1, 1, H, W] and [frames, tokens, 1, 1, H, W]
    masks = torch.stack(masks)
    # Result: [batch, tokens, 1, 1, H, W] or [batch, frames, tokens, 1, 1, H, W]

    token_ids = torch.stack(token_ids)

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "instance_masks": masks,
        "token_ids": token_ids,
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def get_full_repo_name(
    model_id: str, organization: Optional[str] = None, token: Optional[str] = None
):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"
    

def t2i(model, image_size, prompt, uc, sampler, step=20, scale=7.5, batch_size=8, ddim_eta=0., dtype=torch.float32, device="cuda", camera=None, num_frames=1):
    if type(prompt)!=list:
        prompt = [prompt]
    with torch.no_grad(), torch.autocast(device_type=str(device).split(':')[0], dtype=dtype):
        c = model.get_learned_conditioning(prompt).to(device)
        c_ = {"context": c.repeat(batch_size,1,1)}
        uc_ = {"context": uc.repeat(batch_size,1,1)}
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
        x_sample = 255. * x_sample.permute(0,2,3,1).cpu().numpy()

    return list(x_sample.astype(np.uint8))


class SpatialDreambooth:
    def __init__(self):
        self.args = parse_args()
        self.main()

    def main(self):
        logging_dir = Path(self.args.output_dir, self.args.logging_dir)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=self.args.mixed_precision,
            log_with=self.args.report_to,
            logging_dir=logging_dir,
        )

        if (
            self.args.train_text_encoder
            and self.args.gradient_accumulation_steps > 1
            and self.accelerator.num_processes > 1
        ):
            raise ValueError(
                "Gradient accumulation is not supported when training the text encoder in distributed training. "
                "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
            )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        if self.accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if self.args.seed is not None:
            set_seed(self.args.seed)

        # Load MVDream model
        print("Loading MVDream model...")
        if self.args.config_path is None:
            self.mvdream_model = build_model(self.args.model_name, ckpt_path=self.args.ckpt_path)
        else:
            assert self.args.ckpt_path is not None, "ckpt_path must be specified!"
            config = OmegaConf.load(self.args.config_path)
            self.mvdream_model = instantiate_from_config(config.model)
            self.mvdream_model.load_state_dict(torch.load(self.args.ckpt_path, map_location='cpu'))
        
        self.mvdream_model.to(self.accelerator.device)
        print("MVDream model loaded successfully.")

        # Generate class images if prior preservation is enabled.
        if self.args.with_prior_preservation:
            class_images_dir = Path(self.args.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.args.num_class_images:
                torch_dtype = (
                    torch.float16
                    if self.accelerator.device.type == "cuda"
                    else torch.float32
                )
                if self.args.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif self.args.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif self.args.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                
                # Use MVDream for class image generation
                sampler = DDIMSampler(self.mvdream_model)
                uc = self.mvdream_model.get_learned_conditioning([""]).to(self.accelerator.device)
                
                # Generate camera matrices for class images
                if self.args.use_camera:
                    camera = get_camera(self.args.num_frames, elevation=self.args.camera_elev, 
                                      azimuth_start=self.args.camera_azim, azimuth_span=self.args.camera_azim_span)
                    camera = camera.repeat(self.args.sample_batch_size//self.args.num_frames, 1).to(self.accelerator.device)
                else:
                    camera = None

                num_new_images = self.args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(self.args.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(
                    sample_dataset, batch_size=self.args.sample_batch_size
                )

                sample_dataloader = self.accelerator.prepare(sample_dataloader)

                for example in tqdm(
                    sample_dataloader,
                    desc="Generating class images",
                    disable=not self.accelerator.is_local_main_process,
                ):
                    # Generate multi-view images using MVDream
                    images = t2i(self.mvdream_model, self.args.size, example["prompt"], uc, sampler, 
                               step=50, scale=7.5, batch_size=self.args.sample_batch_size, 
                               ddim_eta=0.0, dtype=torch_dtype, device=self.accelerator.device, 
                               camera=camera, num_frames=self.args.num_frames)

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = (
                            class_images_dir
                            / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        )
                        Image.fromarray(image).save(image_filename)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Handle the repository creation
        if self.accelerator.is_main_process:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # Extract components from MVDream model
        self.unet = self.mvdream_model.model.diffusion_model
        self.vae = self.mvdream_model.first_stage_model
        self.text_encoder = self.mvdream_model.cond_stage_model

        # Load scheduler and tokenizer
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.args.pretrained_model_name_or_path, subfolder="scheduler"
        )

        # Load the tokenizer
        if self.args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.tokenizer_name, revision=self.args.revision, use_fast=False
            )
        elif self.args.pretrained_model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.args.revision,
                use_fast=False,
            )

        # Add assets tokens to tokenizer
        self.placeholder_tokens = [
            self.args.placeholder_token.replace(">", f"{idx}>")
            for idx in range(self.args.num_of_assets)
        ]
        num_added_tokens = self.tokenizer.add_tokens(self.placeholder_tokens)
        assert num_added_tokens == self.args.num_of_assets
        self.placeholder_token_ids = self.tokenizer.convert_tokens_to_ids(
            self.placeholder_tokens
        )
        
        # Handle MVDream's FrozenOpenCLIPEmbedder vs standard transformers text encoder
        if hasattr(self.text_encoder, 'resize_token_embeddings'):
            # Standard transformers text encoder
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            token_embeds = self.text_encoder.get_input_embeddings().weight.data
        else:
            # MVDream's FrozenOpenCLIPEmbedder - find the correct embedding layer
            # Let's explore the structure first
            print(f"Text encoder type: {type(self.text_encoder)}")
            print(f"Text encoder attributes: {dir(self.text_encoder)}")
            
            # Try to find the embedding layer in different possible paths
            embedding_layer = None
            token_embeds = None
            
            if hasattr(self.text_encoder, 'model') and hasattr(self.text_encoder.model, 'token_embedding'):
                # Path: text_encoder.model.token_embedding
                embedding_layer = self.text_encoder.model.token_embedding
                print("Found embeddings at: text_encoder.model.token_embedding")
            elif hasattr(self.text_encoder, 'transformer') and hasattr(self.text_encoder.transformer, 'token_embedding'):
                # Path: text_encoder.transformer.token_embedding
                embedding_layer = self.text_encoder.transformer.token_embedding
                print("Found embeddings at: text_encoder.transformer.token_embedding")
            elif hasattr(self.text_encoder, 'embeddings'):
                # Path: text_encoder.embeddings
                embedding_layer = self.text_encoder.embeddings
                print("Found embeddings at: text_encoder.embeddings")
            else:
                # Fallback: search for any embedding layer
                for name, module in self.text_encoder.named_modules():
                    if isinstance(module, torch.nn.Embedding):
                        print(f"Found embedding layer at: {name}")
                        embedding_layer = module
                        break
            
            if embedding_layer is None:
                # If we can't find embeddings, skip token addition for now
                print("Warning: Could not find token embedding layer. Skipping token addition.")
                token_embeds = None
            else:
                old_embeddings = embedding_layer.weight.data
                old_vocab_size, embedding_dim = old_embeddings.shape
                new_vocab_size = len(self.tokenizer)
                
                # Create new embedding layer
                new_embedding_layer = torch.nn.Embedding(new_vocab_size, embedding_dim)
                new_embedding_layer.weight.data[:old_vocab_size] = old_embeddings
                
                # Initialize new tokens randomly
                if len(self.args.initializer_tokens) == 0:
                    # Random initialization for new tokens
                    new_embedding_layer.weight.data[old_vocab_size:] = torch.randn(
                        new_vocab_size - old_vocab_size, embedding_dim
                    ) * 0.02
                
                # Replace the embedding layer (update the parent module)
                parent_modules = []
                attr_name = None
                for name, module in self.text_encoder.named_modules():
                    if module is embedding_layer:
                        parent_path = name.split('.')
                        attr_name = parent_path[-1]
                        parent = self.text_encoder
                        for attr in parent_path[:-1]:
                            parent = getattr(parent, attr)
                        setattr(parent, attr_name, new_embedding_layer)
                        break
                
                token_embeds = new_embedding_layer.weight.data
                self.text_encoder_embedding_layer = new_embedding_layer  # Keep reference
        
        self.args.instance_prompt = "a photo of " + " and ".join(
            self.placeholder_tokens
        )

        if len(self.args.initializer_tokens) > 0 and token_embeds is not None:
            # Use initializer tokens
            for tkn_idx, initializer_token in enumerate(self.args.initializer_tokens):
                curr_token_ids = self.tokenizer.encode(
                    initializer_token, add_special_tokens=False
                )
                if len(curr_token_ids) > 0:
                    token_embeds[self.placeholder_token_ids[tkn_idx]] = token_embeds[
                        curr_token_ids[0]
                    ]

        # Set validation scheduler for logging
        self.validation_scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.validation_scheduler.set_timesteps(50)

        # We start by only optimizing the embeddings
        self.vae.requires_grad_(False)
        self.unet.requires_grad_(False)
        
        # Freeze all parameters except for the token embeddings in text encoder
        # Handle MVDream's FrozenOpenCLIPEmbedder vs standard transformers
        if hasattr(self.text_encoder, 'text_model'):
            # Standard transformers text encoder
            self.text_encoder.text_model.encoder.requires_grad_(False)
            self.text_encoder.text_model.final_layer_norm.requires_grad_(False)
            self.text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
        else:
            # MVDream's FrozenOpenCLIPEmbedder - freeze everything except token embeddings
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            # Only allow token embeddings to be trainable
            if hasattr(self, 'text_encoder_embedding_layer'):
                self.text_encoder_embedding_layer.weight.requires_grad = True

        if self.args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError(
                    "xformers is not available. Make sure it is installed correctly"
                )

        if self.args.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            if self.args.train_text_encoder:
                self.text_encoder.gradient_checkpointing_enable()

        if self.args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if self.args.scale_lr:
            self.args.learning_rate = (
                self.args.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.train_batch_size
                * self.accelerator.num_processes
            )

        if self.args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        # We start by only optimizing the embeddings
        if hasattr(self.text_encoder, 'get_input_embeddings'):
            # Standard transformers text encoder
            params_to_optimize = self.text_encoder.get_input_embeddings().parameters()
        else:
            # MVDream's FrozenOpenCLIPEmbedder
            if hasattr(self, 'text_encoder_embedding_layer'):
                params_to_optimize = [self.text_encoder_embedding_layer.weight]
            else:
                # Fallback: optimize all text encoder parameters if we couldn't find embeddings
                params_to_optimize = self.text_encoder.parameters()
            
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.args.initial_learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        # In phase 1 (textual inversion), we start with use_multiview=False
        # In phase 2, we switch to use_multiview=True if view_mode is "multi"
        # The view_mode parameter controls what data is loaded:
        #   - "multi": Load all views (original behavior)
        #   - "single": Load only the specified view_index
        use_multiview = False  # Start with False for phase 1, will switch in phase 2

        self.train_dataset = DreamBoothDataset(
            instance_data_root=self.args.instance_data_dir,
            placeholder_tokens=self.placeholder_tokens,
            class_data_root=self.args.class_data_dir
            if self.args.with_prior_preservation
            else None,
            class_prompt=self.args.class_prompt,
            tokenizer=self.tokenizer,
            size=self.args.resolution,
            center_crop=self.args.center_crop,
            num_of_assets=self.args.num_of_assets,
            num_frames=self.args.num_frames,
            use_multiview=use_multiview,
            view_index=self.args.view_index,
            view_mode=self.args.view_mode,
            concept_names=self.args.concept_names if hasattr(self.args, 'concept_names') else None,
        )
        train_dataset = self.train_dataset

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=lambda examples: collate_fn(
                examples, self.args.with_prior_preservation
            ),
            num_workers=self.args.dataloader_num_workers,
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if self.args.max_train_steps is None:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.args.lr_warmup_steps
            * self.args.gradient_accumulation_steps,
            num_training_steps=self.args.max_train_steps
            * self.args.gradient_accumulation_steps,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

        (
            self.unet,
            self.text_encoder,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = self.accelerator.prepare(
            self.unet, self.text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16

        # Move vae and text_encoder to device and cast to weight_dtype
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)

        low_precision_error_string = (
            "Please make sure to always have all model weights in full float32 precision when starting training - even if"
            " doing mixed precision training. copy of the weights should still be float32."
        )

        if self.accelerator.unwrap_model(self.unet).dtype != torch.float32:
            raise ValueError(
                f"Unet loaded as datatype {self.accelerator.unwrap_model(self.unet).dtype}. {low_precision_error_string}"
            )

        if self.args.train_text_encoder:
            # Check dtype for text encoder - handle MVDream's different structure
            text_encoder = self.accelerator.unwrap_model(self.text_encoder)
            if hasattr(text_encoder, 'dtype'):
                # Standard transformers text encoder
                if text_encoder.dtype != torch.float32:
                    raise ValueError(
                        f"Text encoder loaded as datatype {text_encoder.dtype}."
                        f" {low_precision_error_string}"
                    )
            else:
                # MVDream's FrozenOpenCLIPEmbedder - check a parameter instead
                first_param = next(text_encoder.parameters(), None)
                if first_param is not None and first_param.dtype != torch.float32:
                    raise ValueError(
                        f"Text encoder loaded as datatype {first_param.dtype}."
                        f" {low_precision_error_string}"
                    )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / self.args.gradient_accumulation_steps
        )
        if overrode_max_train_steps:
            self.args.max_train_steps = (
                self.args.num_train_epochs * num_update_steps_per_epoch
            )
        # Afterwards we recalculate our number of training epochs
        self.args.num_train_epochs = math.ceil(
            self.args.max_train_steps / num_update_steps_per_epoch
        )

        if len(self.args.initializer_tokens) > 0:
            # Only for logging
            self.args.initializer_tokens = ", ".join(self.args.initializer_tokens)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if self.accelerator.is_main_process:
            tracker_config = vars(self.args)

            # Initialize wandb with custom project and run name if using wandb
            if "wandb" in self.args.report_to.lower():
                # Import wandb to set monitoring settings
                import wandb as wandb_module

                # Configure wandb settings for GPU monitoring
                wandb_settings = wandb_module.Settings()
                # Monitor GPU/CPU/memory every second in the background
                os.environ["WANDB_WATCH"] = "all"

                init_kwargs = {
                    "wandb": {
                        "name": self.args.wandb_run_name if self.args.wandb_run_name else None,
                        "settings": wandb_settings
                    }
                }

                # Use wandb_project as the project name (first param to init_trackers)
                project_name = self.args.wandb_project if self.args.wandb_project else "dreambooth"
                self.accelerator.init_trackers(project_name, config=tracker_config, init_kwargs=init_kwargs)

                # Enable system monitoring (GPU, CPU, memory, etc.) - this runs in background
                print("Wandb system monitoring enabled (GPU, CPU, memory tracking)")
            else:
                self.accelerator.init_trackers("dreambooth", config=tracker_config)

        # Train
        total_batch_size = (
            self.args.train_batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {self.args.num_train_epochs}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.train_batch_size}"
        )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {self.args.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint:
            if self.args.resume_from_checkpoint != "latest":
                path = os.path.basename(self.args.resume_from_checkpoint)
            else:
                # Get the mos recent checkpoint
                dirs = os.listdir(self.args.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                self.accelerator.print(
                    f"Checkpoint '{self.args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                self.args.resume_from_checkpoint = None
            else:
                self.accelerator.print(f"Resuming from checkpoint {path}")
                self.accelerator.load_state(os.path.join(self.args.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.args.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (
                    num_update_steps_per_epoch * self.args.gradient_accumulation_steps
                )

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(
            range(global_step, self.args.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
        )
        progress_bar.set_description("Steps")

        # For tracking step time
        step_start_time = time.time()

        # keep original embeddings as reference
        if hasattr(self.text_encoder, 'get_input_embeddings'):
            # Standard transformers text encoder
            orig_embeds_params = (
                self.accelerator.unwrap_model(self.text_encoder)
                .get_input_embeddings()
                .weight.data.clone()
            )
        else:
            # MVDream's FrozenOpenCLIPEmbedder
            if hasattr(self, 'text_encoder_embedding_layer'):
                orig_embeds_params = (
                    self.accelerator.unwrap_model(self.text_encoder_embedding_layer)
                    .weight.data.clone()
                )
            else:
                # If we can't find embeddings, create a dummy tensor
                orig_embeds_params = torch.zeros(1)

        # Create attention controller (only for standard UNets)
        self.controller = AttentionStore()
        self.has_attention_control = self.register_attention_control(self.controller)

        for epoch in range(first_epoch, self.args.num_train_epochs):
            self.unet.train()
            if self.args.train_text_encoder:
                self.text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                # Switch to multi-view mode at the start of phase 2 (only if view_mode is "multi")
                if self.args.phase1_train_steps == global_step:
                    if self.args.view_mode == "multi":
                        print(f"\n{'='*50}")
                        print(f"PHASE 2: Switching to MULTI-VIEW mode - all {self.args.num_frames} views will be processed together!")
                        print(f"{'='*50}\n")
                        # Enable multi-view mode in the dataset
                        self.train_dataset.use_multiview = True
                    else:
                        print(f"\n{'='*50}")
                        print(f"PHASE 2: Continuing with SINGLE-VIEW mode (view {self.args.view_index + 1})")
                        print(f"{'='*50}\n")
                        # Keep single-view mode
                        self.train_dataset.use_multiview = False

                if self.args.phase1_train_steps == global_step:
                    self.unet.requires_grad_(True)
                    if self.args.train_text_encoder:
                        self.text_encoder.requires_grad_(True)
                    unet_params = self.unet.parameters()

                    if self.args.train_text_encoder:
                        params_to_optimize = itertools.chain(unet_params, self.text_encoder.parameters())
                    else:
                        # Only optimize UNet + token embeddings
                        if hasattr(self.text_encoder, 'get_input_embeddings'):
                            # Standard transformers text encoder
                            text_params = self.text_encoder.get_input_embeddings().parameters()
                        else:
                            # MVDream's FrozenOpenCLIPEmbedder
                            if hasattr(self, 'text_encoder_embedding_layer'):
                                text_params = [self.text_encoder_embedding_layer.weight]
                            else:
                                text_params = []
                        params_to_optimize = itertools.chain(unet_params, text_params)
                        
                    del optimizer
                    optimizer = optimizer_class(
                        params_to_optimize,
                        lr=self.args.learning_rate,
                        betas=(self.args.adam_beta1, self.args.adam_beta2),
                        weight_decay=self.args.adam_weight_decay,
                        eps=self.args.adam_epsilon,
                    )
                    del lr_scheduler
                    lr_scheduler = get_scheduler(
                        self.args.lr_scheduler,
                        optimizer=optimizer,
                        num_warmup_steps=self.args.lr_warmup_steps
                        * self.args.gradient_accumulation_steps,
                        num_training_steps=self.args.max_train_steps
                        * self.args.gradient_accumulation_steps,
                        num_cycles=self.args.lr_num_cycles,
                        power=self.args.lr_power,
                    )
                    optimizer, lr_scheduler = self.accelerator.prepare(
                        optimizer, lr_scheduler
                    )

                logs = {}

                # Skip steps until we reach the resumed step
                if (
                    self.args.resume_from_checkpoint
                    and epoch == first_epoch
                    and step < resume_step
                ):
                    if step % self.args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                    continue

                # Determine training mode for this step
                use_3d_mode = self.should_use_3d_mode(global_step)

                with self.accelerator.accumulate(self.unet):
                    # Handle multi-view input reshaping
                    pixel_values = batch["pixel_values"]
                    original_shape = pixel_values.shape
                    is_multiview_batch = pixel_values.dim() == 5  # [batch, frames, C, H, W]

                    if is_multiview_batch:
                        # Multi-view data: [batch, frames, C, H, W]
                        batch_size, num_frames = pixel_values.shape[:2]
                        # Reshape to [batch * frames, C, H, W] to process all views together
                        pixel_values = pixel_values.view(-1, *pixel_values.shape[2:])
                        # print(f"Processing {batch_size} batches × {num_frames} views = {batch_size * num_frames} total images")
                    else:
                        # Single-view data: [batch, C, H, W]
                        batch_size = pixel_values.shape[0]
                        num_frames = 1
                    
                    # Convert images to latent space
                    vae_output = self.vae.encode(
                        pixel_values.to(dtype=self.weight_dtype)
                    )
                    
                    # Handle different VAE output formats
                    if hasattr(vae_output, 'latent_dist'):
                        # Standard diffusers VAE
                        latents = vae_output.latent_dist.sample()
                    else:
                        # MVDream VAE returns DiagonalGaussianDistribution directly
                        latents = vae_output.sample()
                    
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0,
                        self.noise_scheduler.config.num_train_timesteps,
                        (bsz,),
                        device=latents.device,
                    )
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = self.noise_scheduler.add_noise(
                        latents, noise, timesteps
                    )

                    # Get the text embedding for conditioning
                    # MVDream's text encoder expects text strings, not token IDs
                    if hasattr(self.text_encoder, 'encode_with_transformer'):
                        # MVDream's FrozenOpenCLIPEmbedder - convert token IDs back to text
                        batch_texts = []
                        for input_ids in batch["input_ids"]:
                            # Remove padding and special tokens, then decode
                            valid_ids = input_ids[(input_ids != 0) & (input_ids != 49406) & (input_ids != 49407)]
                            text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                            batch_texts.append(text)
                        encoder_hidden_states = self.text_encoder(batch_texts)
                    else:
                        # Standard transformers text encoder
                        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

                    # CRITICAL: If we reshaped pixel_values to [batch*frames, ...], we must repeat text embeddings
                    if is_multiview_batch:
                        # encoder_hidden_states is [original_batch, seq_len, dim]
                        # We need to repeat it num_frames times to match [batch*frames, seq_len, dim]
                        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)
                        # Now it's [batch*frames, seq_len, dim]

                    # Prepare conditioning for MVDream
                    if use_3d_mode:
                        # Generate camera matrices for training
                        if self.args.use_camera:
                            camera = get_camera(self.args.num_frames, elevation=self.args.camera_elev, 
                                              azimuth_start=self.args.camera_azim, azimuth_span=self.args.camera_azim_span)
                        else:
                            camera = None
                        
                        # Ensure batch size is compatible with num_frames
                        original_batch_size = bsz
                        if bsz % self.args.num_frames != 0:
                            # For MVDream, we need batch size divisible by num_frames
                            # Pad or reshape the batch to be compatible
                            target_batch_size = ((bsz // self.args.num_frames) + 1) * self.args.num_frames
                            pad_size = target_batch_size - bsz
                            
                            if pad_size > 0:
                                # Pad tensors to make batch size divisible by num_frames
                                noisy_latents = torch.cat([noisy_latents, noisy_latents[-pad_size:]], dim=0)
                                timesteps = torch.cat([timesteps, timesteps[-pad_size:]], dim=0)
                                encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states[-pad_size:]], dim=0)
                                bsz = target_batch_size
                        
                        # Size camera tensor to match batch size
                        if camera is not None:
                            # For MVDream, camera tensor should have exactly bsz entries
                            # If bsz is divisible by num_frames, repeat each camera matrix appropriately
                            cameras_per_frame = bsz // self.args.num_frames
                            if bsz % self.args.num_frames == 0:
                                # Repeat each of the 4 camera matrices cameras_per_frame times
                                camera = camera.repeat_interleave(cameras_per_frame, dim=0).to(noisy_latents.device)
                            else:
                                # If not divisible, just use the first bsz cameras
                                camera = camera[:bsz].to(noisy_latents.device)
                        
                        # Prepare MVDream conditioning
                        cond = {
                            "context": encoder_hidden_states,
                            "camera": camera,
                            "num_frames": self.args.num_frames
                        }
                        
                        # Clear cache before heavy 3D computation
                        torch.cuda.empty_cache()
                        
                        # Use MVDream's apply_model method
                        try:
                            model_pred = self.mvdream_model.apply_model(noisy_latents, timesteps, cond=cond)
                        except RuntimeError as e:
                            if "out of memory" in str(e):
                                print(f"Warning: OOM in 3D mode, falling back to 2D mode for this step")
                                torch.cuda.empty_cache()
                                # Fallback to 2D mode for this step
                                unet_output = self.unet(
                                    noisy_latents[:original_batch_size], 
                                    timesteps[:original_batch_size], 
                                    encoder_hidden_states[:original_batch_size]
                                )
                                if hasattr(unet_output, 'sample'):
                                    model_pred = unet_output.sample
                                else:
                                    model_pred = unet_output
                            else:
                                raise e
                        
                        # If we padded the batch, remove the padding from the output
                        if bsz > original_batch_size:
                            model_pred = model_pred[:original_batch_size]
                    else:
                        # Standard 2D mode - use regular UNet
                        unet_output = self.unet(
                            noisy_latents, timesteps, encoder_hidden_states
                        )
                        # Handle different UNet output formats
                        if hasattr(unet_output, 'sample'):
                            model_pred = unet_output.sample
                        else:
                            model_pred = unet_output

                    # Get the target for loss depending on the prediction type
                    if self.noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif self.noise_scheduler.config.prediction_type == "v_prediction":
                        target = self.noise_scheduler.get_velocity(
                            latents, noise, timesteps
                        )
                    else:
                        raise ValueError(
                            f"Unknown prediction type {self.noise_scheduler.config.prediction_type}"
                        )
                    
                    # Handle padding adjustment for target as well
                    if use_3d_mode and model_pred.shape[0] != target.shape[0]:
                        # If we had to pad for MVDream, handle target accordingly
                        if model_pred.shape[0] > target.shape[0]:
                            # Pad target to match model_pred
                            pad_size = model_pred.shape[0] - target.shape[0]
                            target = torch.cat([target, target[-pad_size:]], dim=0)
                        # Then trim both to original size for loss computation
                        original_batch_size = latents.shape[0]
                        model_pred = model_pred[:original_batch_size]
                        target = target[:original_batch_size]

                    if self.args.with_prior_preservation:
                        # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                        model_pred_prior, model_pred = torch.chunk(model_pred, 2, dim=0)
                        target_prior, target = torch.chunk(target, 2, dim=0)

                        if self.args.apply_masked_loss:
                            instance_masks = batch["instance_masks"]

                            # Handle multi-view masks: [batch, frames, tokens, 1, 1, H, W] or [batch, tokens, 1, 1, H, W]
                            if instance_masks.dim() == 7:  # Multi-view: [batch, frames, tokens, 1, 1, H, W]
                                # Reshape to [batch * frames, tokens, 1, 1, H, W]
                                instance_masks = instance_masks.view(-1, *instance_masks.shape[2:])
                            elif instance_masks.dim() == 6 and is_multiview_batch:  # [batch, frames, tokens, 1, H, W]
                                # Current: [batch, frames, tokens, 1, H, W] = [1, 4, 2, 1, 256, 256]
                                # Need: [batch*frames, tokens, 1, H, W] = [4, 2, 1, 256, 256]
                                batch_size, num_frames = instance_masks.shape[0], instance_masks.shape[1]
                                instance_masks = instance_masks.view(batch_size * num_frames, *instance_masks.shape[2:])

                            # instance_masks is now [batch*frames, tokens, 1, 1, H, W] or [batch, tokens, 1, 1, H, W]
                            # Take max over tokens dimension (dim=1)
                            max_masks = torch.max(instance_masks, dim=1).values  # [batch*frames, 1, 1, H, W]
                            # Squeeze to get [batch*frames, H, W] then unsqueeze to [batch*frames, 1, H, W] for interpolation
                            max_masks = max_masks.squeeze(1).squeeze(1).unsqueeze(1)

                            latent_size = self.args.resolution // 8
                            downsampled_mask = F.interpolate(
                                input=max_masks, size=(latent_size, latent_size)
                            )
                            model_pred = model_pred * downsampled_mask
                            target = target * downsampled_mask

                        # Compute instance loss
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                        # Compute prior loss
                        prior_loss = F.mse_loss(
                            model_pred_prior.float(),
                            target_prior.float(),
                            reduction="mean",
                        )

                        # Add the prior loss to the instance loss.
                        loss = loss + self.args.prior_loss_weight * prior_loss
                    else:
                        if self.args.apply_masked_loss:
                            instance_masks = batch["instance_masks"]

                            # Handle multi-view masks: [batch, frames, tokens, 1, 1, H, W] or [batch, tokens, 1, 1, H, W]
                            if instance_masks.dim() == 7:  # Multi-view: [batch, frames, tokens, 1, 1, H, W]
                                # Reshape to [batch * frames, tokens, 1, 1, H, W]
                                instance_masks = instance_masks.view(-1, *instance_masks.shape[2:])
                            elif instance_masks.dim() == 6 and is_multiview_batch:  # [batch, frames, tokens, 1, H, W]
                                # Current: [batch, frames, tokens, 1, H, W] = [1, 4, 2, 1, 256, 256]
                                # Need: [batch*frames, tokens, 1, H, W] = [4, 2, 1, 256, 256]
                                batch_size, num_frames = instance_masks.shape[0], instance_masks.shape[1]
                                instance_masks = instance_masks.view(batch_size * num_frames, *instance_masks.shape[2:])

                            # instance_masks is now [batch*frames, tokens, 1, 1, H, W] or [batch, tokens, 1, 1, H, W]
                            # Take max over tokens dimension (dim=1)
                            max_masks = torch.max(instance_masks, dim=1).values  # [batch*frames, 1, 1, H, W]
                            # Squeeze to get [batch*frames, H, W] then unsqueeze to [batch*frames, 1, H, W] for interpolation
                            max_masks = max_masks.squeeze(1).squeeze(1).unsqueeze(1)

                            latent_size = self.args.resolution // 8
                            downsampled_mask = F.interpolate(
                                input=max_masks, size=(latent_size, latent_size)
                            )
                            model_pred = model_pred * downsampled_mask
                            target = target * downsampled_mask
                        loss = F.mse_loss(
                            model_pred.float(), target.float(), reduction="mean"
                        )

                    # Attention loss (only in 2D mode and if attention control is available)
                    if self.args.lambda_attention != 0 and not use_3d_mode and self.has_attention_control:
                        attn_loss = 0
                        for batch_idx in range(self.args.train_batch_size):
                            instance_masks = batch["instance_masks"]
                            GT_masks = F.interpolate(
                                input=instance_masks[batch_idx], size=(16, 16)
                            )
                            agg_attn = self.aggregate_attention(
                                res=16,
                                from_where=("up", "down"),
                                is_cross=True,
                                select=batch_idx,
                            )
                            curr_cond_batch_idx = self.args.train_batch_size + batch_idx

                            for mask_id in range(len(GT_masks)):
                                curr_placeholder_token_id = self.placeholder_token_ids[
                                    batch["token_ids"][batch_idx][mask_id]
                                ]

                                asset_idx = (
                                    (
                                        batch["input_ids"][curr_cond_batch_idx]
                                        == curr_placeholder_token_id
                                    )
                                    .nonzero()
                                    .item()
                                )
                                asset_attn_mask = agg_attn[..., asset_idx]
                                asset_attn_mask = (
                                    asset_attn_mask / asset_attn_mask.max()
                                )
                                attn_loss += F.mse_loss(
                                    GT_masks[mask_id, 0].float(),
                                    asset_attn_mask.float(),
                                    reduction="mean",
                                )

                        attn_loss = self.args.lambda_attention * (
                            attn_loss / self.args.train_batch_size
                        )
                        logs["attn_loss"] = attn_loss.detach().item()
                        loss += attn_loss

                    self.accelerator.backward(loss)

                    # No need to keep the attention store
                    if self.has_attention_control:
                        self.controller.attention_store = {}
                        self.controller.cur_step = 0

                    if self.accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(
                                self.unet.parameters(), self.text_encoder.parameters()
                            )
                            if self.args.train_text_encoder
                            else self.unet.parameters()
                        )
                        self.accelerator.clip_grad_norm_(
                            params_to_clip, self.args.max_grad_norm
                        )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=self.args.set_grads_to_none)

                    if global_step < self.args.phase1_train_steps:
                        # Let's make sure we don't update any embedding weights besides the newly added token
                        with torch.no_grad():
                            if hasattr(self.text_encoder, 'get_input_embeddings'):
                                # Standard transformers text encoder
                                self.accelerator.unwrap_model(
                                    self.text_encoder
                                ).get_input_embeddings().weight[
                                    : -self.args.num_of_assets
                                ] = orig_embeds_params[
                                    : -self.args.num_of_assets
                                ]
                            elif hasattr(self, 'text_encoder_embedding_layer') and orig_embeds_params.numel() > 1:
                                # MVDream's FrozenOpenCLIPEmbedder
                                self.accelerator.unwrap_model(
                                    self.text_encoder_embedding_layer
                                ).weight[
                                    : -self.args.num_of_assets
                                ] = orig_embeds_params[
                                    : -self.args.num_of_assets
                                ]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if global_step % self.args.checkpointing_steps == 0:
                        if self.accelerator.is_main_process:
                            save_path = os.path.join(
                                self.args.output_dir, f"checkpoint-{global_step}"
                            )
                            self.accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                    if (
                        self.args.log_checkpoints
                        and global_step % self.args.img_log_steps == 0
                        and global_step > self.args.phase1_train_steps
                    ):
                        ckpts_path = os.path.join(
                            self.args.output_dir, "checkpoints", f"{global_step:05}"
                        )
                        os.makedirs(ckpts_path, exist_ok=True)
                        self.save_pipeline(ckpts_path)

                        img_logs_path = os.path.join(self.args.output_dir, "img_logs")
                        os.makedirs(img_logs_path, exist_ok=True)

                        if self.args.lambda_attention != 0 and not use_3d_mode and self.has_attention_control:
                            self.controller.cur_step = 1
                            last_sentence = batch["input_ids"][curr_cond_batch_idx]
                            last_sentence = last_sentence[
                                (last_sentence != 0)
                                & (last_sentence != 49406)
                                & (last_sentence != 49407)
                            ]
                            last_sentence = self.tokenizer.decode(last_sentence)
                            self.save_cross_attention_vis(
                                last_sentence,
                                attention_maps=agg_attn.detach().cpu(),
                                path=os.path.join(
                                    img_logs_path, f"{global_step:05}_step_attn.jpg"
                                ),
                            )
                        
                        if self.has_attention_control:
                            self.controller.cur_step = 0
                            self.controller.attention_store = {}

                        self.perform_full_inference(
                            path=os.path.join(
                                img_logs_path, f"{global_step:05}_full_pred.jpg"
                            )
                        )
                        if not use_3d_mode and self.has_attention_control:
                            full_agg_attn = self.aggregate_attention(
                                res=16, from_where=("up", "down"), is_cross=True, select=0
                            )
                            self.save_cross_attention_vis(
                                self.args.instance_prompt,
                                attention_maps=full_agg_attn.detach().cpu(),
                                path=os.path.join(
                                    img_logs_path, f"{global_step:05}_full_attn.jpg"
                                ),
                            )
                        
                        if self.has_attention_control:
                            self.controller.cur_step = 0
                            self.controller.attention_store = {}

                # Calculate step time
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                logs["loss"] = loss.detach().item()
                logs["lr"] = lr_scheduler.get_last_lr()[0]
                logs["mode"] = "3D" if use_3d_mode else "2D"
                logs["step_time_sec"] = step_time

                # Calculate throughput
                if step_time > 0:
                    logs["samples_per_sec"] = self.args.train_batch_size / step_time

                # Note: GPU metrics are automatically tracked by wandb in the background
                # No need to manually sample them here - wandb monitors continuously

                progress_bar.set_postfix(**logs)
                self.accelerator.log(logs, step=global_step)

                # Reset step timer
                step_start_time = time.time()

                if global_step >= self.args.max_train_steps:
                    break
                
        self.save_pipeline(self.args.output_dir)
        self.accelerator.end_training()

    def should_use_3d_mode(self, global_step):
        """Determine whether to use 3D attention mode based on training strategy."""
        # Always use 2D mode in phase 1 (textual inversion)
        if global_step < self.args.phase1_train_steps:
            return False
            
        # In phase 2, use the specified training mode but with reduced probability to save memory
        if self.args.mvdream_training_mode == "2d":
            return False
        elif self.args.mvdream_training_mode == "3d":
            return True
        elif self.args.mvdream_training_mode == "mixed":
            # Use 3D mode 30% of the time instead of 70% to reduce memory usage
            return random.random() < 0.3
        else:
            return False

    def save_pipeline(self, path):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            # Save the complete MVDream model state (contains UNet + Text Encoder + all components)
            torch.save(self.mvdream_model.state_dict(), os.path.join(path, "mvdream_model.pth"))

            # Save tokenizer (needed for inference)
            if self.tokenizer is not None:
                self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

            # Note: Individual component saves (unet.pth, text_encoder.pth) removed to save ~4.7GB disk space
            # The complete mvdream_model.pth contains all necessary weights for inference

    def register_attention_control(self, controller):
        """Register attention control if the UNet supports it. Returns True if successful."""
        if not hasattr(self.unet, 'attn_processors'):
            print("UNet does not support attention processors (MVDream UNet). Skipping attention control.")
            return False
            
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            cross_attention_dim = (
                None
                if name.endswith("attn1.processor")
                else self.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = self.unet.config.block_out_channels[-1]
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.unet.config.block_out_channels))[
                    block_id
                ]
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.unet.config.block_out_channels[block_id]
                place_in_unet = "down"
            else:
                continue
            cross_att_count += 1
            attn_procs[name] = P2PCrossAttnProcessor(
                controller=controller, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        controller.num_att_layers = cross_att_count
        return True

    def get_average_attention(self):
        average_attention = {
            key: [
                item / self.controller.cur_step
                for item in self.controller.attention_store[key]
            ]
            for key in self.controller.attention_store
        }
        return average_attention

    def aggregate_attention(
        self, res: int, from_where: List[str], is_cross: bool, select: int
    ):
        out = []
        attention_maps = self.get_average_attention()
        num_pixels = res**2
        for location in from_where:
            for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(
                        self.args.train_batch_size, -1, res, res, item.shape[-1]
                    )[select]
                    out.append(cross_maps)
        out = torch.cat(out, dim=0)
        out = out.sum(0) / out.shape[0]
        return out

    @torch.no_grad()
    def perform_full_inference(self, path, guidance_scale=7.5):
        self.unet.eval()
        self.text_encoder.eval()

        latent_size = self.args.resolution // 8
        latents = torch.randn((1, 4, latent_size, latent_size), device=self.accelerator.device)
        
        # Handle different text encoder types
        if hasattr(self.text_encoder, 'encode_with_transformer'):
            # MVDream's FrozenOpenCLIPEmbedder - pass text directly
            cond_embeddings = self.text_encoder([self.args.instance_prompt])
            uncond_embeddings = self.text_encoder([""])
        else:
            # Standard transformers text encoder
            uncond_input = self.tokenizer(
                [""],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).to(self.accelerator.device)
            input_ids = self.tokenizer(
                [self.args.instance_prompt],
                padding="max_length",
                truncation=True,
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids.to(self.accelerator.device)
            cond_embeddings = self.text_encoder(input_ids)[0]
            uncond_embeddings = self.text_encoder(uncond_input.input_ids)[0]
            
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        for t in self.validation_scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.validation_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )
            # Handle different UNet output formats
            if hasattr(pred, 'sample'):
                noise_pred = pred.sample
            else:
                noise_pred = pred

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            latents = self.validation_scheduler.step(noise_pred, t, latents).prev_sample
        latents = 1 / 0.18215 * latents

        vae_output = self.vae.decode(latents.to(self.weight_dtype))
        # Handle different VAE output formats
        if hasattr(vae_output, 'sample'):
            images = vae_output.sample
        else:
            images = vae_output
            
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")

        self.unet.train()
        if self.args.train_text_encoder:
            self.text_encoder.train()

        Image.fromarray(images[0]).save(path)

    @torch.no_grad()
    def save_cross_attention_vis(self, prompt, attention_maps, path):
        tokens = self.tokenizer.encode(prompt)
        images = []
        for i in range(len(tokens)):
            image = attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = ptp_utils.text_under_image(
                image, self.tokenizer.decode(int(tokens[i]))
            )
            images.append(image)
        vis = ptp_utils.view_images(np.stack(images, axis=0))
        vis.save(path)


class P2PCrossAttnProcessor:
    def __init__(self, controller, place_in_unet):
        super().__init__()
        self.controller = controller
        self.place_in_unet = place_in_unet

    def __call__(
        self,
        attn: CrossAttention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # one line change
        self.controller(attention_probs, is_cross, self.place_in_unet)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


if __name__ == "__main__":
    SpatialDreambooth()