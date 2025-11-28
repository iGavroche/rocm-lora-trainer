"""
Z-Image image generation script for testing LoRA outputs.
Follows the pattern from qwen_image_generate_image.py but adapted for Z-Image's single-stream architecture.
"""

import argparse
import random
from typing import Optional

import torch
from safetensors.torch import load_file
from tqdm import tqdm

from musubi_tuner.z_image import z_image_model
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.networks import lora_z_image

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype


def parse_args() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Z-Image inference script")

    parser.add_argument("--dit", type=str, default=None, help="Z-Image transformer model path (deprecated, use --model_path)")
    parser.add_argument("--vae", type=str, default=None, help="VAE model path (deprecated, use --model_path)")
    parser.add_argument("--text_encoder", type=str, default=None, help="Text encoder (Qwen2.5-VL) path (deprecated, use --model_path)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Z-Image model directory or HuggingFace model ID (e.g., Tongyi-MAI/Z-Image-Turbo). If provided, loads from pipeline.")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s)")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier(s)")

    # Inference
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale (0.0 for Turbo models, no CFG)")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="Image size [height, width]")
    parser.add_argument("--num_inference_steps", type=int, default=8, help="Number of inference steps (default 8 for Turbo)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    # Model options
    parser.add_argument("--fp8_vl", action="store_true", help="Use fp8 for text encoder")
    parser.add_argument("--split_attn", action="store_true", help="Split attention for ROCm stability")

    return parser.parse_args()


def load_lora_weights(model, lora_paths: list[str], multipliers: list[float]):
    """Load and apply LoRA weights to the model"""
    if not lora_paths:
        return

    logger.info(f"Loading LoRA weights from {lora_paths}")
    for lora_path, multiplier in zip(lora_paths, multipliers):
        logger.info(f"Loading LoRA: {lora_path} with multiplier {multiplier}")
        lora_weights = load_file(lora_path)
        lora_weights = filter_lora_state_dict(lora_weights)

        # Create LoRA network and apply weights
        network = lora_z_image.create_arch_network_from_weights(
            multiplier=multiplier,
            weights_sd=lora_weights,
            text_encoders=None,
            unet=model,
            for_inference=True,
        )
        network.apply_to()
        logger.info(f"Applied LoRA from {lora_path}")


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
) -> torch.Tensor:
    """
    Main generation function for Z-Image.
    Uses ZImagePipeline for generation, with optional LoRA support.
    """
    device = gen_settings.device
    model_path = getattr(args, "model_path", None) or args.dit

    # Prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    logger.info(f"Using seed: {seed}")

    # Load pipeline
    logger.info(f"Loading Z-Image pipeline from {model_path}")
    pipe = z_image_model.load_z_image_pipeline(
        model_path=model_path,
        torch_dtype=torch.bfloat16,
        device=device,
        low_cpu_mem_usage=False,
    )

    # Load and apply LoRA if specified
    if args.lora_weight:
        logger.info("Loading LoRA weights")
        multipliers = args.lora_multiplier if isinstance(args.lora_multiplier, list) else [args.lora_multiplier] * len(args.lora_weight)
        load_lora_weights(pipe.transformer, args.lora_weight, multipliers)

    # Set attention backend if needed
    if args.split_attn:
        # For ROCm stability, keep SDPA (default)
        logger.info("Using SDPA attention (split_attn flag noted for ROCm stability)")

    height, width = args.image_size
    logger.info(f"Generating image: {height}x{width}, steps: {args.num_inference_steps}")

    # Generate using pipeline
    # Z-Image-Turbo uses guidance_scale=0.0 (no CFG)
    with torch.no_grad():
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt if args.negative_prompt else None,
            height=height,
            width=width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,  # 0.0 for Turbo
            generator=generator,
        ).images[0]

    # Convert PIL Image to tensor for return (if needed for further processing)
    # For now, we'll save it directly
    return image


def save_output(args: argparse.Namespace, image, device: torch.device):
    """Save generated image"""
    # image is already a PIL Image from the pipeline
    if isinstance(image, torch.Tensor):
        # If somehow we got a tensor, convert it
        from PIL import Image as PILImage
        import numpy as np
        if image.dim() == 4:
            image = image[0]
        image_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        image = PILImage.fromarray(image_np)
    
    image.save(args.output)
    logger.info(f"Saved image to {args.output}")


def main():
    args = parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")

    dit_weight_dtype = torch.bfloat16
    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)

    logger.info("Starting Z-Image generation")
    
    image = generate(args, gen_settings)
    save_output(args, image, device)

    logger.info("Generation complete")


if __name__ == "__main__":
    main()

