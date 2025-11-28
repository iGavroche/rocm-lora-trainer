import argparse
import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_Z_IMAGE,
    ItemInfo,
    save_latent_cache_z_image,
)
from musubi_tuner.z_image import z_image_utils
from musubi_tuner.z_image.z_image_model import load_z_image_vae
from musubi_tuner.utils.device_utils import synchronize_device
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_z_image(batch: List[ItemInfo]) -> torch.Tensor:
    """
    Preprocess images for Z-Image VAE encoding.
    item.content: target image (H, W, C) in 0-255 range
    
    Returns:
        Tensor of shape (B, C, H, W) normalized to [-1, 1]
    """
    contents = []
    for item in batch:
        contents.append(torch.from_numpy(item.content))  # target image (H, W, C)

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    return contents


def encode_and_save_batch(vae, batch: List[ItemInfo]):
    """
    Encode images to latents and save cache for Z-Image.
    Includes ROCm-specific workarounds to prevent SIGSEGV.
    
    Args:
        vae: Z-Image VAE model (from diffusers pipeline)
        batch: List of ItemInfo objects
    """
    # Check if we're on ROCm
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    device = vae.device if hasattr(vae, 'device') else torch.device("cuda")
    
    # Ensure VAE is in eval mode
    vae.eval()
    
    # Get VAE dtype
    vae_dtype = vae.dtype if hasattr(vae, 'dtype') else torch.bfloat16
    
    # item.content: target image (H, W, C)
    contents = preprocess_contents_z_image(batch)  # B, C, H, W
    
    # ROCm workaround: synchronize before encoding
    if device.type == "cuda":
        synchronize_device(device)
        torch.cuda.empty_cache()  # Clear cache before encoding

    with torch.no_grad():
        # ROCm workaround: Use torch.amp.autocast() instead of letting VAE handle it
        # This prevents SIGSEGV from GPU queue evictions
        try:
            if is_rocm:
                # On ROCm, use explicit autocast with synchronization
                # Process on CPU first, then move to GPU with careful synchronization
                logger.debug(f"ROCm: Processing batch of {len(batch)} images")
                
                # Move contents to CPU first (ROCm workaround)
                contents_cpu = contents.cpu().float()
                synchronize_device(device)
                
                # Move to GPU with synchronization
                synchronize_device(device)
                contents_gpu = contents_cpu.to(device, dtype=vae_dtype)
                synchronize_device(device)
                
                # Try encode_pixels_to_latents first (Qwen-Image style interface)
                if hasattr(vae, "encode_pixels_to_latents"):
                    logger.debug("Using encode_pixels_to_latents method")
                    # Qwen-Image style needs frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
                    contents_with_frame = contents_gpu.unsqueeze(2)  # (B, C, 1, H, W)
                    synchronize_device(device)
                    latents = vae.encode_pixels_to_latents(contents_with_frame)
                    synchronize_device(device)
                elif hasattr(vae, "encode"):
                    logger.debug("Using encode() method - diffusers VAE expects 4D (B, C, H, W)")
                    # Standard diffusers encode() method expects 4D: (B, C, H, W)
                    # Do NOT add frame dimension for diffusers VAE
                    synchronize_device(device)
                    encoded = vae.encode(contents_gpu)  # Already 4D: (B, C, H, W)
                    synchronize_device(device)
                    # Extract latents from encoded output
                    # Diffusers VAE encode() typically returns a dict or LatentDistribution
                    if isinstance(encoded, dict):
                        if "latent_dist" in encoded:
                            latents = encoded["latent_dist"].sample()
                        elif "latents" in encoded:
                            latents = encoded["latents"]
                        else:
                            # Try to get the first tensor value
                            tensor_values = [v for v in encoded.values() if isinstance(v, torch.Tensor)]
                            if tensor_values:
                                latents = tensor_values[0]
                            else:
                                raise ValueError(f"Could not extract latents from encoded dict: {encoded.keys()}")
                    elif hasattr(encoded, "latent_dist"):
                        latents = encoded.latent_dist.sample()
                    elif hasattr(encoded, "latents"):
                        latents = encoded.latents
                    elif isinstance(encoded, tuple):
                        # If it's a tuple, take the first tensor element
                        tensor_elements = [e for e in encoded if isinstance(e, torch.Tensor)]
                        if tensor_elements:
                            latents = tensor_elements[0]
                        elif len(encoded) > 0:
                            latents = encoded[0]
                        else:
                            raise ValueError(f"Empty tuple returned from encode(): {encoded}")
                    elif isinstance(encoded, torch.Tensor):
                        latents = encoded
                    else:
                        raise ValueError(f"Unexpected encode() return type: {type(encoded)}, value: {encoded}")
                    synchronize_device(device)
                else:
                    raise ValueError("VAE does not have encode() or encode_pixels_to_latents() method")
                
                # Move latents to CPU immediately to free GPU memory
                synchronize_device(device)
                latents = latents.cpu()
                synchronize_device(device)
                
                # Diffusers VAE returns 4D latents (B, C, H, W), need to add frame dimension for Z-Image cache format
                # Z-Image cache expects (F, C, H, W) where F=1 for images
                if latents.dim() == 4:  # (B, C, H, W)
                    # Add frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
                    latents = latents.unsqueeze(2)  # (B, C, 1, H, W)
            else:
                # On CUDA, use standard approach
                if hasattr(vae, "encode_pixels_to_latents"):
                    # Qwen-Image style needs frame dimension
                    contents_with_frame = contents.unsqueeze(2)  # (B, C, 1, H, W)
                    latents = vae.encode_pixels_to_latents(contents_with_frame.to(vae.device, dtype=vae.dtype))
                elif hasattr(vae, "encode"):
                    # Diffusers VAE expects 4D: (B, C, H, W) - do NOT add frame dimension
                    encoded = vae.encode(contents.to(vae.device, dtype=vae.dtype))
                    # Handle different return types from encode()
                    if isinstance(encoded, dict):
                        if "latent_dist" in encoded:
                            latents = encoded["latent_dist"].sample()
                        elif "latents" in encoded:
                            latents = encoded["latents"]
                        else:
                            tensor_values = [v for v in encoded.values() if isinstance(v, torch.Tensor)]
                            if tensor_values:
                                latents = tensor_values[0]
                            else:
                                raise ValueError(f"Could not extract latents from encoded dict: {encoded.keys()}")
                    elif hasattr(encoded, "latent_dist"):
                        latents = encoded.latent_dist.sample()
                    elif hasattr(encoded, "latents"):
                        latents = encoded.latents
                    elif isinstance(encoded, tuple):
                        tensor_elements = [e for e in encoded if isinstance(e, torch.Tensor)]
                        if tensor_elements:
                            latents = tensor_elements[0]
                        elif len(encoded) > 0:
                            latents = encoded[0]
                        else:
                            raise ValueError(f"Empty tuple returned from encode()")
                    elif isinstance(encoded, torch.Tensor):
                        latents = encoded
                    else:
                        raise ValueError(f"Unexpected encode() return type: {type(encoded)}")
                else:
                    raise ValueError("VAE does not have encode() or encode_pixels_to_latents() method")
        except Exception as e:
            logger.error(f"Error during VAE encoding: {e}", exc_info=True)
            if is_rocm:
                logger.error("This may indicate a ROCm driver issue. Try processing smaller batches.")
            # Clean up GPU memory
            if device.type == "cuda":
                synchronize_device(device)
                torch.cuda.empty_cache()
            raise
        finally:
            # ROCm workaround: synchronize after encoding and clear cache
            if device.type == "cuda":
                synchronize_device(device)
                torch.cuda.empty_cache()

    # Z-Image cache expects (F, C, H, W) format where F is frames
    # For images, F=1, so we need to ensure latents have frame dimension
    if latents.dim() == 4 and latents.shape[1] != 1:
        # If latents are (B, C, H, W), add frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
        if latents.shape[1] > 1:  # C dimension
            latents = latents.unsqueeze(2)  # Add frame dimension
    elif latents.dim() == 3:
        # If latents are (B, H, W), add channel and frame dimensions
        latents = latents.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)

    # save cache for each item in the batch
    for b, item in enumerate(batch):
        # Extract single image latent: (B, C, F, H, W) -> (C, F, H, W) for single item
        # Or if already (F, C, H, W), just take [b]
        if latents.dim() == 5:  # (B, C, F, H, W)
            target_latent = latents[b]  # (C, F, H, W)
            # Transpose to (F, C, H, W) format expected by save_latent_cache_z_image
            target_latent = target_latent.permute(1, 0, 2, 3)  # (F, C, H, W)
        elif latents.dim() == 4:
            # Already in (B, C, H, W) or (B, F, C, H, W) format
            if latents.shape[1] == 1:  # Frame dimension exists
                target_latent = latents[b]  # (F, C, H, W) or (C, H, W)
                if target_latent.dim() == 3:  # (C, H, W), add frame dimension
                    target_latent = target_latent.unsqueeze(0)  # (1, C, H, W)
            else:
                # (B, C, H, W) -> add frame dimension
                target_latent = latents[b].unsqueeze(0)  # (1, C, H, W)
        else:
            raise ValueError(f"Unexpected latent shape: {latents.shape}")

        logger.info(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}, "
            f"latents shape: {target_latent.shape}"
        )

        save_latent_cache_z_image(item_info=item, latent=target_latent)


def z_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Z-Image specific parser setup for latent caching"""
    # VAE path is handled by common parser (--vae)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to Z-Image model directory or HuggingFace model ID (e.g., Tongyi-MAI/Z-Image-Turbo). "
        "If provided, will load from pipeline. Auto-detects files in models/z-image/ if not specified.",
    )
    return parser


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    parser = z_image_setup_parser(parser)

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_Z_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    # Determine VAE path - check for local files first, then model_path, then HuggingFace
    import os
    
    model_path = getattr(args, "model_path", None)
    vae_path = args.vae
    
    # Check for local model files in models/z-image/
    local_model_dir = "models/z-image"
    local_vae_file = os.path.join(local_model_dir, "ae.safetensors")
    local_transformer_file = os.path.join(local_model_dir, "z_image_turbo_bf16.safetensors")
    local_text_encoder_file = os.path.join(local_model_dir, "qwen_3_4b.safetensors")
    
    # If local files exist, use them
    if os.path.exists(local_vae_file) and os.path.exists(local_transformer_file):
        logger.info(f"Found local Z-Image model files in {local_model_dir}")
        if model_path is None:
            model_path = local_model_dir
            logger.info(f"Using local model directory: {model_path}")
    elif model_path is None:
        # Try HuggingFace as fallback
        model_path = "Tongyi-MAI/Z-Image-Turbo"
        logger.info(f"No local model files found, will download from HuggingFace: {model_path}")
    
    if model_path is None and vae_path is None:
        raise ValueError("Either --model_path or --vae must be provided, or place model files in models/z-image/")

    logger.info(f"Loading VAE model from {model_path or vae_path}")
    
    # Try to load VAE using pipeline approach
    vae = None
    try:
        vae = load_z_image_vae(
            device=device,
            model_path=model_path,
            torch_dtype=torch.bfloat16,
        )
        vae.to(device)
        logger.info("Successfully loaded VAE from pipeline")
    except Exception as e:
        logger.error(f"Failed to load VAE from pipeline: {e}")
        raise ValueError(
            f"Failed to load VAE. Please ensure:\n"
            f"  1. Model files are in {local_model_dir}/ (ae.safetensors, z_image_turbo_bf16.safetensors, qwen_3_4b.safetensors), OR\n"
            f"  2. diffusers is installed and can download from HuggingFace (Tongyi-MAI/Z-Image-Turbo)\n"
            f"Error: {e}"
        )

    vae.eval()

    # Check if we're on ROCm
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    
    # ROCm workaround: Skip test encoding as it can cause SIGSEGV
    # The actual encoding will work fine, we just need to be careful with synchronization
    if is_rocm:
        logger.info("ROCm detected: Using ROCm-specific workarounds (single-image processing, extra synchronization)")
    
    # encoding closure
    def encode(batch: List[ItemInfo]):
        logger.info(f"Processing batch of {len(batch)} items")
        # ROCm workaround: ALWAYS process one image at a time to reduce GPU queue pressure
        # This is slower but prevents SIGSEGV on Strix Halo
        if is_rocm:
            logger.info(f"ROCm: Processing {len(batch)} images one at a time to prevent SIGSEGV")
            for idx, item in enumerate(batch):
                logger.info(f"ROCm: Processing item {idx+1}/{len(batch)}: {item.item_key}")
                try:
                    synchronize_device(device)
                    torch.cuda.empty_cache()
                    encode_and_save_batch(vae, [item])  # Process single item
                    synchronize_device(device)
                    torch.cuda.empty_cache()
                    logger.info(f"Successfully encoded {item.item_key}")
                except Exception as e:
                    logger.error(f"Failed to encode {item.item_key}: {e}", exc_info=True)
                    synchronize_device(device)
                    torch.cuda.empty_cache()
                    raise
        else:
            encode_and_save_batch(vae, batch)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()

