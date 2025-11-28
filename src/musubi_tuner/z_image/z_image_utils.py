import json
import logging
import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor
from transformers.image_utils import ImageInput
from accelerate import init_empty_weights
from PIL import Image

from musubi_tuner.utils.safetensors_utils import load_safetensors
from musubi_tuner.qwen_image.qwen_image_utils import load_qwen2_5_vl, get_qwen_prompt_embeds

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Z-Image uses Qwen2.5-VL for text encoding, so we can reuse those utilities
# VAE scale factor - Z-Image uses 8x compression like Qwen-Image
VAE_SCALE_FACTOR = 8

# Scheduler constants for flow matching
SCHEDULER_BASE_IMAGE_SEQ_LEN = 256
SCHEDULER_BASE_SHIFT = 0.5
SCHEDULER_MAX_IMAGE_SEQ_LEN = 8192
SCHEDULER_MAX_SHIFT = 0.9


def load_text_encoder(
    text_encoder_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    fp8: bool = False,
    model_path: Optional[str] = None,
) -> tuple[Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration]:
    """
    Load Qwen2.5-VL text encoder for Z-Image.
    Can load from pipeline or directly from safetensors.
    """
    # Try to use pipeline if model_path is provided
    if model_path is not None:
        try:
            from musubi_tuner.z_image.z_image_model import load_z_image_pipeline
            pipe = load_z_image_pipeline(model_path=model_path, torch_dtype=dtype or torch.bfloat16, device=device)
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
            logger.info("Loaded text encoder from Z-Image pipeline")
            return tokenizer, text_encoder
        except Exception as e:
            logger.warning(f"Failed to load from pipeline: {e}, falling back to direct loading")
    
    # Fallback to direct loading
    if text_encoder_path is None:
        raise ValueError("Either model_path or text_encoder_path must be provided")
    
    logger.info(f"Loading Qwen2.5-VL text encoder from {text_encoder_path}")
    tokenizer, text_encoder = load_qwen2_5_vl(
        text_encoder_path,
        dtype=dtype,
        device=device,
        disable_mmap=disable_mmap,
    )
    return tokenizer, text_encoder


def get_prompt_embeds(
    tokenizer: Qwen2Tokenizer,
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    prompt: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get prompt embeddings using Qwen2.5-VL.
    Returns (embeds, mask) where embeds is (1, seq_len, hidden_dim) and mask is (1, seq_len).
    Includes ROCm-specific workarounds.
    """
    from musubi_tuner.utils.device_utils import synchronize_device
    
    # Check if we're on ROCm
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    device = text_encoder.device if hasattr(text_encoder, 'device') else torch.device("cuda")
    
    # ROCm workaround: synchronize before encoding
    if is_rocm and device.type == "cuda":
        synchronize_device(device)
        torch.cuda.empty_cache()
    
    try:
        result = get_qwen_prompt_embeds(tokenizer, text_encoder, prompt)
        # ROCm workaround: synchronize after encoding
        if is_rocm and device.type == "cuda":
            synchronize_device(device)
            torch.cuda.empty_cache()
        return result
    except Exception as e:
        if is_rocm and device.type == "cuda":
            synchronize_device(device)
            torch.cuda.empty_cache()
        raise


def load_vae(
    vae_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    model_path: Optional[str] = None,
):
    """
    Load Z-Image VAE from pipeline or safetensors.
    Prefers pipeline loading if model_path is provided.
    """
    # Try to use pipeline if model_path is provided
    if model_path is not None:
        try:
            from musubi_tuner.z_image.z_image_model import load_z_image_pipeline
            pipe = load_z_image_pipeline(model_path=model_path, torch_dtype=torch.bfloat16, device=device)
            vae = pipe.vae
            vae.eval()
            logger.info("Loaded VAE from Z-Image pipeline")
            return vae
        except Exception as e:
            logger.warning(f"Failed to load from pipeline: {e}, falling back to direct loading")
    
    # Fallback to direct loading from safetensors
    if vae_path is None:
        raise ValueError("Either model_path or vae_path must be provided")
    
    logger.info(f"Loading VAE from {vae_path}")
    state_dict = load_safetensors(vae_path, device=device, disable_mmap=disable_mmap)
    logger.info(f"Loaded VAE with {len(state_dict)} parameters")
    # Return state dict for backward compatibility
    return state_dict


def calculate_shift_z_image(image_seq_len: int) -> float:
    """
    Calculate shift parameter for Z-Image flow matching scheduler based on image sequence length.
    Similar to Qwen-Image's shift calculation.
    """
    mu = (SCHEDULER_MAX_SHIFT - SCHEDULER_BASE_SHIFT) / (
        SCHEDULER_MAX_IMAGE_SEQ_LEN - SCHEDULER_BASE_IMAGE_SEQ_LEN
    ) * (image_seq_len - SCHEDULER_BASE_IMAGE_SEQ_LEN) + SCHEDULER_BASE_SHIFT
    return math.exp(mu)


def get_scheduler(discrete_flow_shift: Optional[float] = None):
    """
    Get flow matching scheduler for Z-Image.
    Uses the same scheduler as Qwen-Image.
    """
    from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler

    if discrete_flow_shift is None:
        discrete_flow_shift = SCHEDULER_BASE_SHIFT

    scheduler = FlowMatchDiscreteScheduler(
        shift=discrete_flow_shift,
        reverse=True,
        solver="euler",
    )
    return scheduler


def retrieve_timesteps(
    scheduler,
    num_inference_steps: int,
    device: Union[str, torch.device],
    sigmas: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
) -> tuple[torch.Tensor, int]:
    """
    Retrieve timesteps for Z-Image generation.
    Similar to Qwen-Image's timestep retrieval.
    """
    if sigmas is None:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

    if mu is not None:
        scheduler.shift = mu

    timesteps = scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)
    return timesteps, num_inference_steps


def prepare_latents(
    batch_size: int,
    num_channels_latents: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Prepare random latents for Z-Image generation.
    Returns latents in shape (B, C, H, W) where H and W are latent dimensions.
    """
    shape = (
        batch_size,
        num_channels_latents,
        height // VAE_SCALE_FACTOR,
        width // VAE_SCALE_FACTOR,
    )
    if generator is not None:
        latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = torch.randn(shape, device=device, dtype=dtype)
    return latents

