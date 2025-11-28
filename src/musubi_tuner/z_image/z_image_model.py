"""
Z-Image model loading module using diffusers ZImagePipeline.
"""

import os
import logging
from typing import Optional, Union

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel

from musubi_tuner.utils.device_utils import synchronize_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _detect_local_model_path(model_dir: str) -> Optional[str]:
    """
    Detect if model_dir contains Z-Image model files.
    Returns the directory path if valid, None otherwise.
    
    Checks for:
    - ae.safetensors (VAE)
    - z_image_turbo_bf16.safetensors (transformer)
    - qwen_3_4b.safetensors (text encoder)
    """
    if not os.path.isdir(model_dir):
        return None
    
    # Check for required model files
    vae_files = ["ae.safetensors", "vae.safetensors"]
    transformer_files = [
        "z_image_turbo_bf16.safetensors",
        "transformer.safetensors",
        "model.safetensors",
        "diffusion_pytorch_model.safetensors",
    ]
    text_encoder_files = ["qwen_3_4b.safetensors", "text_encoder.safetensors"]
    
    has_vae = any(os.path.exists(os.path.join(model_dir, f)) for f in vae_files)
    has_transformer = any(os.path.exists(os.path.join(model_dir, f)) for f in transformer_files)
    has_text_encoder = any(os.path.exists(os.path.join(model_dir, f)) for f in text_encoder_files)
    
    # If we have at least transformer and VAE, consider it valid
    # Text encoder might be loaded separately
    if has_transformer and has_vae:
        logger.info(f"Detected local Z-Image model files in {model_dir}")
        if has_text_encoder:
            logger.info("  - Found VAE, transformer, and text encoder files")
        else:
            logger.info("  - Found VAE and transformer files (text encoder may be loaded from HuggingFace)")
        return model_dir
    
    return None


def load_z_image_pipeline(
    model_path: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = "cuda",
    low_cpu_mem_usage: bool = False,
) -> ZImagePipeline:
    """
    Load Z-Image pipeline from local directory or HuggingFace.
    
    Args:
        model_path: Path to local model directory (models/z-image/) or HuggingFace model ID.
                   If None, tries local first, then HuggingFace.
        torch_dtype: Dtype for model weights (default: bfloat16)
        device: Target device
        low_cpu_mem_usage: Whether to use low CPU memory usage loading
        
    Returns:
        ZImagePipeline instance
    """
    if model_path is None:
        # Try local first
        local_path = _detect_local_model_path("models/z-image")
        if local_path:
            logger.info(f"Found local Z-Image model at {local_path}")
            model_path = local_path
        else:
            # Fallback to HuggingFace
            model_path = "Tongyi-MAI/Z-Image-Turbo"
            logger.info(f"Using HuggingFace model: {model_path}")
    elif os.path.isdir(model_path):
        # Local directory
        logger.info(f"Loading Z-Image pipeline from local directory: {model_path}")
    else:
        # HuggingFace model ID
        logger.info(f"Loading Z-Image pipeline from HuggingFace: {model_path}")
    
    logger.info(f"Loading Z-Image pipeline with dtype={torch_dtype}, device={device}")
    try:
        # Try loading from local directory or HuggingFace
        # Note: If local directory has individual safetensors files but not a full repository structure,
        # diffusers might not be able to load directly. In that case, it will fall back to HuggingFace.
        pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
        pipe.to(device)
        synchronize_device(device)
        logger.info("Successfully loaded Z-Image pipeline")
    except Exception as e:
        # If loading from local directory failed, try HuggingFace as fallback
        if os.path.isdir(model_path) and model_path != "Tongyi-MAI/Z-Image-Turbo":
            logger.warning(f"Failed to load from local directory {model_path}: {e}")
            logger.info("Attempting to download from HuggingFace instead...")
            try:
                pipe = ZImagePipeline.from_pretrained(
                    "Tongyi-MAI/Z-Image-Turbo",
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=low_cpu_mem_usage,
                )
                pipe.to(device)
                synchronize_device(device)
                logger.info("Successfully loaded Z-Image pipeline from HuggingFace")
            except Exception as e2:
                logger.error(f"Failed to load from HuggingFace: {e2}")
                raise ValueError(
                    f"Failed to load Z-Image pipeline from both local directory and HuggingFace.\n"
                    f"Local directory error: {e}\n"
                    f"HuggingFace error: {e2}\n"
                    f"Please ensure:\n"
                    f"  1. Model files are in {model_path}/ with proper structure, OR\n"
                    f"  2. diffusers can download from HuggingFace (Tongyi-MAI/Z-Image-Turbo)"
                ) from e2
        else:
            raise
    
    return pipe


def load_z_image_transformer(
    device: Union[str, torch.device],
    dit_path: Optional[str] = None,
    attn_mode: str = "sdpa",
    split_attn: bool = False,
    loading_device: Union[str, torch.device] = "cpu",
    dit_weight_dtype: Optional[torch.dtype] = None,
    disable_numpy_memmap: bool = False,
    model_path: Optional[str] = None,
) -> ZImageTransformer2DModel:
    """
    Load Z-Image transformer model.
    
    Args:
        device: Target device for the model
        dit_path: Path to transformer (deprecated, use model_path instead)
        attn_mode: Attention mode ("sdpa", "flash", "_flash_3")
        split_attn: Whether to split attention (for ROCm stability)
        loading_device: Device to load weights on (usually "cpu")
        dit_weight_dtype: Dtype for weights (bfloat16 for Z-Image-Turbo)
        disable_numpy_memmap: Whether to disable numpy memmap
        model_path: Path to model directory or HuggingFace ID (preferred)
        
    Returns:
        ZImageTransformer2DModel instance
    """
    # Use model_path if provided, otherwise dit_path for backward compatibility
    if model_path is None:
        model_path = dit_path
    
    # Determine dtype
    if dit_weight_dtype is None:
        dit_weight_dtype = torch.bfloat16
    
    # Load pipeline and extract transformer
    logger.info(f"Loading Z-Image transformer from {model_path}")
    pipe = load_z_image_pipeline(
        model_path=model_path,
        torch_dtype=dit_weight_dtype,
        device=loading_device,
        low_cpu_mem_usage=not disable_numpy_memmap,
    )
    
    transformer = pipe.transformer
    
    # Set attention backend
    if attn_mode == "flash":
        transformer.set_attention_backend("flash")
        logger.info("Using Flash Attention 2")
    elif attn_mode == "_flash_3":
        transformer.set_attention_backend("_flash_3")
        logger.info("Using Flash Attention 3")
    else:
        logger.info("Using SDPA (default)")
    
    # Move transformer to target device
    transformer = transformer.to(device)
    synchronize_device(device)
    
    logger.info(f"Z-Image transformer loaded on {device}")
    return transformer


def load_z_image_vae(
    device: Union[str, torch.device],
    vae_path: Optional[str] = None,
    model_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    """
    Load Z-Image VAE from pipeline.
    
    Args:
        device: Target device
        vae_path: Path to VAE (deprecated, use model_path instead)
        model_path: Path to model directory or HuggingFace ID (preferred)
        torch_dtype: Dtype for VAE
        
    Returns:
        VAE model instance
    """
    # Use model_path if provided, otherwise vae_path for backward compatibility
    if model_path is None:
        model_path = vae_path
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    
    # Load pipeline and extract VAE
    logger.info(f"Loading Z-Image VAE from {model_path}")
    pipe = load_z_image_pipeline(
        model_path=model_path,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    vae = pipe.vae
    vae.eval()
    
    logger.info(f"Z-Image VAE loaded on {device}")
    return vae


def load_z_image_text_encoder(
    device: Union[str, torch.device],
    text_encoder_path: Optional[str] = None,
    model_path: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> torch.nn.Module:
    """
    Load Z-Image text encoder (Qwen2.5-VL) from pipeline.
    
    Args:
        device: Target device
        text_encoder_path: Path to text encoder (deprecated, use model_path instead)
        model_path: Path to model directory or HuggingFace ID (preferred)
        torch_dtype: Dtype for text encoder
        
    Returns:
        Text encoder model instance
    """
    # Use model_path if provided, otherwise text_encoder_path for backward compatibility
    if model_path is None:
        model_path = text_encoder_path
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    
    # Load pipeline and extract text encoder
    logger.info(f"Loading Z-Image text encoder from {model_path}")
    pipe = load_z_image_pipeline(
        model_path=model_path,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    
    logger.info(f"Z-Image text encoder loaded on {device}")
    return text_encoder
