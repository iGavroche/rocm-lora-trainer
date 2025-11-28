import argparse
from typing import Optional

import torch
import accelerate

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_Z_IMAGE,
    ItemInfo,
    save_text_encoder_output_cache_z_image,
)

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging

from musubi_tuner.z_image import z_image_utils
from musubi_tuner.utils.device_utils import synchronize_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer,
    text_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    accelerator: Optional[accelerate.Accelerator],
):
    """
    Encode prompts and save text encoder cache for Z-Image.
    Z-Image uses Qwen2.5-VL same as Qwen-Image, so encoding is text-only (no images).
    Includes ROCm-specific workarounds to prevent SIGSEGV.
    """
    # Check if we're on ROCm
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    
    prompts = [item.caption for item in batch]

    for i, item in enumerate(batch):
        logger.info(f"Item {i}: {item.item_key}, prompt: {item.caption}")

    # ROCm workaround: synchronize before encoding
    if device.type == "cuda":
        synchronize_device(device)
        torch.cuda.empty_cache()

    # encode prompt using Qwen2.5-VL
    # ROCm workaround: Process prompts one at a time to avoid queue buildup
    if is_rocm and len(prompts) > 1:
        logger.debug(f"ROCm: Encoding {len(prompts)} prompts one at a time")
        embeds_list = []
        masks_list = []
        for idx, prompt in enumerate(prompts):
            logger.debug(f"ROCm: Encoding prompt {idx+1}/{len(prompts)}")
            try:
                synchronize_device(device)
                torch.cuda.empty_cache()
                with torch.no_grad():
                    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                        embed, mask = z_image_utils.get_prompt_embeds(tokenizer, text_encoder, [prompt])
                        synchronize_device(device)
                        if embed.dtype == torch.float8_e4m3fn:
                            embed = embed.to(torch.bfloat16)
                        synchronize_device(device)
                embeds_list.append(embed)
                masks_list.append(mask)
                synchronize_device(device)
                torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error encoding prompt {idx+1}: {e}", exc_info=True)
                synchronize_device(device)
                torch.cuda.empty_cache()
                raise
        # Concatenate results
        embed = torch.cat(embeds_list, dim=0)
        mask = torch.cat(masks_list, dim=0)
    else:
        with torch.no_grad():
            try:
                if is_rocm:
                    # On ROCm, use torch.amp.autocast() instead of accelerator.autocast()
                    dtype = torch.bfloat16
                    with torch.amp.autocast(device_type="cuda", dtype=dtype):
                        synchronize_device(device)
                        embed, mask = z_image_utils.get_prompt_embeds(tokenizer, text_encoder, prompts)
                        synchronize_device(device)
                        if embed.dtype == torch.float8_e4m3fn:  # QwenVL-2.5 may return fp8
                            embed = embed.to(torch.bfloat16)
                        synchronize_device(device)
                elif accelerator is not None:
                    with accelerator.autocast():
                        embed, mask = z_image_utils.get_prompt_embeds(tokenizer, text_encoder, prompts)
                        if embed.dtype == torch.float8_e4m3fn:  # QwenVL-2.5 may return fp8
                            embed = embed.to(torch.bfloat16)
                else:
                    embed, mask = z_image_utils.get_prompt_embeds(tokenizer, text_encoder, prompts)
            except Exception as e:
                logger.error(f"Error during text encoder encoding: {e}", exc_info=True)
                if device.type == "cuda":
                    synchronize_device(device)
                    torch.cuda.empty_cache()
                raise
            finally:
                # ROCm workaround: synchronize after encoding
                if device.type == "cuda":
                    synchronize_device(device)
                    torch.cuda.empty_cache()

    # save prompt cache
    for item, (embed_i, mask_i) in zip(batch, zip(embed, mask)):
        txt_len = mask_i.to(dtype=torch.bool).sum().item()  # length of the text in the batch
        embed_i = embed_i[:txt_len]
        save_text_encoder_output_cache_z_image(item, embed_i)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = z_image_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_Z_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # define accelerator for fp8 inference
    vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
    accelerator = None
    if args.fp8_vl:
        accelerator = accelerate.Accelerator(mixed_precision="bf16")

    # prepare cache files and paths: all_cache_files_for_dataset = existing cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(
        datasets
    )

    # Determine text encoder path - check for local files first, then model_path, then HuggingFace
    import os
    
    model_path = getattr(args, "model_path", None)
    text_encoder_path = getattr(args, "text_encoder", None)
    
    # Check for local model files in models/z-image/
    local_model_dir = "models/z-image"
    local_vae_file = os.path.join(local_model_dir, "ae.safetensors")
    local_transformer_file = os.path.join(local_model_dir, "z_image_turbo_bf16.safetensors")
    local_text_encoder_file = os.path.join(local_model_dir, "qwen_3_4b.safetensors")
    
    # If local files exist, use them
    if os.path.exists(local_text_encoder_file) and os.path.exists(local_transformer_file):
        logger.info(f"Found local Z-Image model files in {local_model_dir}")
        if model_path is None:
            model_path = local_model_dir
            logger.info(f"Using local model directory: {model_path}")
    elif model_path is None:
        # Try HuggingFace as fallback
        model_path = "Tongyi-MAI/Z-Image-Turbo"
        logger.info(f"No local model files found, will download from HuggingFace: {model_path}")
    
    if model_path is None and text_encoder_path is None:
        raise ValueError("Either --model_path or --text_encoder must be provided, or place model files in models/z-image/")

    # Check if we're on ROCm
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

    # Load Qwen2.5-VL
    logger.info(f"Loading Qwen2.5-VL from {model_path or text_encoder_path}")
    try:
        # ROCm workaround: Load to CPU first, then move to GPU only when needed
        if is_rocm:
            logger.info("ROCm: Loading text encoder to CPU first to prevent SIGSEGV")
            tokenizer, text_encoder = z_image_utils.load_text_encoder(
                text_encoder_path=text_encoder_path,
                dtype=vl_dtype,
                device="cpu",  # Load to CPU first
                disable_mmap=True,
                fp8=args.fp8_vl,
                model_path=model_path,
            )
            logger.info("Text encoder loaded to CPU, will move to GPU only during encoding")
        else:
            tokenizer, text_encoder = z_image_utils.load_text_encoder(
                text_encoder_path=text_encoder_path,
                dtype=vl_dtype,
                device=device,
                disable_mmap=True,
                fp8=args.fp8_vl,
                model_path=model_path,
            )
            text_encoder.to(device)
        logger.info("Successfully loaded text encoder")
    except Exception as e:
        logger.error(f"Failed to load text encoder: {e}")
        raise ValueError(
            f"Failed to load text encoder. Please ensure:\n"
            f"  1. Model files are in {local_model_dir}/ (ae.safetensors, z_image_turbo_bf16.safetensors, qwen_3_4b.safetensors), OR\n"
            f"  2. diffusers is installed and can download from HuggingFace (Tongyi-MAI/Z-Image-Turbo)\n"
            f"Error: {e}"
        )

    # Check if we're on ROCm
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    if is_rocm:
        logger.info("ROCm detected: Using ROCm-specific workarounds (single-item processing, extra synchronization)")

    # Encode with Qwen2.5-VL
    logger.info("Encoding with Qwen2.5-VL")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal tokenizer, text_encoder, device, accelerator, is_rocm
        # ROCm workaround: ALWAYS process one prompt at a time to reduce GPU queue pressure
        # Also move text encoder to CPU between operations to prevent queue buildup
        if is_rocm:
            logger.info(f"ROCm: Processing {len(batch)} prompts one at a time with CPU-based encoding")
            
            for idx, item in enumerate(batch):
                logger.info(f"ROCm: Processing prompt {idx+1}/{len(batch)}: {item.item_key}")
                try:
                    # ROCm workaround: Use CPU for text encoding to avoid SIGSEGV
                    # This is slower but stable
                    synchronize_device(device)
                    torch.cuda.empty_cache()
                    
                    # Move text encoder to CPU for encoding (safer on ROCm)
                    text_encoder_cpu = text_encoder.to("cpu")
                    synchronize_device(device)
                    
                    # Encode on CPU
                    encode_and_save_batch(tokenizer, text_encoder_cpu, [item], torch.device("cpu"), accelerator)
                    
                    synchronize_device(device)
                    torch.cuda.empty_cache()
                    
                    # Small delay to let system settle
                    import time
                    time.sleep(0.05)
                except Exception as e:
                    logger.error(f"Failed to encode {item.item_key}: {e}", exc_info=True)
                    synchronize_device(device)
                    torch.cuda.empty_cache()
                    raise
        else:
            encode_and_save_batch(tokenizer, text_encoder, batch, device, accelerator)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        requires_content=False,  # Z-Image text encoding is text-only, no images
    )
    del text_encoder

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def z_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Z-Image specific parser setup for text encoder caching"""
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Text Encoder (Qwen2.5-VL) checkpoint path (optional if using --model_path)",
    )
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to Z-Image model directory or HuggingFace model ID (e.g., Tongyi-MAI/Z-Image-Turbo). "
        "If provided, will load from pipeline.",
    )

    return parser


if __name__ == "__main__":
    main()

