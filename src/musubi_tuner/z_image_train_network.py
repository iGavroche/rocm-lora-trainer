import argparse
import gc
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_Z_IMAGE,
    ARCHITECTURE_Z_IMAGE_FULL,
)
from musubi_tuner.z_image import z_image_model, z_image_utils
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ZImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_Z_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_Z_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16  # Z-Image-Turbo uses bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 0.0  # Z-Image-Turbo uses guidance_scale=0.0 (no CFG)

        if args.mixed_precision not in ["bf16", "no"]:
            logger.warning(
                f"Z-Image-Turbo weights are in bfloat16, but mixed_precision is {args.mixed_precision}. "
                "Consider using bf16 or no mixed precision."
            )

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load Qwen2.5-VL (Z-Image uses the same text encoder)
        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        model_path = getattr(args, "model_path", None)
        tokenizer, text_encoder = z_image_utils.load_text_encoder(
            text_encoder_path=args.text_encoder,
            dtype=vl_dtype,
            device=device,
            disable_mmap=True,
            fp8=args.fp8_vl,
            model_path=model_path,
        )

        # Encode prompts
        logger.info("Encoding prompts with Qwen2.5-VL")

        sample_prompts_te_outputs = {}  # prompt -> (embeds, mask)

        with torch.amp.autocast(device_type=device.type, dtype=vl_dtype), torch.no_grad():
            for prompt_dict in prompts:
                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = " "

                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", " ")]:
                    if p is None or p in sample_prompts_te_outputs:
                        continue

                    logger.info(f"cache Text Encoder outputs for prompt: {p}")
                    embed, mask = z_image_utils.get_prompt_embeds(tokenizer, text_encoder, p)
                    sample_prompts_te_outputs[p] = (embed, mask)

        del tokenizer, text_encoder
        gc.collect()
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            embed, mask = sample_prompts_te_outputs[p]
            prompt_dict_copy["vl_embed"] = embed
            prompt_dict_copy["vl_mask"] = mask

            p = prompt_dict.get("negative_prompt", " ")
            if p:
                neg_embed, neg_mask = sample_prompts_te_outputs[p]
                prompt_dict_copy["negative_vl_embed"] = neg_embed
                prompt_dict_copy["negative_vl_mask"] = neg_mask

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """
        Generate sample images during training.
        Note: This is a placeholder - full implementation requires Z-Image model forward pass.
        """
        logger.warning("Z-Image do_inference is not yet fully implemented")
        logger.warning("Model forward pass needs to be implemented based on Z-Image architecture")
        # TODO: Implement full generation loop once model structure is complete
        return None

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae
        model_path = getattr(args, "model_path", None)

        logger.info(f"Loading VAE model from {vae_path or model_path}")
        vae = z_image_model.load_z_image_vae(
            device="cpu",
            vae_path=vae_path,
            model_path=model_path,
            torch_dtype=vae_dtype,
        )
        vae.eval()
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        # Use model_path if available, otherwise dit_path
        model_path = getattr(args, "model_path", None) or dit_path
        model = z_image_model.load_z_image_transformer(
            accelerator.device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            disable_numpy_memmap=args.disable_numpy_memmap,
            model_path=model_path,
        )
        return model

    def compile_transformer(self, args, transformer):
        transformer: z_image_model.ZImageTransformer2DModel = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.layers] if hasattr(transformer, "layers") else [], disable_linear=self.blocks_to_swap > 0
        )
    
    def enable_gradient_checkpointing_if_requested(self, args, transformer):
        """Enable gradient checkpointing for Z-Image transformer if requested."""
        if args.gradient_checkpointing:
            # diffusers ZImageTransformer2DModel uses _gradient_checkpointing_func
            # which should be a callable, not a bool
            if hasattr(transformer, "enable_gradient_checkpointing"):
                try:
                    transformer.enable_gradient_checkpointing()
                    logger.info("Z-Image: Gradient checkpointing enabled via enable_gradient_checkpointing()")
                except Exception as e:
                    logger.warning(f"Z-Image: enable_gradient_checkpointing() failed: {e}, trying manual setup")
                    self._enable_gradient_checkpointing_manual(transformer)
            else:
                self._enable_gradient_checkpointing_manual(transformer)
    
    def _enable_gradient_checkpointing_manual(self, transformer):
        """Manually enable gradient checkpointing by setting _gradient_checkpointing_func."""
        from torch.utils.checkpoint import checkpoint as gradient_checkpointing_func
        if hasattr(transformer, "_gradient_checkpointing_func"):
            # Set the function, not a bool
            transformer._gradient_checkpointing_func = gradient_checkpointing_func
            # Also set gradient_checkpointing flag if it exists
            if hasattr(transformer, "gradient_checkpointing"):
                transformer.gradient_checkpointing = True
            logger.info("Z-Image: Gradient checkpointing enabled manually (set _gradient_checkpointing_func)")
        else:
            logger.warning("Z-Image: _gradient_checkpointing_func not found - gradient checkpointing may not work")

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        """
        Forward pass through Z-Image transformer.
        Z-Image forward expects:
        - x: List[torch.Tensor] - list of image tensors (one per batch item), each (C, F, H, W) where F=1 for images
        - t: torch.Tensor - timestep tensor
        - cap_feats: List[torch.Tensor] - list of caption feature tensors (one per batch item)
        """
        from diffusers import ZImageTransformer2DModel
        model: ZImageTransformer2DModel = transformer

        bsize = latents.shape[0]
        # Use latents parameter directly (already from batch, no need to re-extract)
        # latents is already B, C, 1, H, W (or B, C, H, W for images) from the training loop

        # Get text embeddings (caption features)
        vl_embed = batch["vl_embed"]  # list of (L, D) or tensor
        if isinstance(vl_embed, list):
            cap_feats = vl_embed  # Already a list
        else:
            # Convert tensor to list
            cap_feats = [vl_embed[i] for i in range(bsize)]

        # Prepare image tensors as list (one per batch item)
        # Z-Image expects (C, F, H, W) where F=1 for images
        if len(noisy_model_input.shape) == 5:  # B, C, F, H, W
            x_list = []
            for i in range(bsize):
                x_item = noisy_model_input[i]  # (C, F, H, W)
                # Fix dimension order if needed: (1, 16, 64, 64) -> (16, 1, 64, 64)
                if x_item.shape[0] == 1 and x_item.shape[1] > 1:
                    x_item = x_item.permute(1, 0, 2, 3)  # (1, C, H, W) -> (C, 1, H, W)
                x_list.append(x_item)
        elif len(noisy_model_input.shape) == 4:  # B, C, H, W
            x_list = [noisy_model_input[i].unsqueeze(1) for i in range(bsize)]  # Add frame dim: (C, H, W) -> (C, 1, H, W)
        else:
            raise ValueError(f"Unexpected noisy_model_input shape: {noisy_model_input.shape}")

        # Ensure tensors are on correct device and dtype
        x_list = [x.to(device=accelerator.device, dtype=network_dtype) for x in x_list]
        cap_feats = [cap.to(device=accelerator.device, dtype=network_dtype) for cap in cap_feats]
        timesteps_normalized = (timesteps / 1000.0).to(device=accelerator.device, dtype=network_dtype)  # Z-Image uses 0-1 timesteps

        # Ensure gradients for gradient checkpointing
        if args.gradient_checkpointing:
            for x in x_list:
                x.requires_grad_(True)
            for cap in cap_feats:
                cap.requires_grad_(True)

        # Call model forward - process entire batch at once
        with accelerator.autocast():
            model_output = model(
                x=x_list,
                t=timesteps_normalized,
                cap_feats=cap_feats,
                patch_size=2,  # Default patch size
                f_patch_size=1,  # Frame patch size (1 for images)
            )
        
        # model_output is a tuple: (x, {}) where x is List[torch.Tensor]
        model_pred_list = model_output[0]  # List of (C, H, W) tensors
        
        # Stack back to batch format: (B, C, H, W)
        model_pred = torch.stack(model_pred_list, dim=0)  # (B, C, H, W)
        
        # Add frame dimension if needed to match latents shape
        if len(latents.shape) == 5:  # B, C, 1, H, W
            model_pred = model_pred.unsqueeze(2)  # Add frame dim

        # Flow matching loss
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents

        return model_pred, target

    # endregion model specific


def z_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Z-Image specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder (Qwen2.5-VL) checkpoint path (optional if using model_path)")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument("--num_layers", type=int, default=30, help="Number of layers in the DiT model, default is 30 for Turbo")
    parser.add_argument("--model_path", type=str, default=None, help="Path to Z-Image model directory or HuggingFace model ID (e.g., Tongyi-MAI/Z-Image-Turbo). If provided, will load from pipeline.")
    return parser


def main():
    parser = setup_parser_common()
    parser = z_image_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"  # DiT dtype is bfloat16 for Z-Image-Turbo
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # VAE dtype

    trainer = ZImageNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()

