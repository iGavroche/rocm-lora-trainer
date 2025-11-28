# LoRA module for Z-Image

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# Z-Image uses ZImageTransformerBlock in diffusers implementation
# The blocks are stored in self.layers ModuleList
# Target the transformer block class name
Z_IMAGE_TARGET_REPLACE_MODULES = ["ZImageTransformerBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # Exclude embedding layers, normalization layers, and modulation layers
    # Based on Z-Image structure: x_embedder, t_embedder, cap_embedder, adaLN_modulation, norms
    exclude_patterns.extend([
        r".*(x_embedder|t_embedder|cap_embedder).*",
        r".*(adaLN_modulation|attention_norm|ffn_norm).*",
        r".*(context_refiner|noise_refiner|final_layer).*",
        r".*(embedder|embedding).*",
    ])

    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        Z_IMAGE_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        Z_IMAGE_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )

