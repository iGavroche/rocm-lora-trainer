import ast
import asyncio
from datetime import timedelta
import gc
import importlib
import argparse
import math
import os
import pathlib
import re
import sys
import random
import time
import json
from multiprocessing import Value
from typing import Any, Dict, List, Optional
import accelerate
import numpy as np
from packaging.version import Version
from PIL import Image

import huggingface_hub
import toml

import torch
from tqdm import tqdm
from accelerate.utils import TorchDynamoPlugin, set_seed, DynamoBackend
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState
from safetensors.torch import load_file
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION

from musubi_tuner.dataset import config_utils
from musubi_tuner.hunyuan_model.models import load_transformer, get_rotary_pos_embed_by_shape, HYVideoDiffusionTransformer
import musubi_tuner.hunyuan_model.text_encoder as text_encoder_module
from musubi_tuner.hunyuan_model.vae import load_vae, VAE_VER
import musubi_tuner.hunyuan_model.vae as vae_module
from musubi_tuner.modules.lr_schedulers import RexLR
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
import musubi_tuner.networks.lora as lora_module
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HUNYUAN_VIDEO, ARCHITECTURE_HUNYUAN_VIDEO_FULL
from musubi_tuner.hv_generate_video import save_images_grid, save_videos_grid, resize_image_to_bucket, encode_to_latents

import logging
import os

from musubi_tuner.utils import huggingface_utils, model_utils, train_utils, sai_model_spec

logger = logging.getLogger(__name__)

# Enable verbose logging based on environment variables
log_level = os.environ.get("MUSUBI_LOG_LEVEL", "INFO").upper()
if log_level == "DEBUG":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
else:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Enable PyTorch verbose logging if requested
if os.environ.get("PYTORCH_VERBOSE", "0") == "1":
    import torch._dynamo
    torch._dynamo.config.verbose = True
    torch._dynamo.config.log_level = logging.DEBUG
    
    # Enable autograd anomaly detection
    torch.autograd.set_detect_anomaly(True)
    logger.info("PyTorch verbose logging and anomaly detection enabled")

# Log ROCm/HIP environment variables for debugging
if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip:
    logger.info("ROCm environment variables:")
    rocm_vars = ["HIP_LAUNCH_BLOCKING", "AMD_LOG_LEVEL", "HIP_VISIBLE_DEVICES", 
                 "ROCM_DEBUG", "HIP_PROFILE", "HIP_DISABLE_IPC", "HSA_OVERRIDE_GFX_VERSION"]
    for var in rocm_vars:
        value = os.environ.get(var, "not set")
        logger.info(f"  {var}={value}")


SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


# for collate_fn: epoch and step is multiprocessing.Value
class collator_class:
    def __init__(self, epoch, dataset):
        self.current_epoch = epoch
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch for validation
        dataset.set_current_epoch(self.current_epoch.value)
        return examples[0]  # batch size is always 1, so we unwrap it here


def prepare_accelerator(args: argparse.Namespace) -> Accelerator:
    """
    DeepSpeed is not supported in this script currently.
    """
    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]

    dynamo_plugin = None
    if args.dynamo_backend.upper() != "NO":
        dynamo_plugin = TorchDynamoPlugin(
            backend=DynamoBackend(args.dynamo_backend.upper()),
            mode=args.dynamo_mode,
            fullgraph=args.dynamo_fullgraph,
            dynamic=args.dynamo_dynamic,
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision if args.mixed_precision else None,
        log_with=log_with,
        project_dir=logging_dir,
        dynamo_plugin=dynamo_plugin,
        kwargs_handlers=kwargs_handlers,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"f (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["frame_count"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"fs ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["discrete_flow_shift"] = float(m.group(1))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["cfg_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"i (.+)", parg, re.IGNORECASE)
            if m:  # image path
                prompt_dict["image_path"] = m.group(1).strip()
                continue

            m = re.match(r"ei (.+)", parg, re.IGNORECASE)
            if m:  # end image path
                prompt_dict["end_image_path"] = m.group(1).strip()
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["control_video_path"] = m.group(1).strip()
                continue

            m = re.match(r"ci (.+)", parg, re.IGNORECASE)
            if m:
                # can be multiple control images
                control_image_path = m.group(1).strip()
                if "control_image_path" not in prompt_dict:
                    prompt_dict["control_image_path"] = []
                prompt_dict["control_image_path"].append(control_image_path)
                continue

            m = re.match(r"of (.+)", parg, re.IGNORECASE)
            if m:  # output folder
                prompt_dict["one_frame"] = m.group(1).strip()
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析エラー: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> list[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)

    # Check each timestep individually and use rounding if not found
    step_indices = []
    for t in timesteps:
        matches = (schedule_timesteps == t).nonzero()
        if matches.numel() > 0:
            step_indices.append(matches.item())
        else:
            # Round to nearest timestep if not found
            nearest_idx = torch.argmin(torch.abs(schedule_timesteps - t)).item()
            step_indices.append(nearest_idx)
            logger.debug(f"Timestep {t.item()} not in schedule, using nearest: {schedule_timesteps[nearest_idx].item()}")

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_loss_weighting_for_sd3(weighting_scheme: str, noise_scheduler, timesteps, device, dtype):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt" or weighting_scheme == "cosmap":
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=5, dtype=dtype)
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        else:
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
    else:
        weighting = None  # torch.ones_like(sigmas)
    return weighting


def should_sample_images(args, steps, epoch=None):
    if steps == 0:
        if not args.sample_at_first:
            return False
    else:
        should_sample_by_steps = args.sample_every_n_steps is not None and steps % args.sample_every_n_steps == 0
        should_sample_by_epochs = (
            args.sample_every_n_epochs is not None and epoch is not None and epoch % args.sample_every_n_epochs == 0
        )
        if not should_sample_by_steps and not should_sample_by_epochs:
            return False
    return True


class NetworkTrainer:
    def __init__(self):
        self.blocks_to_swap = None
        self.timestep_range_pool = []
        self.num_timestep_buckets: Optional[int] = None  # for get_bucketed_timestep()

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        network_train_unet_only = True
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().endswith("Prodigy".lower()):
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
        else:
            idx = 0
            if not network_train_unet_only:
                logs["lr/textencoder"] = float(lrs[0])
                idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower().endswith(
                    "Prodigy".lower()
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None:
                    logs[f"lr/d*lr/group{i}"] = optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]

        return logs

    def get_optimizer(self, args, trainable_params: list[torch.nn.Parameter]) -> tuple[str, str, torch.optim.Optimizer]:
        # adamw, adamw8bit, adafactor

        optimizer_type = args.optimizer_type.lower()

        # split optimizer_type and optimizer_args
        optimizer_kwargs = {}
        if args.optimizer_args is not None and len(args.optimizer_args) > 0:
            for arg in args.optimizer_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                optimizer_kwargs[key] = value

        lr = args.learning_rate
        optimizer = None
        optimizer_class = None

        if optimizer_type.endswith("8bit".lower()):
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError("No bitsandbytes / bitsandbytesがインストールされていないようです")

            # Check if bitsandbytes ROCm binary is available
            # IMPORTANT: Don't try to access cextension.lib as it may hang trying to load the library
            # Instead, we'll catch the error during optimizer.step() and fall back then
            # For now, assume it might work and let the error handler in optimizer.step() catch it
            bnb_available = True  # Try to use it, will fall back if it fails during step()

            if optimizer_type == "AdamW8bit".lower():
                if bnb_available:
                    logger.info(f"use 8-bit AdamW optimizer | {optimizer_kwargs}")
                    optimizer_class = bnb.optim.AdamW8bit
                    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)
                else:
                    # Fall back to regular AdamW
                    logger.warning("Falling back to regular AdamW optimizer (bitsandbytes ROCm binary not available)")
                    optimizer_class = torch.optim.AdamW
                    optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "Adafactor".lower():
            # Adafactor: check relative_step and warmup_init
            if "relative_step" not in optimizer_kwargs:
                optimizer_kwargs["relative_step"] = True  # default
            if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
                logger.info(
                    "set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
                )
                optimizer_kwargs["relative_step"] = True
            logger.info(f"use Adafactor optimizer | {optimizer_kwargs}")

            if optimizer_kwargs["relative_step"]:
                logger.info("relative_step is true / relative_stepがtrueです")
                if lr != 0.0:
                    logger.warning("learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
                args.learning_rate = None

                if args.lr_scheduler != "adafactor":
                    logger.info("use adafactor_scheduler / スケジューラにadafactor_schedulerを使用します")
                args.lr_scheduler = f"adafactor:{lr}"  # ちょっと微妙だけど

                lr = None
            else:
                if args.max_grad_norm != 0.0:
                    logger.warning(
                        "because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
                    )
                if args.lr_scheduler != "constant_with_warmup":
                    logger.warning("constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
                if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
                    logger.warning("clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")

            optimizer_class = transformers.optimization.Adafactor
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        elif optimizer_type == "AdamW".lower():
            logger.info(f"use AdamW optimizer | {optimizer_kwargs}")
            optimizer_class = torch.optim.AdamW
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        if optimizer is None:
            # 任意のoptimizerを使う
            case_sensitive_optimizer_type = args.optimizer_type  # not lower
            logger.info(f"use {case_sensitive_optimizer_type} | {optimizer_kwargs}")

            if "." not in case_sensitive_optimizer_type:  # from torch.optim
                optimizer_module = torch.optim
            else:  # from other library
                values = case_sensitive_optimizer_type.split(".")
                optimizer_module = importlib.import_module(".".join(values[:-1]))
                case_sensitive_optimizer_type = values[-1]

            optimizer_class = getattr(optimizer_module, case_sensitive_optimizer_type)
            optimizer = optimizer_class(trainable_params, lr=lr, **optimizer_kwargs)

        # for logging
        optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
        optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

        # get train and eval functions
        if hasattr(optimizer, "train") and callable(optimizer.train):
            train_fn = optimizer.train
            eval_fn = optimizer.eval
        else:
            train_fn = lambda: None
            eval_fn = lambda: None

        return optimizer_name, optimizer_args, optimizer, train_fn, eval_fn

    def is_schedulefree_optimizer(self, optimizer: torch.optim.Optimizer, args: argparse.Namespace) -> bool:
        return args.optimizer_type.lower().endswith("schedulefree".lower())  # or args.optimizer_schedulefree_wrapper

    def get_dummy_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        # dummy scheduler for schedulefree optimizer. supports only empty step(), get_last_lr() and optimizers.
        # this scheduler is used for logging only.
        # this isn't be wrapped by accelerator because of this class is not a subclass of torch.optim.lr_scheduler._LRScheduler
        class DummyScheduler:
            def __init__(self, optimizer: torch.optim.Optimizer):
                self.optimizer = optimizer

            def step(self):
                pass

            def get_last_lr(self):
                return [group["lr"] for group in self.optimizer.param_groups]

        return DummyScheduler(optimizer)

    def get_lr_scheduler(self, args, optimizer: torch.optim.Optimizer, num_processes: int):
        """
        Unified API to get any scheduler from its name.
        """
        # if schedulefree optimizer, return dummy scheduler
        if self.is_schedulefree_optimizer(optimizer, args):
            return self.get_dummy_scheduler(optimizer)

        name = args.lr_scheduler
        num_training_steps = args.max_train_steps * num_processes  # * args.gradient_accumulation_steps
        num_warmup_steps: Optional[int] = (
            int(args.lr_warmup_steps * num_training_steps) if isinstance(args.lr_warmup_steps, float) else args.lr_warmup_steps
        )
        num_decay_steps: Optional[int] = (
            int(args.lr_decay_steps * num_training_steps) if isinstance(args.lr_decay_steps, float) else args.lr_decay_steps
        )
        num_stable_steps = num_training_steps - num_warmup_steps - num_decay_steps
        num_cycles = args.lr_scheduler_num_cycles
        power = args.lr_scheduler_power
        timescale = args.lr_scheduler_timescale
        min_lr_ratio = args.lr_scheduler_min_lr_ratio

        lr_scheduler_kwargs = {}  # get custom lr_scheduler kwargs
        if args.lr_scheduler_args is not None and len(args.lr_scheduler_args) > 0:
            for arg in args.lr_scheduler_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                lr_scheduler_kwargs[key] = value

        def wrap_check_needless_num_warmup_steps(return_vals):
            if num_warmup_steps is not None and num_warmup_steps != 0:
                raise ValueError(f"{name} does not require `num_warmup_steps`. Set None or 0.")
            return return_vals

        # using any lr_scheduler from other library
        if args.lr_scheduler_type:
            lr_scheduler_type = args.lr_scheduler_type
            logger.info(f"use {lr_scheduler_type} | {lr_scheduler_kwargs} as lr_scheduler")
            if "." not in lr_scheduler_type:  # default to use torch.optim
                lr_scheduler_module = torch.optim.lr_scheduler
            else:
                values = lr_scheduler_type.split(".")
                lr_scheduler_module = importlib.import_module(".".join(values[:-1]))
                lr_scheduler_type = values[-1]
            lr_scheduler_class = getattr(lr_scheduler_module, lr_scheduler_type)
            lr_scheduler = lr_scheduler_class(optimizer, **lr_scheduler_kwargs)
            return lr_scheduler

        if name.startswith("adafactor"):
            assert type(optimizer) == transformers.optimization.Adafactor, (
                "adafactor scheduler must be used with Adafactor optimizer / adafactor schedulerはAdafactorオプティマイザと同時に使ってください"
            )
            initial_lr = float(name.split(":")[1])
            # logger.info(f"adafactor scheduler init lr {initial_lr}")
            return wrap_check_needless_num_warmup_steps(transformers.optimization.AdafactorSchedule(optimizer, initial_lr))

        if name.lower() == "rex":
            return RexLR(
                optimizer,
                max_lr=args.learning_rate,
                min_lr=(  # Will start and end with min_lr, use non-zero min_lr by default
                    args.learning_rate * min_lr_ratio if min_lr_ratio is not None else args.learning_rate * 0.01
                ),
                num_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                **lr_scheduler_kwargs,
            )

        if name == DiffusersSchedulerType.PIECEWISE_CONSTANT.value:
            name = DiffusersSchedulerType(name)
            schedule_func = DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION[name]
            return schedule_func(optimizer, **lr_scheduler_kwargs)  # step_rules and last_epoch are given as kwargs

        name = SchedulerType(name)
        schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]

        if name == SchedulerType.CONSTANT:
            return wrap_check_needless_num_warmup_steps(schedule_func(optimizer, **lr_scheduler_kwargs))

        # All other schedulers require `num_warmup_steps`
        if num_warmup_steps is None:
            raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

        if name == SchedulerType.CONSTANT_WITH_WARMUP:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **lr_scheduler_kwargs)

        if name == SchedulerType.INVERSE_SQRT:
            return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, timescale=timescale, **lr_scheduler_kwargs)

        # All other schedulers require `num_training_steps`
        if num_training_steps is None:
            raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

        if name == SchedulerType.COSINE_WITH_RESTARTS:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.POLYNOMIAL:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=power,
                **lr_scheduler_kwargs,
            )

        if name == SchedulerType.COSINE_WITH_MIN_LR:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles / 2,
                min_lr_rate=min_lr_ratio,
                **lr_scheduler_kwargs,
            )

        # these schedulers do not require `num_decay_steps`
        if name == SchedulerType.LINEAR or name == SchedulerType.COSINE:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                **lr_scheduler_kwargs,
            )

        # All other schedulers require `num_decay_steps`
        if num_decay_steps is None:
            raise ValueError(f"{name} requires `num_decay_steps`, please provide that argument.")
        if name == SchedulerType.WARMUP_STABLE_DECAY:
            return schedule_func(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_stable_steps=num_stable_steps,
                num_decay_steps=num_decay_steps,
                num_cycles=num_cycles / 2,
                min_lr_ratio=min_lr_ratio if min_lr_ratio is not None else 0.0,
                **lr_scheduler_kwargs,
            )

        return schedule_func(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_decay_steps=num_decay_steps,
            **lr_scheduler_kwargs,
        )

    def resume_from_local_or_hf_if_specified(self, accelerator: Accelerator, args: argparse.Namespace) -> bool:
        if not args.resume:
            return False

        if not args.resume_from_huggingface:
            logger.info(f"resume training from local state: {args.resume}")
            accelerator.load_state(args.resume)
            return True

        logger.info(f"resume training from huggingface state: {args.resume}")
        repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
        path_in_repo = "/".join(args.resume.split("/")[2:])
        revision = None
        repo_type = None
        if ":" in path_in_repo:
            divided = path_in_repo.split(":")
            if len(divided) == 2:
                path_in_repo, revision = divided
                repo_type = "model"
            else:
                path_in_repo, revision, repo_type = divided
        logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

        list_files = huggingface_utils.list_dir(
            repo_id=repo_id,
            subfolder=path_in_repo,
            revision=revision,
            token=args.huggingface_token,
            repo_type=repo_type,
        )

        async def download(filename) -> str:
            def task():
                return huggingface_hub.hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    repo_type=repo_type,
                    token=args.huggingface_token,
                )

            return await asyncio.get_event_loop().run_in_executor(None, task)

        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
        if len(results) == 0:
            raise ValueError(
                "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした"
            )
        dirname = os.path.dirname(results[0])
        accelerator.load_state(dirname)

        return True

    def get_bucketed_timestep(self) -> float:
        if self.num_timestep_buckets is None or self.num_timestep_buckets <= 1:
            return random.random()

        if len(self.timestep_range_pool) == 0:
            bucket_size = 1.0 / self.num_timestep_buckets
            for i in range(self.num_timestep_buckets):
                self.timestep_range_pool.append((i * bucket_size, (i + 1) * bucket_size))
            random.shuffle(self.timestep_range_pool)

        # print(f"timestep_range_pool: {self.timestep_range_pool}")
        a, b = self.timestep_range_pool.pop()
        return random.uniform(a, b)

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[List[float]],
        noise_scheduler: FlowMatchDiscreteScheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        batch_size = noise.shape[0]

        if timesteps is not None:
            timesteps = torch.tensor(timesteps, device=device)

        # This function converts uniform distribution samples to logistic distribution samples.
        # The final distribution of the samples after shifting significantly differs from the original normal distribution.
        # So we cannot use this.
        # def uniform_to_normal(t_samples: torch.Tensor) -> torch.Tensor:
        #     # Clip small values to prevent log(0)
        #     eps = 1e-7
        #     t_samples = torch.clamp(t_samples, eps, 1.0 - eps)
        #     # Convert to logit space with inverse function
        #     x_samples = torch.log(t_samples / (1.0 - t_samples))
        #     return x_samples

        def uniform_to_normal_ppF(t_uniform: torch.Tensor) -> torch.Tensor:
            """Use `torch.erfinv` to compute the inverse CDF to generate values from a normal distribution."""
            # Clip small values to prevent inf in erfinv
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

            # PPF of standard normal distribution: sqrt(2) * erfinv(2q - 1)
            term = 2.0 * t_uniform - 1.0
            x_normal = math.sqrt(2.0) * torch.erfinv(term)
            return x_normal

        def uniform_to_logsnr_ppF_pytorch(t_uniform: torch.Tensor, mean: float, std: float) -> torch.Tensor:
            """Use erfinv to compute the inverse CDF."""
            # Clip small values to prevent inf in erfinv
            eps = 1e-7
            t_uniform = torch.clamp(t_uniform, eps, 1.0 - eps)

            term = 2.0 * t_uniform - 1.0
            logsnr = mean + std * math.sqrt(2.0) * torch.erfinv(term)
            return logsnr

        if (
            args.timestep_sampling == "uniform"
            or args.timestep_sampling == "sigmoid"
            or args.timestep_sampling == "shift"
            or args.timestep_sampling == "flux_shift"
            or args.timestep_sampling == "qwen_shift"
            or args.timestep_sampling == "logsnr"
            or args.timestep_sampling == "qinglong_flux"
            or args.timestep_sampling == "qinglong_qwen"
        ):

            def compute_sampling_timesteps(org_timesteps: Optional[torch.Tensor]) -> torch.Tensor:
                def rand(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return torch.rand((bs,), device=device) if org_ts is None else org_ts

                def randn(bs: int, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    return uniform_to_normal_ppF(org_ts) if org_ts is not None else torch.randn((bs,), device=device)

                def rand_logsnr(bs: int, mean: float, std: float, org_ts: Optional[torch.Tensor] = None) -> torch.Tensor:
                    nonlocal device
                    logsnr = (
                        uniform_to_logsnr_ppF_pytorch(org_ts, mean, std)
                        if org_ts is not None
                        else torch.normal(mean=mean, std=std, size=(bs,), device=device)
                    )
                    return logsnr

                if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
                    # Simple random t-based noise sampling
                    if args.timestep_sampling == "sigmoid":
                        t = torch.sigmoid(args.sigmoid_scale * randn(batch_size, org_timesteps))
                    else:
                        t = rand(batch_size, org_timesteps)

                elif args.timestep_sampling.endswith("shift"):
                    if args.timestep_sampling == "shift":
                        shift = args.discrete_flow_shift
                    else:
                        h, w = latents.shape[-2:]
                        # we are pre-packed so must adjust for packed size
                        if args.timestep_sampling == "flux_shift":
                            mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "qwen_shift":
                            mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                        # def time_shift(mu: float, sigma: float, t: torch.Tensor):
                        #     return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma) # sigma=1.0
                        shift = math.exp(mu)

                    logits_norm = randn(batch_size, org_timesteps)
                    logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
                    t = logits_norm.sigmoid()
                    t = (t * shift) / (1 + (shift - 1) * t)

                elif args.timestep_sampling == "logsnr":
                    # https://arxiv.org/abs/2411.14793v3
                    logsnr = rand_logsnr(batch_size, args.logit_mean, args.logit_std, org_timesteps)
                    t = torch.sigmoid(-logsnr / 2)

                elif args.timestep_sampling.startswith("qinglong"):
                    # Qinglong triple hybrid sampling: mid_shift:logsnr:logsnr2 = .80:.075:.125
                    # First decide which method to use for each sample independently
                    decision_t = torch.rand((batch_size,), device=device)

                    # Create masks based on decision_t: .80 for mid_shift, 0.075 for logsnr, and 0.125 for logsnr2
                    mid_mask = decision_t < 0.80  # 80% for mid_shift
                    logsnr_mask = (decision_t >= 0.80) & (decision_t < 0.875)  # 7.5% for logsnr
                    logsnr_mask2 = decision_t >= 0.875  # 12.5% for logsnr with -logit_mean

                    # Initialize output tensor
                    t = torch.zeros((batch_size,), device=device)

                    # Generate mid_shift samples for selected indices (80%)
                    if mid_mask.any():
                        mid_count = mid_mask.sum().item()
                        h, w = latents.shape[-2:]
                        if args.timestep_sampling == "qinglong_flux":
                            mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
                        elif args.timestep_sampling == "qinglong_qwen":
                            mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
                        shift = math.exp(mu)
                        logits_norm_mid = randn(mid_count, org_timesteps[mid_mask] if org_timesteps is not None else None)
                        logits_norm_mid = logits_norm_mid * args.sigmoid_scale
                        t_mid = logits_norm_mid.sigmoid()
                        t_mid = (t_mid * shift) / (1 + (shift - 1) * t_mid)

                        t[mid_mask] = t_mid

                    # Generate logsnr samples for selected indices (7.5%)
                    if logsnr_mask.any():
                        logsnr_count = logsnr_mask.sum().item()
                        logsnr = rand_logsnr(
                            logsnr_count,
                            args.logit_mean,
                            args.logit_std,
                            org_timesteps[logsnr_mask] if org_timesteps is not None else None,
                        )
                        t_logsnr = torch.sigmoid(-logsnr / 2)

                        t[logsnr_mask] = t_logsnr

                    # Generate logsnr2 samples with -logit_mean for selected indices (12.5%)
                    if logsnr_mask2.any():
                        logsnr2_count = logsnr_mask2.sum().item()
                        logsnr2 = rand_logsnr(
                            logsnr2_count, 5.36, 1.0, org_timesteps[logsnr_mask2] if org_timesteps is not None else None
                        )
                        t_logsnr2 = torch.sigmoid(-logsnr2 / 2)

                        t[logsnr_mask2] = t_logsnr2

                return t  # 0 to 1

            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000.0
            t_min /= 1000.0
            t_max /= 1000.0

            if not args.preserve_distribution_shape:
                t = compute_sampling_timesteps(timesteps)
                t = t * (t_max - t_min) + t_min  # scale to [t_min, t_max], default [0, 1]
            else:
                max_loops = 1000
                available_t = []
                for i in range(max_loops):
                    t = None
                    if self.num_timestep_buckets is not None:
                        t = torch.tensor([self.get_bucketed_timestep() for _ in range(batch_size)], device=device)
                    t = compute_sampling_timesteps(t)
                    for t_i in t:
                        if t_min <= t_i <= t_max:
                            available_t.append(t_i)
                        if len(available_t) == batch_size:
                            break
                    if len(available_t) == batch_size:
                        break
                if len(available_t) < batch_size:
                    logger.warning(
                        f"Could not sample {batch_size} valid timesteps in {max_loops} loops / {max_loops}ループで{batch_size}個の有効なタイムステップをサンプリングできませんでした"
                    )
                    available_t = compute_sampling_timesteps(timesteps)
                else:
                    t = torch.stack(available_t, dim=0)  # [batch_size, ]

            timesteps = t * 1000.0
            t = t.view(-1, 1, 1, 1, 1) if latents.ndim == 5 else t.view(-1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise

            timesteps += 1  # 1 to 1000
        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=batch_size,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            # indices = (u * noise_scheduler.config.num_train_timesteps).long()
            t_min = args.min_timestep if args.min_timestep is not None else 0
            t_max = args.max_timestep if args.max_timestep is not None else 1000
            indices = (u * (t_max - t_min) + t_min).long()

            timesteps = noise_scheduler.timesteps[indices].to(device=device)  # 1 to 1000

            # Add noise according to flow matching.
            sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # print(f"actual timesteps: {timesteps}")
        return noisy_model_input, timesteps

    def show_timesteps(self, args: argparse.Namespace):
        N_TRY = 100000
        BATCH_SIZE = 1000
        CONSOLE_WIDTH = 64
        N_TIMESTEPS_PER_LINE = 25

        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")
        # print(f"Noise scheduler timesteps: {noise_scheduler.timesteps}")

        latents = torch.zeros(BATCH_SIZE, 1, 1, 1024 // 8, 1024 // 8, dtype=torch.float16)
        noise = torch.ones_like(latents)

        # sample timesteps
        sampled_timesteps = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(N_TRY // BATCH_SIZE)):
            bucketed_timesteps = None
            if args.num_timestep_buckets is not None and args.num_timestep_buckets > 1:
                self.num_timestep_buckets = args.num_timestep_buckets
                bucketed_timesteps = [self.get_bucketed_timestep() for _ in range(BATCH_SIZE)]

            # we use noise=1, so retured noisy_model_input is same as timestep, because `noisy_model_input = (1 - t) * latents + t * noise`
            actual_timesteps, _ = self.get_noisy_model_input_and_timesteps(
                args, noise, latents, bucketed_timesteps, noise_scheduler, "cpu", torch.float16
            )
            actual_timesteps = actual_timesteps[:, 0, 0, 0, 0] * 1000
            for t in actual_timesteps:
                t = int(t.item())
                sampled_timesteps[t] += 1

        # sample weighting
        sampled_weighting = [0] * noise_scheduler.config.num_train_timesteps
        for i in tqdm(range(len(sampled_weighting))):
            timesteps = torch.tensor([i + 1], device="cpu")
            weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, noise_scheduler, timesteps, "cpu", torch.float16)
            if weighting is None:
                weighting = torch.tensor(1.0, device="cpu")
            elif torch.isinf(weighting).any():
                weighting = torch.tensor(1.0, device="cpu")
            sampled_weighting[i] = weighting.item()

        # show results
        if args.show_timesteps == "image":
            # show timesteps with matplotlib
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
            plt.title("Sampled timesteps")
            plt.xlabel("Timestep")
            plt.ylabel("Count")

            plt.subplot(1, 2, 2)
            plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
            plt.title("Sampled loss weighting")
            plt.xlabel("Timestep")
            plt.ylabel("Weighting")

            plt.tight_layout()
            plt.show()

        else:
            sampled_timesteps = np.array(sampled_timesteps)
            sampled_weighting = np.array(sampled_weighting)

            # average per line
            sampled_timesteps = sampled_timesteps.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
            sampled_weighting = sampled_weighting.reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

            max_count = max(sampled_timesteps)
            print(f"Sampled timesteps: max count={max_count}")
            for i, t in enumerate(sampled_timesteps):
                line = f"{(i) * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
                line += "#" * int(t / max_count * CONSOLE_WIDTH)
                print(line)

            max_weighting = max(sampled_weighting)
            print(f"Sampled loss weighting: max weighting={max_weighting}")
            for i, w in enumerate(sampled_weighting):
                line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: {w:8.2f} "
                line += "#" * int(w / max_weighting * CONSOLE_WIDTH)
                print(line)

    def sample_images(self, accelerator: Accelerator, args, epoch, steps, vae, transformer, sample_parameters, dit_dtype):
        """architecture independent sample images"""
        if not should_sample_images(args, steps, epoch):
            return

        logger.info("")
        logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
        if sample_parameters is None:
            logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
            return

        distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

        # Use the unwrapped model
        transformer = accelerator.unwrap_model(transformer)
        transformer.switch_block_swap_for_inference()

        # Create a directory to save the samples
        save_dir = os.path.join(args.output_dir, "sample")
        os.makedirs(save_dir, exist_ok=True)

        # save random state to restore later
        rng_state = torch.get_rng_state()
        cuda_rng_state = None
        try:
            cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        except Exception:
            pass

        if distributed_state.num_processes <= 1:
            # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
            with torch.no_grad(), accelerator.autocast():
                for sample_parameter in sample_parameters:
                    self.sample_image_inference(
                        accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                    )
                    clean_memory_on_device(accelerator.device)
        else:
            # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
            # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
            per_process_params = []  # list of lists
            for i in range(distributed_state.num_processes):
                per_process_params.append(sample_parameters[i :: distributed_state.num_processes])

            with torch.no_grad():
                with distributed_state.split_between_processes(per_process_params) as sample_parameter_lists:
                    for sample_parameter in sample_parameter_lists[0]:
                        self.sample_image_inference(
                            accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps
                        )
                        clean_memory_on_device(accelerator.device)

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        transformer.switch_block_swap_for_training()
        clean_memory_on_device(accelerator.device)

    def sample_image_inference(self, accelerator, args, transformer, dit_dtype, vae, save_dir, sample_parameter, epoch, steps):
        """architecture independent sample images"""
        sample_steps = sample_parameter.get("sample_steps", 20)
        width = sample_parameter.get("width", 256)  # make smaller for faster and memory saving inference
        height = sample_parameter.get("height", 256)
        frame_count = sample_parameter.get("frame_count", 1)
        guidance_scale = sample_parameter.get("guidance_scale", self.default_guidance_scale)
        discrete_flow_shift = sample_parameter.get("discrete_flow_shift", 14.5)
        seed = sample_parameter.get("seed")
        prompt: str = sample_parameter.get("prompt", "")
        cfg_scale = sample_parameter.get("cfg_scale", None)  # None for architecture default
        negative_prompt = sample_parameter.get("negative_prompt", None)

        # round width and height to multiples of 8
        width = (width // 8) * 8
        height = (height // 8) * 8

        frame_count = (frame_count - 1) // 4 * 4 + 1  # 1, 5, 9, 13, ... For HunyuanVideo and Wan2.1

        if self.i2v_training:
            image_path = sample_parameter.get("image_path", None)
            if image_path is None:
                logger.error("No image_path for i2v model / i2vモデルのサンプル画像生成にはimage_pathが必要です")
                return
        else:
            image_path = None

        if self.control_training:
            control_video_path = sample_parameter.get("control_video_path", None)
            if control_video_path is None:
                logger.error(
                    "No control_video_path for control model / controlモデルのサンプル画像生成にはcontrol_video_pathが必要です"
                )
                return
        else:
            control_video_path = None

        device = accelerator.device
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            # True random sample image generation
            torch.seed()
            torch.cuda.seed()
            generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

        logger.info(f"prompt: {prompt}")
        logger.info(f"height: {height}")
        logger.info(f"width: {width}")
        logger.info(f"frame count: {frame_count}")
        logger.info(f"sample steps: {sample_steps}")
        logger.info(f"guidance scale: {guidance_scale}")
        logger.info(f"discrete flow shift: {discrete_flow_shift}")
        if seed is not None:
            logger.info(f"seed: {seed}")

        do_classifier_free_guidance = False
        if negative_prompt is not None:
            do_classifier_free_guidance = True
            logger.info(f"negative prompt: {negative_prompt}")
            logger.info(f"cfg scale: {cfg_scale}")

        if self.i2v_training:
            logger.info(f"image path: {image_path}")
        if self.control_training:
            logger.info(f"control video path: {control_video_path}")

        # inference: architecture dependent
        video = self.do_inference(
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
            frame_count,
            generator,
            do_classifier_free_guidance,
            guidance_scale,
            cfg_scale,
            image_path=image_path,
            control_video_path=control_video_path,
        )

        # Save video
        if video is None:
            logger.error("No video generated / 生成された動画がありません")
            return

        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
        seed_suffix = "" if seed is None else f"_{seed}"
        prompt_idx = sample_parameter.get("enum", 0)
        save_path = (
            f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{prompt_idx:02d}_{ts_str}{seed_suffix}"
        )

        wandb_tracker = None
        try:
            wandb_tracker = accelerator.get_tracker("wandb")  # raises ValueError if wandb is not initialized
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
        except:  # wandb 無効時
            wandb = None

        if video.shape[2] == 1:
            image_paths = save_images_grid(video, save_dir, save_path, create_subdir=False)
            if wandb_tracker is not None and wandb is not None:
                for image_path in image_paths:
                    wandb_tracker.log({f"sample_{prompt_idx}": wandb.Image(image_path)}, step=steps)
        else:
            video_path = os.path.join(save_dir, save_path) + ".mp4"
            save_videos_grid(video, video_path)
            if wandb_tracker is not None and wandb is not None:
                wandb_tracker.log({f"sample_{prompt_idx}": wandb.Video(video_path)}, step=steps)

        # Move models back to initial state
        vae.to("cpu")
        clean_memory_on_device(device)

    # region model specific

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_HUNYUAN_VIDEO_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.pos_embed_cache = {}

        self._i2v_training = args.dit_in_channels == 32  # may be changed in the future
        if self._i2v_training:
            logger.info("I2V training mode")

        self._control_training = False  # HunyuanVideo does not support control training yet

        self.default_guidance_scale = 6.0

    @property
    def i2v_training(self) -> bool:
        return self._i2v_training

    @property
    def control_training(self) -> bool:
        return self._control_training

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        text_encoder1, text_encoder2, fp8_llm = args.text_encoder1, args.text_encoder2, args.fp8_llm

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        def encode_for_text_encoder(text_encoder, is_llm=True):
            sample_prompts_te_outputs = {}  # (prompt) -> (embeds, mask)
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", None)]:
                        if p is None:
                            continue
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"cache Text Encoder outputs for prompt: {p}")

                            data_type = "video"
                            text_inputs = text_encoder.text2tokens(p, data_type=data_type)

                            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                            sample_prompts_te_outputs[p] = (prompt_outputs.hidden_state, prompt_outputs.attention_mask)

            return sample_prompts_te_outputs

        # Load Text Encoder 1 and encode
        text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else model_utils.str_to_dtype(args.text_encoder_dtype)
        logger.info(f"loading text encoder 1: {text_encoder1}")
        text_encoder_1 = text_encoder_module.load_text_encoder_1(text_encoder1, accelerator.device, fp8_llm, text_encoder_dtype)

        logger.info("encoding with Text Encoder 1")
        te_outputs_1 = encode_for_text_encoder(text_encoder_1)
        del text_encoder_1

        # Load Text Encoder 2 and encode
        logger.info(f"loading text encoder 2: {text_encoder2}")
        text_encoder_2 = text_encoder_module.load_text_encoder_2(text_encoder2, accelerator.device, text_encoder_dtype)

        logger.info("encoding with Text Encoder 2")
        te_outputs_2 = encode_for_text_encoder(text_encoder_2, is_llm=False)
        del text_encoder_2

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["llm_embeds"] = te_outputs_1[p][0]
            prompt_dict_copy["llm_mask"] = te_outputs_1[p][1]
            prompt_dict_copy["clipL_embeds"] = te_outputs_2[p][0]
            prompt_dict_copy["clipL_mask"] = te_outputs_2[p][1]

            p = prompt_dict.get("negative_prompt", None)
            if p is not None:
                prompt_dict_copy["negative_llm_embeds"] = te_outputs_1[p][0]
                prompt_dict_copy["negative_llm_mask"] = te_outputs_1[p][1]
                prompt_dict_copy["negative_clipL_embeds"] = te_outputs_2[p][0]
                prompt_dict_copy["negative_clipL_mask"] = te_outputs_2[p][1]

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
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        """architecture dependent inference"""
        device = accelerator.device
        if cfg_scale is None:
            cfg_scale = 1.0
        do_classifier_free_guidance = do_classifier_free_guidance and cfg_scale != 1.0

        # Prepare scheduler for each prompt
        scheduler = FlowMatchDiscreteScheduler(shift=discrete_flow_shift, reverse=True, solver="euler")

        # Number of inference steps for sampling
        scheduler.set_timesteps(sample_steps, device=device)
        timesteps = scheduler.timesteps

        # Calculate latent video length based on VAE version
        if "884" in VAE_VER:
            latent_video_length = (frame_count - 1) // 4 + 1
        elif "888" in VAE_VER:
            latent_video_length = (frame_count - 1) // 8 + 1
        else:
            latent_video_length = frame_count

        # Get embeddings
        prompt_embeds = sample_parameter["llm_embeds"].to(device=device, dtype=dit_dtype)
        prompt_mask = sample_parameter["llm_mask"].to(device=device)
        prompt_embeds_2 = sample_parameter["clipL_embeds"].to(device=device, dtype=dit_dtype)

        if do_classifier_free_guidance:
            negative_prompt_embeds = sample_parameter["negative_llm_embeds"].to(device=device, dtype=dit_dtype)
            negative_prompt_mask = sample_parameter["negative_llm_mask"].to(device=device)
            negative_prompt_embeds_2 = sample_parameter["negative_clipL_embeds"].to(device=device, dtype=dit_dtype)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_mask = torch.cat([negative_prompt_mask, prompt_mask], dim=0)
            prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2], dim=0)

        num_channels_latents = 16  # transformer.config.in_channels
        vae_scale_factor = 2 ** (4 - 1)  # Assuming 4 VAE blocks

        # Initialize latents
        shape_or_frame = (
            1,
            num_channels_latents,
            1,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        latents = []
        for _ in range(latent_video_length):
            latents.append(torch.randn(shape_or_frame, generator=generator, device=device, dtype=dit_dtype))
        latents = torch.cat(latents, dim=2)

        if self.i2v_training:
            # Move VAE to the appropriate device for sampling
            vae.to(device)
            vae.eval()

            image = Image.open(image_path)
            image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float()  # 1, C, 1, H, W
            image = image / 255.0

            logger.info("Encoding image to latents")
            image_latents = encode_to_latents(args, image, device)  # 1, C, 1, H, W
            image_latents = image_latents.to(device=device, dtype=dit_dtype)

            vae.to("cpu")
            clean_memory_on_device(device)

            zero_latents = torch.zeros_like(latents)
            zero_latents[:, :, :1, :, :] = image_latents
            image_latents = zero_latents
        else:
            image_latents = None

        # Guidance scale
        guidance_expand = torch.tensor([guidance_scale * 1000.0], dtype=torch.float32, device=device).to(dit_dtype)

        # Get rotary positional embeddings
        freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(transformer, latents.shape[2:])
        freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)

        # Wrap the inner loop with tqdm to track progress over timesteps
        prompt_idx = sample_parameter.get("enum", 0)
        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, desc=f"Sampling timesteps for prompt {prompt_idx + 1}")):
                latents_input = scheduler.scale_model_input(latents, t)

                if do_classifier_free_guidance:
                    latents_input = torch.cat([latents_input, latents_input], dim=0)  # 2, C, F, H, W

                if image_latents is not None:
                    latents_image_input = (
                        image_latents if not do_classifier_free_guidance else torch.cat([image_latents, image_latents], dim=0)
                    )
                    latents_input = torch.cat([latents_input, latents_image_input], dim=1)  # 1 or 2, C*2, F, H, W

                noise_pred = transformer(
                    latents_input,
                    t.repeat(latents.shape[0]).to(device=device, dtype=dit_dtype),
                    text_states=prompt_embeds,
                    text_mask=prompt_mask,
                    text_states_2=prompt_embeds_2,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                    guidance=guidance_expand,
                    return_dict=True,
                )["x"]

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
            latents = latents / vae.config.scaling_factor + vae.config.shift_factor
        else:
            latents = latents / vae.config.scaling_factor

        latents = latents.to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.cpu().float()

        return video

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device="cpu", vae_path=vae_path)

        if args.vae_chunk_size is not None:
            vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
            logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
        if args.vae_spatial_tile_sample_min_size is not None:
            vae.enable_spatial_tiling(True)
            vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
            vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
        elif args.vae_tiling:
            vae.enable_spatial_tiling(True)

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
        transformer = load_transformer(dit_path, attn_mode, split_attn, loading_device, dit_weight_dtype, args.dit_in_channels)

        if args.img_in_txt_in_offloading:
            logger.info("Enable offloading img_in and txt_in to CPU")
            transformer.enable_img_in_txt_in_offloading()

        return transformer

    def compile_transformer(self, args, transformer):
        transformer: HYVideoDiffusionTransformer = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.double_blocks, transformer.single_blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        latents = latents * vae_module.SCALING_FACTOR
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer_arg,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        transformer: HYVideoDiffusionTransformer = transformer_arg
        bsz = latents.shape[0]

        # I2V training
        if self.i2v_training:
            image_latents = torch.zeros_like(latents)
            image_latents[:, :, :1, :, :] = latents[:, :, :1, :, :]
            noisy_model_input = torch.cat([noisy_model_input, image_latents], dim=1)  # concat along channel dim

        # ensure guidance_scale in args is float
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)  # , dtype=dit_dtype)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        pos_emb_shape = latents.shape[1:]
        if pos_emb_shape not in self.pos_embed_cache:
            freqs_cos, freqs_sin = get_rotary_pos_embed_by_shape(transformer, latents.shape[2:])
            # freqs_cos = freqs_cos.to(device=accelerator.device, dtype=dit_dtype)
            # freqs_sin = freqs_sin.to(device=accelerator.device, dtype=dit_dtype)
            self.pos_embed_cache[pos_emb_shape] = (freqs_cos, freqs_sin)
        else:
            freqs_cos, freqs_sin = self.pos_embed_cache[pos_emb_shape]

        # call DiT
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        with accelerator.autocast():
            model_pred = transformer(
                noisy_model_input,
                timesteps,
                text_states=batch["llm"],
                text_mask=batch["llm_mask"],
                text_states_2=batch["clipL"],
                freqs_cos=freqs_cos,
                freqs_sin=freqs_sin,
                guidance=guidance_vec,
                return_dict=False,
            )

        # flow matching loss
        target = noise - latents

        return model_pred, target

    # endregion model specific

    def _check_gpu_state(self, checkpoint_name: str, accelerator: Accelerator):
        """Check GPU state at a specific checkpoint in the training sequence."""
        log_file = "debug_batch.txt"
        with open(log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"GPU STATE CHECK: {checkpoint_name}\n")
            f.write(f"{'='*80}\n")
            
            if not torch.cuda.is_available():
                f.write("GPU not available\n")
                return
            
            device = accelerator.device
            
            # Test 1: Can GPU create tensors?
            try:
                test_tensor = torch.randn(100, device=device)
                test_max = test_tensor.abs().max().item()
                if test_max > 1e-6:
                    f.write(f"GPU tensor creation: SUCCESS (test max={test_max:.12e})\n")
                else:
                    f.write(f"GPU tensor creation: FAILED - GPU cannot create valid tensors!\n")
                    f.write(f"  Test tensor max={test_max:.12e} (should be > 0)\n")
                    f.write(f"  CRITICAL: GPU is corrupted at checkpoint '{checkpoint_name}'\n")
            except Exception as e:
                f.write(f"GPU tensor creation: EXCEPTION - {str(e)}\n")
            
            # Test 2: Can simple CPU->GPU transfer work?
            try:
                test_cpu = torch.randn(100, device='cpu')
                test_cpu_max = test_cpu.abs().max().item()
                test_gpu = test_cpu.to(device, non_blocking=False)
                test_gpu_max = test_gpu.abs().max().item()
                
                if test_gpu_max > 1e-6:
                    f.write(f"Simple CPU->GPU transfer: SUCCESS\n")
                    f.write(f"  CPU max={test_cpu_max:.12e}, GPU max={test_gpu_max:.12e}\n")
                else:
                    f.write(f"Simple CPU->GPU transfer: FAILED\n")
                    f.write(f"  CPU max={test_cpu_max:.12e}, GPU max={test_gpu_max:.12e}\n")
                    f.write(f"  CRITICAL: GPU transfers broken at checkpoint '{checkpoint_name}'\n")
            except Exception as e:
                f.write(f"Simple CPU->GPU transfer: EXCEPTION - {str(e)}\n")
            
            # GPU memory state
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            f.write(f"GPU memory: allocated={allocated:.4f} GB, reserved={reserved:.4f} GB, max={max_allocated:.4f} GB\n")
            
            f.write(f"{'='*80}\n\n")
    
    def train(self, args):
        if torch.cuda.is_available():
            if args.cuda_allow_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("Enabled TF32 on CUDA / CUDAでTF32を有効化しました")
            if args.cuda_cudnn_benchmark:
                torch.backends.cudnn.benchmark = True
                logger.info("Enabled cuDNN benchmark / cuDNNベンチマークを有効化しました")

        # check required arguments
        if args.dataset_config is None:
            raise ValueError("dataset_config is required / dataset_configが必要です")
        if args.dit is None:
            raise ValueError("path to DiT model is required / DiTモデルのパスが必要です")
        assert not args.fp8_scaled or args.fp8_base, "fp8_scaled requires fp8_base / fp8_scaledはfp8_baseが必要です"

        if args.sage_attn:
            raise ValueError(
                "SageAttention doesn't support training currently. Please use `--sdpa` or `--xformers` etc. instead."
                " / SageAttentionは現在学習をサポートしていないようです。`--sdpa`や`--xformers`などの他のオプションを使ってください"
            )

        if args.disable_numpy_memmap:
            logger.info(
                "Disabling numpy memory mapping for model loading (for Wan, FramePack and Qwen-Image). This may lead to higher memory usage but can speed up loading in some cases."
                " / モデル読み込み時のnumpyメモリマッピングを無効にします（Wan、FramePack、Qwen-Imageでのみ有効）。これによりメモリ使用量が増える可能性がありますが、場合によっては読み込みが高速化されることがあります"
            )

        # Set default output_dir if not specified (needed for epoch saves)
        if args.output_dir is None:
            args.output_dir = "."

        # check model specific arguments
        self.handle_model_specific_args(args)

        # show timesteps for debugging
        if args.show_timesteps:
            self.show_timesteps(args)
            return

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        # setup_logging(args, reset=True)

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # Load dataset config
        if args.num_timestep_buckets is not None:
            logger.info(f"Using timestep bucketing. Number of buckets: {args.num_timestep_buckets}")
        self.num_timestep_buckets = args.num_timestep_buckets  # None or int, None makes all the behavior same as before

        current_epoch = Value("i", 0)  # shared between processes

        blueprint_generator = BlueprintGenerator(ConfigSanitizer())
        logger.info(f"Load dataset config from {args.dataset_config}")
        user_config = config_utils.load_user_config(args.dataset_config)
        blueprint = blueprint_generator.generate(user_config, args, architecture=self.architecture)
        train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
            blueprint.dataset_group, training=True, num_timestep_buckets=self.num_timestep_buckets, shared_epoch=current_epoch
        )

        if train_dataset_group.num_train_items == 0:
            raise ValueError(
                "No training items found in the dataset. Please ensure that the latent/Text Encoder cache has been created beforehand."
                " / データセットに学習データがありません。latent/Text Encoderキャッシュを事前に作成したか確認してください"
            )

        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = collator_class(current_epoch, ds_for_collator)

        # prepare accelerator
        logger.info("preparing accelerator")
        accelerator = prepare_accelerator(args)
        if args.mixed_precision is None:
            args.mixed_precision = accelerator.mixed_precision
            logger.info(f"mixed precision set to {args.mixed_precision} / mixed precisionを{args.mixed_precision}に設定")
        is_main_process = accelerator.is_main_process
        
        # Check GPU state after Accelerator creation
        self._check_gpu_state("1. After Accelerator Creation", accelerator)

        # prepare dtype
        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # HunyuanVideo: bfloat16 or float16, Wan2.1: bfloat16
        dit_dtype = torch.bfloat16 if args.dit_dtype is None else model_utils.str_to_dtype(args.dit_dtype)
        dit_weight_dtype = (None if args.fp8_scaled else torch.float8_e4m3fn) if args.fp8_base else dit_dtype
        logger.info(f"DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

        # get embedding for sampling images
        vae_dtype = torch.float16 if args.vae_dtype is None else model_utils.str_to_dtype(args.vae_dtype)
        sample_parameters = None
        vae = None
        if args.sample_prompts:
            sample_parameters = self.process_sample_prompts(args, accelerator, args.sample_prompts)

            # Load VAE model for sampling images: VAE is loaded to cpu to save gpu memory
            vae = self.load_vae(args, vae_dtype=vae_dtype, vae_path=args.vae)
            vae.requires_grad_(False)
            vae.eval()

        # load DiT model
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        self.blocks_to_swap = blocks_to_swap
        loading_device = "cpu" if blocks_to_swap > 0 else accelerator.device

        logger.info(f"Loading DiT model from {args.dit}")
        if args.sdpa:
            attn_mode = "torch"
        elif args.flash_attn:
            attn_mode = "flash"
        elif args.sage_attn:
            attn_mode = "sageattn"
        elif args.xformers:
            attn_mode = "xformers"
        elif args.flash3:
            attn_mode = "flash3"
        else:
            raise ValueError(
                "either --sdpa, --flash-attn, --flash3, --sage-attn or --xformers must be specified / --sdpa, --flash-attn, --flash3, --sage-attn, --xformersのいずれかを指定してください"
            )
        transformer = self.load_transformer(
            accelerator, args, args.dit, attn_mode, args.split_attn, loading_device, dit_weight_dtype
        )
        transformer.eval()
        transformer.requires_grad_(False)
        
        # Check GPU state after model load
        self._check_gpu_state("2. After Model Load", accelerator)

        if blocks_to_swap > 0:
            logger.info(
                f"enable swap {blocks_to_swap} blocks to CPU from device: {accelerator.device}, use pinned memory: {args.use_pinned_memory_for_block_swap}"
            )
            transformer.enable_block_swap(
                blocks_to_swap, accelerator.device, supports_backward=True, use_pinned_memory=args.use_pinned_memory_for_block_swap
            )
            transformer.move_to_device_except_swap_blocks(accelerator.device)

        # load network model for differential training
        sys.path.append(os.path.dirname(__file__))
        
        # ROCm WORKAROUND: Use PEFT for LoRA training on Windows + ROCm to avoid tensor transfer bug
        # PEFT has been verified to work correctly and avoids the known ROCm bug (GitHub issue #3874)
        if args.use_peft:
            accelerator.print("Using Hugging Face PEFT for LoRA training (ROCm compatibility mode)")
            try:
                import musubi_tuner.networks.peft_lora as network_module
                accelerator.print("PEFT LoRA module loaded successfully")
            except ImportError as e:
                accelerator.print(f"ERROR: PEFT not available: {e}")
                accelerator.print("Install with: pip install peft")
                return
        else:
            accelerator.print("import network module:", args.network_module)
            network_module: lora_module = importlib.import_module(args.network_module)  # actual module may be different

        # For PEFT, base_weights merging may need special handling
        if args.base_weights is not None:
            # if base_weights is specified, merge the weights to DiT model
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                weights_sd = load_file(weight_path)
                if args.use_peft:
                    # PEFT base weights merging - may need special handling
                    logger.warning("PEFT base_weights merging not yet fully implemented - weights may not merge correctly")
                module = network_module.create_arch_network_from_weights(
                    multiplier, weights_sd, unet=transformer, for_inference=True
                )
                module.merge_to(None, transformer, weights_sd, weight_dtype, "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if args.dim_from_weights:
            logger.info(f"Loading network from weights: {args.dim_from_weights}")
            weights_sd = load_file(args.dim_from_weights)
            network, _ = network_module.create_arch_network_from_weights(1, weights_sd, unet=transformer)
        else:
            # We use the name create_arch_network for compatibility with LyCORIS
            if hasattr(network_module, "create_arch_network"):
                network = network_module.create_arch_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    None,
                    transformer,
                    neuron_dropout=args.network_dropout,
                    **net_kwargs,
                )
            else:
                # LyCORIS compatibility
                network = network_module.create_network(
                    1.0,
                    args.network_dim,
                    args.network_alpha,
                    vae,
                    None,
                    transformer,
                    **net_kwargs,
                )
        if network is None:
            return

        if hasattr(network_module, "prepare_network"):
            network.prepare_network(args)

        # apply network to DiT
        # With PEFT, the model is already wrapped, but we call apply_to for compatibility
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        
        # For PEFT, the transformer is already wrapped, so we need to use the PEFT model
        if args.use_peft:
            # PEFT wraps the model, so we need to use the wrapped model for training
            # The network.peft_model is the wrapped transformer
            # We'll use it in the training loop instead of the original transformer
            logger.info("PEFT LoRA applied - using PEFT-wrapped model for training")

        if args.network_weights is not None:
            # FIXME consider alpha of weights: this assumes that the alpha is not changed
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            transformer.enable_gradient_checkpointing(args.gradient_checkpointing_cpu_offload)
            network.enable_gradient_checkpointing()  # may have no effect

        # prepare optimizer, data loader etc.
        accelerator.print("prepare optimizer, data loader etc.")

        trainable_params, lr_descriptions = network.prepare_optimizer_params(unet_lr=args.learning_rate)
        optimizer_name, optimizer_args, optimizer, optimizer_train_fn, optimizer_eval_fn = self.get_optimizer(
            args, trainable_params
        )

        # prepare dataloader

        # num workers for data loader: if 0, persistent_workers is not available
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        # persistent_workers requires num_workers > 0
        # On Windows with ROCm, num_workers=0 avoids multiprocessing issues that cause tensor zeroing
        use_persistent_workers = args.persistent_data_loader_workers and n_workers > 0

        # ROCm WORKAROUND: Always disable pin_memory when using direct GPU loading
        # pin_memory only works for CPU tensors, but we're loading directly to GPU to bypass CPU->GPU transfer
        # We'll set MUSUBI_TRAIN_DEVICE after DataLoader creation, so we always disable pin_memory for ROCm
        # This prevents the "cannot pin 'torch.cuda.FloatTensor'" error
        use_pin_memory = False  # Always disabled for ROCm direct GPU loading
        if os.environ.get("HIP_DISABLE_IPC") == "1":
            logger.info("ROCm detected: Disabling pin_memory (tensors loaded directly to GPU, pin_memory only works for CPU tensors)")
        else:
            logger.info("Disabling pin_memory (direct GPU loading enabled)")
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=use_persistent_workers,
            pin_memory=use_pin_memory,
        )

        # calculate max_train_steps
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # send max_train_steps to train_dataset_group
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # prepare lr_scheduler
        lr_scheduler = self.get_lr_scheduler(args, optimizer, accelerator.num_processes)

        # prepare training model. accelerator does some magic here

        # experimental feature: train the model with gradients in fp16/bf16
        network_dtype = torch.float32
        args.full_fp16 = args.full_bf16 = False  # temporary disabled because stochastic rounding is not supported yet
        if args.full_fp16:
            assert args.mixed_precision == "fp16", (
                "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            )
            accelerator.print("enable full fp16 training.")
            network_dtype = weight_dtype
            network.to(network_dtype)
        elif args.full_bf16:
            assert args.mixed_precision == "bf16", (
                "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            )
            accelerator.print("enable full bf16 training.")
            network_dtype = weight_dtype
            network.to(network_dtype)

        if dit_weight_dtype != dit_dtype and dit_weight_dtype is not None:
            logger.info(f"casting model to {dit_weight_dtype}")
            transformer.to(dit_weight_dtype)

        # For PEFT, we need to prepare the PEFT model instead of the original transformer
        if args.use_peft and hasattr(network, 'peft_model'):
            # PEFT model is already wrapped, prepare it with Accelerate
            if blocks_to_swap > 0:
                transformer = accelerator.prepare(network.peft_model, device_placement=[not blocks_to_swap > 0])
                accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
                accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
            else:
                transformer = accelerator.prepare(network.peft_model)
            logger.info("PEFT model prepared with Accelerate")
        else:
            # Standard path for non-PEFT
            if blocks_to_swap > 0:
                transformer = accelerator.prepare(transformer, device_placement=[not blocks_to_swap > 0])
                accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)  # reduce peak memory usage
                accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
            else:
                transformer = accelerator.prepare(transformer)
        
        # Check GPU state after Accelerate prepare
        self._check_gpu_state("3. After Accelerate Prepare", accelerator)

        if args.compile:
            transformer = self.compile_transformer(args, transformer)
            transformer.__dict__["_orig_mod"] = transformer  # for annoying accelerator checks

        # ROCm WORKAROUND: Don't prepare DataLoader with Accelerate
        # Accelerate's DataLoader wrapper corrupts tensors on ROCm (they become zeros)
        # Use raw PyTorch DataLoader instead and handle device placement manually
        # This bypasses Accelerate's potentially buggy DataLoader code
        # 
        # Research findings: Accelerate's DataLoader wrapper is incompatible with ROCm on Windows.
        # Even with device_placement=False, Accelerate may corrupt tensors before we can move them.
        # By not preparing the DataLoader, we use raw PyTorch DataLoader which should work correctly.
        # See docs/ROCM_TENSOR_CORRUPTION_RESEARCH.md for details.
        network, optimizer, lr_scheduler = accelerator.prepare(
            network, optimizer, lr_scheduler
        )
        # DO NOT prepare train_dataloader - use it directly from PyTorch
        # This avoids Accelerate's DataLoader wrapper which corrupts tensors on ROCm
        # We'll handle device placement manually in the training loop
        training_model = network
        
        # ROCm WORKAROUND: Do NOT set MUSUBI_TRAIN_DEVICE - loading directly to GPU corrupts tensors
        # Cache files are valid (verified), but loading directly to GPU via get_tensor(device=cuda) 
        # uses .to(device) which corrupts tensors on ROCm. We must load to CPU first.
        # The dataset will load to CPU, then we'll try to move to GPU in the training loop.
        logger.info(f"Loading cache files to CPU (GPU loading corrupts tensors on ROCm)")

        if args.gradient_checkpointing:
            transformer.train()
        else:
            transformer.eval()

        accelerator.unwrap_model(network).prepare_grad_etc(transformer)

        # Patch accelerator scaler for fp16 training (both full_fp16 and mixed_precision fp16)
        # The scaler tries to unscale fp16 gradients which isn't supported, so we need to allow it
        if args.full_fp16 or args.mixed_precision == "fp16":
            if accelerator.scaler is not None:
                org_unscale_grads = accelerator.scaler._unscale_grads_

                def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
                    return org_unscale_grads(optimizer, inv_scale, found_inf, True)

                accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
                
                # Note: We don't modify _scale directly as it breaks the scaler's internal state (_growth_tracker)
                # The NaN prevention measures in the model forward pass should handle numerical stability

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process:  # or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resume from local or huggingface. accelerator.step is set
        self.resume_from_local_or_hf_if_specified(accelerator, args)  # accelerator.load_state(args.resume)

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # 学習する
        # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train items / 学習画像、動画数: {train_dataset_group.num_train_items}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_num_train_items": train_dataset_group.num_train_items,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_checkpointing_cpu_offload": args.gradient_checkpointing_cpu_offload,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            SS_METADATA_KEY_BASE_MODEL_VERSION: self.architecture_full_name,
            # "ss_network_module": args.network_module,
            # "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            # "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            SS_METADATA_KEY_NETWORK_MODULE: args.network_module,
            SS_METADATA_KEY_NETWORK_DIM: args.network_dim,
            SS_METADATA_KEY_NETWORK_ALPHA: args.network_alpha,
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_seed": args.seed,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            # "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_fp8_base": bool(args.fp8_base),
            # "ss_fp8_llm": bool(args.fp8_llm), # remove this because this is only for HuanyuanVideo TODO set architecure dependent metadata
            "ss_full_fp16": bool(args.full_fp16),
            "ss_full_bf16": bool(args.full_bf16),
            "ss_weighting_scheme": args.weighting_scheme,
            "ss_logit_mean": args.logit_mean,
            "ss_logit_std": args.logit_std,
            "ss_mode_scale": args.mode_scale,
            "ss_guidance_scale": args.guidance_scale,
            "ss_timestep_sampling": args.timestep_sampling,
            "ss_sigmoid_scale": args.sigmoid_scale,
            "ss_discrete_flow_shift": args.discrete_flow_shift,
        }

        datasets_metadata = []
        # tag_frequency = {}  # merge tag frequency for metadata editor # TODO support tag frequency
        for dataset in train_dataset_group.datasets:
            dataset_metadata = dataset.get_metadata()
            datasets_metadata.append(dataset_metadata)

        metadata["ss_datasets"] = json.dumps(datasets_metadata)

        # add extra args
        if args.network_args:
            # metadata["ss_network_args"] = json.dumps(net_kwargs)
            metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(net_kwargs)

        # model name and hash
        # calculate hash takes time, so we omit it for now
        if args.dit is not None:
            # logger.info(f"calculate hash for DiT model: {args.dit}")
            logger.info(f"set DiT model name for metadata: {args.dit}")
            sd_model_name = args.dit
            if os.path.exists(sd_model_name):
                # metadata["ss_sd_model_hash"] = model_utils.model_hash(sd_model_name)
                # metadata["ss_new_sd_model_hash"] = model_utils.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            # logger.info(f"calculate hash for VAE model: {args.vae}")
            logger.info(f"set VAE model name for metadata: {args.vae}")
            vae_name = args.vae
            if os.path.exists(vae_name):
                # metadata["ss_vae_hash"] = model_utils.model_hash(vae_name)
                # metadata["ss_new_vae_hash"] = model_utils.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_utils.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        # TODO skip until initial step
        progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")

        epoch_to_start = 0
        global_step = 0
        noise_scheduler = FlowMatchDiscreteScheduler(shift=args.discrete_flow_shift, reverse=True, solver="euler")

        loss_recorder = train_utils.LossRecorder()
        del train_dataset_group

        # function for saving/removing
        save_dtype = dit_dtype

        def save_model(ckpt_name: str, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata

            title = args.metadata_title if args.metadata_title is not None else args.output_name
            if args.min_timestep is not None or args.max_timestep is not None:
                min_time_step = args.min_timestep if args.min_timestep is not None else 0
                max_time_step = args.max_timestep if args.max_timestep is not None else 1000
                md_timesteps = (min_time_step, max_time_step)
            else:
                md_timesteps = None

            sai_metadata = sai_model_spec.build_metadata(
                None,
                self.architecture,
                time.time(),
                title,
                args.metadata_reso,
                args.metadata_author,
                args.metadata_description,
                args.metadata_license,
                args.metadata_tags,
                timesteps=md_timesteps,
                custom_arch=args.metadata_arch,
            )

            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_utils.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # For --sample_at_first
        if should_sample_images(args, global_step, epoch=0):
            optimizer_eval_fn()
            self.sample_images(accelerator, args, 0, global_step, vae, transformer, sample_parameters, dit_dtype)
            optimizer_train_fn()
        if len(accelerator.trackers) > 0:
            # log empty object to commit the sample images to wandb
            accelerator.log({}, step=0)

        # training loop

        # log device and dtype for each model
        logger.info(f"DiT dtype: {transformer.dtype}, device: {transformer.device}")

        clean_memory_on_device(accelerator.device)

        optimizer_train_fn()  # Set training mode

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch + 1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(transformer)
            
            # Check GPU state before training loop starts (only on first epoch)
            if epoch == epoch_to_start:
                self._check_gpu_state("4. Before Training Loop", accelerator)

            for step, batch in enumerate(train_dataloader):
                # torch.compiler.cudagraph_mark_step_begin() # for cudagraphs
                
                # Check GPU state at first step of first epoch
                if epoch == epoch_to_start and step == 0:
                    self._check_gpu_state("5. At Training Loop Start (Step 0)", accelerator)

                # DEBUG: Comprehensive batch logging
                log_file = "debug_batch.txt"
                with open(log_file, "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"TRAINING STEP {step} - BATCH ANALYSIS\n")
                    f.write(f"{'='*80}\n")
                    f.write(f"Batch keys: {list(batch.keys())}\n")
                    
                    # Log each tensor in batch
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            f.write(f"\nBatch['{key}']:\n")
                            f.write(f"  shape={value.shape}\n")
                            f.write(f"  dtype={value.dtype}\n")
                            f.write(f"  device={value.device}\n")
                            f.write(f"  is_contiguous={value.is_contiguous()}\n")
                            f.write(f"  max={value.abs().max().item():.12e}\n")
                            f.write(f"  min={value.min().item():.12e}\n")
                            f.write(f"  mean={value.mean().item():.12e}\n")
                            f.write(f"  zero_count={(value == 0).sum().item()}/{value.numel()}\n")
                        elif isinstance(value, list):
                            f.write(f"\nBatch['{key}']: list with {len(value)} items\n")
                            for i, item in enumerate(value[:3]):  # First 3 items
                                if isinstance(item, torch.Tensor):
                                    f.write(f"  [{i}]: shape={item.shape}, dtype={item.dtype}, device={item.device}, max={item.abs().max().item():.12e}\n")
                        else:
                            f.write(f"\nBatch['{key}']: {type(value)}\n")
                    
                    f.write(f"{'='*80}\n\n")
                
                if step < 3:
                    logger.info(f"DEBUG Step {step}: Batch keys: {list(batch.keys())}")
                
                if "latents" not in batch:
                    with open(log_file, "a") as f:
                        f.write(f"Step {step}: ERROR - 'latents' key not found! Available keys: {list(batch.keys())}\n")
                    logger.error(f"CRITICAL Step {step}: 'latents' key not found in batch! Available keys: {list(batch.keys())}")
                    # Try to find any latent-like key
                    latent_keys = [k for k in batch.keys() if "latent" in k.lower()]
                    if latent_keys:
                        logger.error(f"  Found potential latent keys: {latent_keys}")
                    continue
                
                # ROCm WORKAROUND: Since we're not using Accelerate's DataLoader wrapper,
                # tensors should be clean on CPU (not corrupted by Accelerate).
                # Move them to GPU manually. If this still fails, it's a deeper ROCm issue.
                # 
                # Note: By bypassing Accelerate's DataLoader wrapper, we avoid the tensor
                # corruption that was happening when Accelerate tried to handle device placement.
                # The tensors should now be valid on CPU and move to GPU correctly.
                
                # Helper function to move tensor to GPU
                # ROCm WORKAROUND: Try multiple methods including pinned memory and chunked transfer
                def move_to_gpu(value, name=""):
                    # Handle non-tensor values (lists, None, etc.)
                    if not isinstance(value, torch.Tensor):
                        return value
                    
                    if value.device.type == "cpu":
                        max_val_cpu = value.abs().max().item()
                        if max_val_cpu < 1e-6:
                            # Already zeros, just move it
                            return value.to(accelerator.device, non_blocking=False)
                        
                        # CRITICAL: Comprehensive logging before transfer
                        log_file = "debug_batch.txt"
                        with open(log_file, "a") as f:
                            f.write(f"\n{'='*80}\n")
                            f.write(f"Step {step}: move_to_gpu({name}) - COMPREHENSIVE DIAGNOSTICS\n")
                            f.write(f"{'='*80}\n")
                            
                            # Tensor properties
                            f.write(f"CPU TENSOR PROPERTIES:\n")
                            f.write(f"  shape={value.shape}\n")
                            f.write(f"  dtype={value.dtype}\n")
                            f.write(f"  device={value.device}\n")
                            f.write(f"  is_contiguous={value.is_contiguous()}\n")
                            f.write(f"  is_pinned={value.is_pinned()}\n")
                            f.write(f"  requires_grad={value.requires_grad}\n")
                            f.write(f"  numel={value.numel()}\n")
                            f.write(f"  element_size={value.element_size()} bytes\n")
                            f.write(f"  total_size={value.numel() * value.element_size() / 1024**2:.2f} MB\n")
                            
                            # Memory layout
                            f.write(f"  storage_offset={value.storage_offset()}\n")
                            f.write(f"  stride={value.stride()}\n")
                            if hasattr(value, 'data_ptr'):
                                f.write(f"  data_ptr={hex(value.data_ptr())}\n")
                            
                            # Tensor statistics
                            f.write(f"CPU TENSOR STATISTICS:\n")
                            f.write(f"  max={max_val_cpu:.12e}\n")
                            f.write(f"  min={value.min().item():.12e}\n")
                            f.write(f"  mean={value.mean().item():.12e}\n")
                            f.write(f"  std={value.std().item():.12e}\n")
                            f.write(f"  abs_max={value.abs().max().item():.12e}\n")
                            
                            # Check for zeros/NaN/Inf
                            zero_count = (value == 0).sum().item()
                            nan_count = torch.isnan(value).sum().item()
                            inf_count = torch.isinf(value).sum().item()
                            f.write(f"  zero_count={zero_count}/{value.numel()} ({100*zero_count/value.numel():.2f}%)\n")
                            f.write(f"  nan_count={nan_count}\n")
                            f.write(f"  inf_count={inf_count}\n")
                            
                            # GPU memory state
                            if torch.cuda.is_available():
                                allocated_before = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                                reserved_before = torch.cuda.memory_reserved(accelerator.device) / 1024**3
                                max_allocated = torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
                                f.write(f"GPU MEMORY STATE:\n")
                                f.write(f"  allocated={allocated_before:.4f} GB\n")
                                f.write(f"  reserved={reserved_before:.4f} GB\n")
                                f.write(f"  max_allocated={max_allocated:.4f} GB\n")
                                
                                # GPU properties
                                props = torch.cuda.get_device_properties(accelerator.device)
                                f.write(f"  total_memory={props.total_memory / 1024**3:.4f} GB\n")
                                f.write(f"  memory_free={props.total_memory / 1024**3 - reserved_before:.4f} GB\n")
                            
                            # Environment
                            f.write(f"ENVIRONMENT:\n")
                            f.write(f"  HIP_DISABLE_IPC={os.environ.get('HIP_DISABLE_IPC', 'not set')}\n")
                            f.write(f"  HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}\n")
                            f.write(f"  HIP_LAUNCH_BLOCKING={os.environ.get('HIP_LAUNCH_BLOCKING', 'not set')}\n")
                            f.write(f"  PyTorch version={torch.__version__}\n")
                            if hasattr(torch.version, 'hip'):
                                f.write(f"  ROCm version={torch.version.hip}\n")
                            if torch.cuda.is_available():
                                f.write(f"  GPU={torch.cuda.get_device_name(0)}\n")
                            
                            # Training context
                            f.write(f"TRAINING CONTEXT:\n")
                            f.write(f"  step={step}\n")
                            f.write(f"  tensor_name={name}\n")
                            f.write(f"  accelerator_device={accelerator.device}\n")
                            
                            f.write(f"{'='*80}\n")
                        
                        # CRITICAL: Make tensor contiguous and clone it first
                        # Non-contiguous or view tensors might cause issues on ROCm
                        if not value.is_contiguous():
                            logger.warning(f"Step {step}: Tensor {name} is not contiguous, making it contiguous...")
                            value = value.contiguous()
                        
                        # Clone to ensure we have a fresh copy (breaks any view relationships)
                        value_clone = value.clone()
                        max_val_clone = value_clone.abs().max().item()
                        
                        if max_val_clone < 1e-6:
                            logger.error(f"Step {step}: CRITICAL - Cloned tensor is zeros! Original max={max_val_cpu:.6e}")
                            with open("debug_batch.txt", "a") as f:
                                f.write(f"Step {step}: CRITICAL ERROR - Tensor {name} became zeros after clone()!\n")
                                f.write(f"  Original CPU max: {max_val_cpu:.6e}\n")
                                f.write(f"  Cloned CPU max: {max_val_clone:.6e}\n")
                            raise RuntimeError(f"Tensor {name} became zeros after clone() - this is a severe ROCm bug")
                        
                        # Method 1: Try with pinned memory (if available)
                        try:
                            if step < 3:
                                logger.warning(f"Step {step}: Trying Method 1 (pinned memory) for {name}...")
                            # Pin the CPU tensor to page-locked memory first
                            if not value_clone.is_pinned():
                                pinned_value = value_clone.pin_memory()
                            else:
                                pinned_value = value_clone
                            
                            tensor_gpu = pinned_value.to(accelerator.device, non_blocking=True)
                            # Synchronize to ensure transfer is complete
                            torch.cuda.synchronize(accelerator.device)
                            max_val_gpu = tensor_gpu.abs().max().item()
                            
                            if step < 3:
                                logger.warning(f"Step {step}: Method 1 result for {name}: CPU max={max_val_cpu:.6e}, Clone max={max_val_clone:.6e}, GPU max={max_val_gpu:.6e}")
                            if max_val_gpu > 1e-6:
                                if step < 3:
                                    logger.info(f"Step {step}: Method 1 (pinned memory) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                                    with open("debug_batch.txt", "a") as f:
                                        f.write(f"Step {step}: Method 1 SUCCESS for {name} - CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}\n")
                                return tensor_gpu
                            else:
                                # CRITICAL: Log detailed failure information
                                with open("debug_batch.txt", "a") as f:
                                    f.write(f"\n{'!'*80}\n")
                                    f.write(f"Step {step}: Method 1 FAILED for {name}\n")
                                    f.write(f"{'!'*80}\n")
                                    f.write(f"CPU tensor max={max_val_cpu:.12e}\n")
                                    f.write(f"GPU tensor max={max_val_gpu:.12e}\n")
                                    f.write(f"GPU tensor shape={tensor_gpu.shape}\n")
                                    f.write(f"GPU tensor dtype={tensor_gpu.dtype}\n")
                                    f.write(f"GPU tensor device={tensor_gpu.device}\n")
                                    f.write(f"GPU tensor is_contiguous={tensor_gpu.is_contiguous()}\n")
                                    # Check GPU tensor statistics
                                    gpu_min = tensor_gpu.min().item()
                                    gpu_mean = tensor_gpu.mean().item()
                                    gpu_std = tensor_gpu.std().item()
                                    gpu_zero_count = (tensor_gpu == 0).sum().item()
                                    f.write(f"GPU tensor min={gpu_min:.12e}, mean={gpu_mean:.12e}, std={gpu_std:.12e}\n")
                                    f.write(f"GPU tensor zero_count={gpu_zero_count}/{tensor_gpu.numel()} ({100*gpu_zero_count/tensor_gpu.numel():.2f}%)\n")
                                    f.write(f"{'!'*80}\n\n")
                                
                                if step < 3:
                                    logger.warning(f"Step {step}: Method 1 failed - GPU tensor is zeros")
                        except Exception as e:
                            if step < 3:
                                logger.warning(f"Step {step}: Method 1 (pinned memory) exception for {name}: {e}")
                                with open("debug_batch.txt", "a") as f:
                                    f.write(f"Step {step}: Method 1 EXCEPTION for {name}: {e}\n")
                        
                        # Method 2: Try direct .to() with non_blocking=False on cloned tensor
                        try:
                            if step < 3:
                                logger.warning(f"Step {step}: Trying Method 2 (direct .to()) for {name}...")
                            tensor_gpu = value_clone.to(accelerator.device, non_blocking=False)
                            max_val_gpu = tensor_gpu.abs().max().item()
                            
                            if step < 3:
                                logger.warning(f"Step {step}: Method 2 result for {name}: CPU max={max_val_cpu:.6e}, Clone max={max_val_clone:.6e}, GPU max={max_val_gpu:.6e}")
                            if max_val_gpu > 1e-6:
                                if step < 3:
                                    logger.info(f"Step {step}: Method 2 (direct .to()) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                                return tensor_gpu
                            else:
                                if step < 3:
                                    logger.warning(f"Step {step}: Method 2 failed - GPU tensor is zeros")
                        except Exception as e:
                            if step < 3:
                                logger.warning(f"Step {step}: Method 2 (direct .to()) exception for {name}: {e}")
                        
                        # Method 3: Try copy_() with empty tensor
                        try:
                            tensor_gpu = torch.empty(value_clone.shape, dtype=value_clone.dtype, device=accelerator.device)
                            tensor_gpu.copy_(value_clone, non_blocking=False)
                            torch.cuda.synchronize(accelerator.device)
                            max_val_gpu = tensor_gpu.abs().max().item()
                            
                            if max_val_gpu > 1e-6:
                                if step < 3:
                                    logger.info(f"Step {step}: Method 3 (copy_()) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                                return tensor_gpu
                        except Exception as e:
                            logger.debug(f"Step {step}: Method 3 (copy_()) failed for {name}: {e}")
                        
                        # Method 4: Try chunked transfer (for large tensors)
                        try:
                            # Split into chunks and transfer separately
                            chunk_size = value_clone.numel() // 4  # 4 chunks
                            if chunk_size > 0:
                                tensor_gpu = torch.empty(value_clone.shape, dtype=value_clone.dtype, device=accelerator.device)
                                flat_cpu = value_clone.flatten()
                                flat_gpu = tensor_gpu.flatten()
                                
                                for i in range(0, flat_cpu.numel(), chunk_size):
                                    end_idx = min(i + chunk_size, flat_cpu.numel())
                                    flat_gpu[i:end_idx].copy_(flat_cpu[i:end_idx], non_blocking=False)
                                
                                torch.cuda.synchronize(accelerator.device)
                                max_val_gpu = tensor_gpu.abs().max().item()
                                
                                if max_val_gpu > 1e-6:
                                    if step < 3:
                                        logger.info(f"Step {step}: Method 4 (chunked transfer) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                                    return tensor_gpu
                        except Exception as e:
                            logger.debug(f"Step {step}: Method 4 (chunked transfer) failed for {name}: {e}")
                        
                        # Method 5: Try using torch.tensor constructor on GPU
                        try:
                            if step < 3:
                                logger.warning(f"Step {step}: Trying Method 5 (tensor constructor) for {name}...")
                            # Convert to numpy, then create tensor directly on GPU
                            numpy_array = value_clone.cpu().numpy()
                            tensor_gpu = torch.tensor(numpy_array, dtype=value_clone.dtype, device=accelerator.device)
                            torch.cuda.synchronize(accelerator.device)
                            max_val_gpu = tensor_gpu.abs().max().item()
                            
                            if step < 3:
                                logger.warning(f"Step {step}: Method 5 result for {name}: CPU max={max_val_cpu:.6e}, Clone max={max_val_clone:.6e}, GPU max={max_val_gpu:.6e}")
                            if max_val_gpu > 1e-6:
                                if step < 3:
                                    logger.info(f"Step {step}: Method 5 (tensor constructor) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                                return tensor_gpu
                            else:
                                if step < 3:
                                    logger.warning(f"Step {step}: Method 5 failed - GPU tensor is zeros")
                        except Exception as e:
                            if step < 3:
                                logger.warning(f"Step {step}: Method 5 (tensor constructor) exception for {name}: {e}")
                        
                        # Method 6: Try using CUDA stream for async transfer
                        # DISABLED: CUDA streams hang on ROCm Windows (hipStreamCreateWithPriority hangs)
                        # Skip this method entirely - it's not critical and causes hangs
                        # if step < 3:
                        #     try:
                        #         logger.warning(f"Step {step}: Trying Method 6 (CUDA stream) for {name}...")
                        #         stream = torch.cuda.Stream(device=accelerator.device)
                        #         with torch.cuda.stream(stream):
                        #             tensor_gpu = value_clone.to(accelerator.device, non_blocking=True)
                        #         stream.synchronize()
                        #         max_val_gpu = tensor_gpu.abs().max().item()
                        #         
                        #         logger.warning(f"Step {step}: Method 6 result for {name}: CPU max={max_val_cpu:.6e}, Clone max={max_val_clone:.6e}, GPU max={max_val_gpu:.6e}")
                        #         if max_val_gpu > 1e-6:
                        #             logger.info(f"Step {step}: Method 6 (CUDA stream) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                        #             return tensor_gpu
                        #         else:
                        #             logger.warning(f"Step {step}: Method 6 failed - GPU tensor is zeros")
                        #     except Exception as e:
                        #         logger.warning(f"Step {step}: Method 6 (CUDA stream) exception for {name}: {e}")
                        if step < 3:
                            logger.debug(f"Step {step}: Skipping Method 6 (CUDA stream) - known to hang on ROCm Windows")
                        
                        # Method 7: CRITICAL ROCm WORKAROUND - Direct GPU allocation + element-wise copy
                        # This bypasses ROCm's broken transfer mechanisms entirely
                        try:
                            if step < 3:
                                logger.warning(f"Step {step}: Trying Method 7 (direct GPU allocation + element copy) for {name}...")
                            
                            # Clear GPU cache first to ensure clean allocation
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize(accelerator.device)
                            
                            # Allocate tensor directly on GPU
                            tensor_gpu = torch.empty(value_clone.shape, dtype=value_clone.dtype, device=accelerator.device)
                            
                            # Copy data element by element using CPU tensor's data pointer
                            # This bypasses ROCm's transfer mechanisms
                            cpu_flat = value_clone.flatten().cpu()
                            gpu_flat = tensor_gpu.flatten()
                            
                            # Use small chunks to avoid memory issues
                            chunk_size = min(1024 * 1024, cpu_flat.numel())  # 1MB chunks or smaller
                            for i in range(0, cpu_flat.numel(), chunk_size):
                                end_idx = min(i + chunk_size, cpu_flat.numel())
                                # Create a new tensor from the CPU data and transfer
                                chunk_cpu = cpu_flat[i:end_idx].clone()
                                gpu_flat[i:end_idx] = chunk_cpu.to(accelerator.device, non_blocking=False)
                            
                            torch.cuda.synchronize(accelerator.device)
                            max_val_gpu = tensor_gpu.abs().max().item()
                            
                            if step < 3:
                                logger.warning(f"Step {step}: Method 7 result for {name}: CPU max={max_val_cpu:.6e}, Clone max={max_val_clone:.6e}, GPU max={max_val_gpu:.6e}")
                            if max_val_gpu > 1e-6:
                                if step < 3:
                                    logger.info(f"Step {step}: Method 7 (direct GPU allocation + element copy) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                                return tensor_gpu
                            else:
                                # Log detailed failure
                                with open("debug_batch.txt", "a") as f:
                                    f.write(f"Step {step}: Method 7 FAILED for {name}\n")
                                    f.write(f"  CPU max={max_val_cpu:.12e}, GPU max={max_val_gpu:.12e}\n")
                                    f.write(f"  Method 7 uses chunked .to(device) transfers - all chunks failed\n")
                                if step < 3:
                                    logger.warning(f"Step {step}: Method 7 failed - GPU tensor is zeros")
                        except Exception as e:
                            with open("debug_batch.txt", "a") as f:
                                f.write(f"Step {step}: Method 7 EXCEPTION for {name}: {e}\n")
                            if step < 3:
                                logger.warning(f"Step {step}: Method 7 (direct GPU allocation) exception for {name}: {e}")
                        
                        # Method 8: Last resort - Use torch.from_numpy with direct GPU placement
                        try:
                            if step < 3:
                                logger.warning(f"Step {step}: Trying Method 8 (numpy + direct GPU) for {name}...")
                            
                            # Convert to numpy
                            numpy_array = value_clone.detach().cpu().numpy()
                            
                            # Clear GPU cache
                            torch.cuda.empty_cache()
                            
                            # Create tensor directly on GPU from numpy
                            # Use torch.as_tensor to avoid copying if possible, but force GPU
                            tensor_gpu = torch.as_tensor(numpy_array, device=accelerator.device)
                            
                            # If that didn't work, try explicit creation
                            if tensor_gpu.device != accelerator.device:
                                tensor_gpu = torch.tensor(numpy_array, dtype=value_clone.dtype, device=accelerator.device)
                            
                            torch.cuda.synchronize(accelerator.device)
                            max_val_gpu = tensor_gpu.abs().max().item()
                            
                            if step < 3:
                                logger.warning(f"Step {step}: Method 8 result for {name}: CPU max={max_val_cpu:.6e}, Clone max={max_val_clone:.6e}, GPU max={max_val_gpu:.6e}")
                            if max_val_gpu > 1e-6:
                                if step < 3:
                                    logger.info(f"Step {step}: Method 8 (numpy + direct GPU) worked for {name}: CPU max={max_val_cpu:.6e}, GPU max={max_val_gpu:.6e}")
                                return tensor_gpu
                            else:
                                # Log detailed failure
                                with open("debug_batch.txt", "a") as f:
                                    f.write(f"Step {step}: Method 8 FAILED for {name}\n")
                                    f.write(f"  CPU max={max_val_cpu:.12e}, GPU max={max_val_gpu:.12e}\n")
                                    f.write(f"  Method 8 uses numpy intermediate - also failed\n")
                                if step < 3:
                                    logger.warning(f"Step {step}: Method 8 failed - GPU tensor is zeros")
                        except Exception as e:
                            with open("debug_batch.txt", "a") as f:
                                f.write(f"Step {step}: Method 8 EXCEPTION for {name}: {e}\n")
                            if step < 3:
                                logger.warning(f"Step {step}: Method 8 (numpy + direct GPU) exception for {name}: {e}")
                        
                        # All methods failed - this is a critical ROCm bug
                        # Log comprehensive failure diagnostics
                        log_file = "debug_batch.txt"
                        with open(log_file, "a") as f:
                            f.write(f"\n{'#'*80}\n")
                            f.write(f"CRITICAL FAILURE: All transfer methods failed for {name}\n")
                            f.write(f"{'#'*80}\n")
                            f.write(f"Step: {step}\n")
                            f.write(f"Tensor name: {name}\n")
                            f.write(f"CPU tensor max: {max_val_cpu:.12e}\n")
                            f.write(f"Clone tensor max: {max_val_clone:.12e}\n")
                            f.write(f"Original tensor shape: {value.shape}\n")
                            f.write(f"Original tensor dtype: {value.dtype}\n")
                            f.write(f"Original tensor contiguous: {value.is_contiguous()}\n")
                            f.write(f"Clone tensor shape: {value_clone.shape}\n")
                            f.write(f"Clone tensor dtype: {value_clone.dtype}\n")
                            f.write(f"Clone tensor contiguous: {value_clone.is_contiguous()}\n")
                            
                            # Final tensor state analysis
                            f.write(f"\nFINAL TENSOR ANALYSIS:\n")
                            f.write(f"  Original CPU tensor:\n")
                            f.write(f"    min={value.min().item():.12e}, max={value.max().item():.12e}\n")
                            f.write(f"    mean={value.mean().item():.12e}, std={value.std().item():.12e}\n")
                            f.write(f"    zero_count={(value == 0).sum().item()}/{value.numel()}\n")
                            f.write(f"  Cloned CPU tensor:\n")
                            f.write(f"    min={value_clone.min().item():.12e}, max={value_clone.max().item():.12e}\n")
                            f.write(f"    mean={value_clone.mean().item():.12e}, std={value_clone.std().item():.12e}\n")
                            f.write(f"    zero_count={(value_clone == 0).sum().item()}/{value_clone.numel()}\n")
                            
                            # GPU memory state at failure
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                                reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**3
                                max_allocated = torch.cuda.max_memory_allocated(accelerator.device) / 1024**3
                                f.write(f"\nGPU MEMORY AT FAILURE:\n")
                                f.write(f"  allocated={allocated:.4f} GB\n")
                                f.write(f"  reserved={reserved:.4f} GB\n")
                                f.write(f"  max_allocated={max_allocated:.4f} GB\n")
                            
                            # Diagnostic tests
                            f.write(f"\nDIAGNOSTIC TESTS:\n")
                            try:
                                # Test 1: Can GPU create tensors?
                                test_tensor = torch.randn(100, device=accelerator.device)
                                test_max = test_tensor.abs().max().item()
                                if test_max > 1e-6:
                                    f.write(f"  GPU tensor creation: SUCCESS (test max={test_max:.12e})\n")
                                else:
                                    f.write(f"  GPU tensor creation: FAILED - GPU cannot create valid tensors!\n")
                                    f.write(f"    Test tensor max={test_max:.12e} (should be > 0)\n")
                                    f.write(f"    CRITICAL: GPU is in corrupted state - cannot create random tensors\n")
                                
                                # Test 2: Can simple CPU->GPU transfer work?
                                test_cpu = torch.randn(100, device='cpu')
                                test_cpu_max = test_cpu.abs().max().item()
                                test_gpu = test_cpu.to(accelerator.device, non_blocking=False)
                                test_gpu_max = test_gpu.abs().max().item()
                                
                                if test_gpu_max > 1e-6:
                                    f.write(f"  Simple CPU->GPU transfer: SUCCESS\n")
                                    f.write(f"    CPU max={test_cpu_max:.12e}, GPU max={test_gpu_max:.12e}\n")
                                    f.write(f"  CRITICAL: Simple transfers work, but this specific tensor fails!\n")
                                    f.write(f"  This suggests tensor-specific issue:\n")
                                    f.write(f"    - Size: {value_clone.numel()} vs {test_cpu.numel()}\n")
                                    f.write(f"    - Shape: {value_clone.shape} vs {test_cpu.shape}\n")
                                    f.write(f"    - Dtype: {value_clone.dtype} vs {test_cpu.dtype}\n")
                                    f.write(f"    - Memory layout differences\n")
                                else:
                                    f.write(f"  Simple CPU->GPU transfer: FAILED\n")
                                    f.write(f"    CPU max={test_cpu_max:.12e}, GPU max={test_gpu_max:.12e}\n")
                                    f.write(f"  CRITICAL: Even simple transfers fail - GPU is completely broken\n")
                                
                                # Test 3: Check if GPU can do any operations
                                if test_max > 1e-6:
                                    test_result = (test_tensor * 2.0).sum().item()
                                    f.write(f"  GPU operations: SUCCESS (test result={test_result:.12e})\n")
                                else:
                                    f.write(f"  GPU operations: Cannot test (GPU tensor creation failed)\n")
                                    
                            except Exception as e:
                                f.write(f"  Diagnostic tests failed: {str(e)}\n")
                                import traceback
                                try:
                                    f.write(f"  Traceback: {traceback.format_exc()}\n")
                                except:
                                    f.write(f"  Could not write traceback (encoding issue)\n")
                            
                            # Environment and system info
                            f.write(f"\nSYSTEM INFORMATION:\n")
                            f.write(f"  HIP_DISABLE_IPC={os.environ.get('HIP_DISABLE_IPC', 'not set')}\n")
                            f.write(f"  HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'not set')}\n")
                            f.write(f"  HIP_LAUNCH_BLOCKING={os.environ.get('HIP_LAUNCH_BLOCKING', 'not set')}\n")
                            f.write(f"  AMD_LOG_LEVEL={os.environ.get('AMD_LOG_LEVEL', 'not set')}\n")
                            f.write(f"  PyTorch version: {torch.__version__}\n")
                            if hasattr(torch.version, 'hip'):
                                f.write(f"  ROCm version: {torch.version.hip}\n")
                            if torch.cuda.is_available():
                                f.write(f"  GPU: {torch.cuda.get_device_name(0)}\n")
                                props = torch.cuda.get_device_properties(0)
                                f.write(f"  GPU total memory: {props.total_memory / 1024**3:.2f} GB\n")
                            
                            f.write(f"\n{'#'*80}\n\n")
                        
                        # Also log to console
                        logger.error(f"CRITICAL Step {step}: All transfer methods failed for {name}!")
                        logger.error(f"  CPU max={max_val_cpu:.6e}, Clone max={max_val_clone:.6e}")
                        logger.error(f"  Detailed diagnostics written to debug_batch.txt")
                        logger.error(f"  This indicates a severe ROCm bug on gfx1151 where CPU→GPU transfer produces zeros.")
                        
                        raise RuntimeError(f"ROCm bug: Cannot move {name} tensor from CPU to GPU - all values become zeros. This is a critical ROCm bug on gfx1151. All {8} transfer methods failed. See debug_batch.txt for detailed diagnostics.")
                    else:
                        return value
                
                # Move all tensors in batch to GPU
                # ROCm WORKAROUND: Use simple transfers when using PEFT (PEFT avoids the ROCm bug)
                if args.use_peft:
                    # PEFT works correctly with simple transfers - no workarounds needed
                    if batch["latents"].device.type != "cuda":
                        latents = batch["latents"].to(accelerator.device, non_blocking=False)
                    else:
                        latents = batch["latents"]
                    # Verify transfer worked (PEFT should avoid the bug, but check anyway)
                    if latents.abs().max().item() < 1e-6 and batch["latents"].abs().max().item() > 1e-6:
                        logger.error(f"Step {step}: CRITICAL - PEFT transfer produced zeros! This should not happen.")
                        raise RuntimeError("PEFT tensor transfer failed - this indicates a deeper ROCm issue")
                else:
                    # Use workarounds for non-PEFT path (affected by ROCm bug)
                    if batch["latents"].device.type == "cuda":
                        logger.debug(f"Step {step}: latents already on GPU (loaded directly from cache), skipping transfer")
                        latents = batch["latents"]
                    else:
                        latents = move_to_gpu(batch["latents"], "latents")
                
                # Also move other tensors that might be needed
                if args.use_peft:
                    # Simple transfers for PEFT (avoids ROCm bug)
                    if "latents_image" in batch and batch["latents_image"].device.type != "cuda":
                        batch["latents_image"] = batch["latents_image"].to(accelerator.device, non_blocking=False)
                    if "t5" in batch and isinstance(batch["t5"], torch.Tensor) and batch["t5"].device.type != "cuda":
                        batch["t5"] = batch["t5"].to(accelerator.device, non_blocking=False)
                    if "timesteps" in batch and batch["timesteps"] is not None:
                        if isinstance(batch["timesteps"], torch.Tensor) and batch["timesteps"].device.type != "cuda":
                            batch["timesteps"] = batch["timesteps"].to(accelerator.device, non_blocking=False)
                else:
                    # Use workarounds for non-PEFT path
                    if "latents_image" in batch:
                        if batch["latents_image"].device.type == "cuda":
                            logger.debug(f"Step {step}: latents_image already on GPU, skipping transfer")
                        else:
                            batch["latents_image"] = move_to_gpu(batch["latents_image"], "latents_image")
                    if "t5" in batch:
                        if isinstance(batch["t5"], torch.Tensor) and batch["t5"].device.type == "cuda":
                            logger.debug(f"Step {step}: t5 already on GPU, skipping transfer")
                        else:
                            batch["t5"] = move_to_gpu(batch["t5"], "t5")
                    if "timesteps" in batch and batch["timesteps"] is not None:
                        if isinstance(batch["timesteps"], torch.Tensor) and batch["timesteps"].device.type == "cuda":
                            logger.debug(f"Step {step}: timesteps already on GPU, skipping transfer")
                        else:
                            batch["timesteps"] = move_to_gpu(batch["timesteps"], "timesteps")
                
                # ROCm DIAGNOSTIC: Test GPU random number generator on first step
                if step == 0:
                    logger.info("ROCm DIAGNOSTIC: Testing GPU random number generator...")
                    try:
                        # Test 1: Direct torch.randn on GPU
                        test_randn = torch.randn(100, device=accelerator.device)
                        test_randn_max = test_randn.abs().max().item()
                        logger.info(f"  Test 1 - torch.randn(100, device='cuda'): max={test_randn_max:.6e}")
                        if test_randn_max < 1e-6:
                            logger.error("  CRITICAL: torch.randn produces zeros! GPU random number generator is broken.")
                        
                        # Test 2: torch.randn_like on a non-zero tensor
                        test_ones = torch.ones(100, device=accelerator.device)
                        test_randn_like = torch.randn_like(test_ones)
                        test_randn_like_max = test_randn_like.abs().max().item()
                        logger.info(f"  Test 2 - torch.randn_like(torch.ones(100)): max={test_randn_like_max:.6e}")
                        if test_randn_like_max < 1e-6:
                            logger.error("  CRITICAL: torch.randn_like produces zeros! GPU random number generator is broken.")
                        
                        # Test 3: Simple GPU operation
                        test_add = test_ones + 1.0
                        test_add_sum = test_add.sum().item()
                        logger.info(f"  Test 3 - Simple GPU operation (ones + 1.0): sum={test_add_sum:.6e} (expected=200.0)")
                        if abs(test_add_sum - 200.0) > 1e-3:
                            logger.error(f"  CRITICAL: GPU operations produce wrong results! Expected 200.0, got {test_add_sum:.6e}")
                        
                    except Exception as e:
                        logger.error(f"  ERROR during GPU diagnostic tests: {e}")
                
                # Write latents info to file immediately
                if step < 3:
                    with open("debug_batch.txt", "a") as f:
                        max_val = latents.abs().max().item()
                        f.write(f"Step {step}: After move_to_gpu - latents state:\n")
                        f.write(f"  shape={latents.shape}, dtype={latents.dtype}, device={latents.device}\n")
                        f.write(f"  is_contiguous={latents.is_contiguous()}, max={max_val:.6e}\n")
                        f.write(f"  min={latents.min().item():.6e}, mean={latents.mean().item():.6e}\n")
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                            reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**3
                            f.write(f"  GPU memory after transfer: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB\n")
                        if max_val < 1e-6:
                            f.write(f"  *** CRITICAL: Latents are all zeros after transfer! ***\n")
                
                # Debug: Check if latents are all zeros (corrupted cache?)
                # Log first 10 steps and every 50 steps to catch issues
                if step < 10 or step % 50 == 0:
                    # Check device and dtype
                    latents_device = latents.device
                    latents_dtype = latents.dtype
                    latents_stats = f"shape={latents.shape}, dtype={latents_dtype}, device={latents_device}, min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}"
                    logger.info(f"DEBUG Step {step}: Raw latents from batch - {latents_stats}")
                    
                    # CRITICAL: Check if latents are all zeros
                    max_val = latents.abs().max().item()
                    if max_val < 1e-6:
                        logger.error(f"CRITICAL Step {step}: Latents are all zeros! This means cached latents are corrupted or VAE encoding failed.")
                        logger.error(f"  Latents device: {latents_device}, dtype: {latents_dtype}")
                        logger.error(f"  This is a ROCm bug where tensors become zeros when Accelerator moves them to GPU.")
                        # Try to create a test tensor to see if ROCm randn works
                        test_tensor = torch.randn_like(latents)
                        test_max = test_tensor.abs().max().item()
                        if test_max < 1e-6:
                            logger.error(f"  CRITICAL: torch.randn_like also produces zeros! This confirms a severe ROCm bug.")
                        else:
                            logger.info(f"  torch.randn_like works (max={test_max:.6e}), so the issue is with the latents tensor itself.")
                    
                    # Check for Inf/NaN values
                    if torch.isinf(latents).any() or torch.isnan(latents).any():
                        logger.error(f"CRITICAL Step {step}: Latents contain Inf or NaN values! This will cause training issues.")

                with accelerator.accumulate(training_model):
                    accelerator.unwrap_model(network).on_step_start()

                    latents = self.scale_shift_latents(latents)
                    
                    # Debug: Check latents after scaling
                    if step < 10 or step % 50 == 0:
                        latents_stats = f"shape={latents.shape}, dtype={latents.dtype}, min={latents.min().item():.6f}, max={latents.max().item():.6f}, mean={latents.mean().item():.6f}"
                        logger.info(f"DEBUG Step {step}: Latents after scale_shift - {latents_stats}")
                        
                        # Detailed memory and device state
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                            reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**3
                            logger.info(f"DEBUG Step {step}: GPU memory - allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
                        
                        if latents.abs().max().item() < 1e-6:
                            logger.error(f"CRITICAL Step {step}: Latents are all zeros after scale_shift!")
                            # Check if tensor is contiguous
                            logger.error(f"  Latents is_contiguous: {latents.is_contiguous()}")
                            logger.error(f"  Latents device: {latents.device}, dtype: {latents.dtype}")
                            # Try to check if GPU state is corrupted
                            test_tensor = torch.randn(10, 10, device=accelerator.device)
                            test_max = test_tensor.abs().max().item()
                            logger.error(f"  GPU test tensor max: {test_max:.6e} (should be > 0)")

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    
                    # Debug: Check noise
                    if step < 10 or step % 50 == 0:
                        noise_stats = f"shape={noise.shape}, dtype={noise.dtype}, min={noise.min().item():.6f}, max={noise.max().item():.6f}, mean={noise.mean().item():.6f}, std={noise.std().item():.6f}"
                        logger.info(f"DEBUG Step {step}: Generated noise - {noise_stats}")
                        if noise.abs().max().item() < 1e-6:
                            logger.error(f"CRITICAL Step {step}: Noise is all zeros! This should never happen with torch.randn_like.")

                    # calculate model input and timesteps
                    noisy_model_input, timesteps = self.get_noisy_model_input_and_timesteps(
                        args, noise, latents, batch["timesteps"], noise_scheduler, accelerator.device, dit_dtype
                    )
                    
                    # Debug: Check noisy_model_input and timesteps
                    if step < 10 or step % 50 == 0:
                        noisy_stats = f"shape={noisy_model_input.shape}, dtype={noisy_model_input.dtype}, min={noisy_model_input.min().item():.6f}, max={noisy_model_input.max().item():.6f}, mean={noisy_model_input.mean().item():.6f}"
                        logger.info(f"DEBUG Step {step}: noisy_model_input - {noisy_stats}")
                        logger.info(f"DEBUG Step {step}: timesteps - shape={timesteps.shape}, dtype={timesteps.dtype}, min={timesteps.min().item():.0f}, max={timesteps.max().item():.0f}")
                        if noisy_model_input.abs().max().item() < 1e-6:
                            logger.error(f"CRITICAL Step {step}: noisy_model_input is all zeros!")

                    weighting = compute_loss_weighting_for_sd3(
                        args.weighting_scheme, noise_scheduler, timesteps, accelerator.device, dit_dtype
                    )
                    
                    # Debug: Check weighting
                    if step < 10 or step % 50 == 0:
                        if weighting is not None:
                            weight_stats = f"shape={weighting.shape}, dtype={weighting.dtype}, min={weighting.min().item():.6f}, max={weighting.max().item():.6f}, mean={weighting.mean().item():.6f}"
                            logger.info(f"DEBUG Step {step}: weighting - {weight_stats}")

                    # CRITICAL: Log before model forward pass
                    if step < 10 or step % 50 == 0:
                        logger.info(f"DEBUG Step {step}: About to call model forward pass (call_dit)")
                        logger.info(f"  latents: shape={latents.shape}, dtype={latents.dtype}, device={latents.device}, max={latents.abs().max().item():.6e}")
                        logger.info(f"  noise: shape={noise.shape}, dtype={noise.dtype}, device={noise.device}, max={noise.abs().max().item():.6e}")
                        logger.info(f"  noisy_model_input: shape={noisy_model_input.shape}, dtype={noisy_model_input.dtype}, device={noisy_model_input.device}, max={noisy_model_input.abs().max().item():.6e}")
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                            reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**3
                            logger.info(f"  GPU memory before forward: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
                    
                    model_pred, target = self.call_dit(
                        args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype
                    )
                    
                    # CRITICAL: Log after model forward pass
                    if step < 10 or step % 50 == 0:
                        logger.info(f"DEBUG Step {step}: After model forward pass (call_dit)")
                        logger.info(f"  model_pred: shape={model_pred.shape}, dtype={model_pred.dtype}, device={model_pred.device}, min={model_pred.min().item():.6f}, max={model_pred.max().item():.6f}, mean={model_pred.mean().item():.6f}")
                        logger.info(f"  target: shape={target.shape}, dtype={target.dtype}, device={target.device}, min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}")
                        if model_pred.abs().max().item() < 1e-6:
                            logger.error(f"CRITICAL Step {step}: model_pred is all zeros after forward pass!")
                        if target.abs().max().item() < 1e-6:
                            logger.error(f"CRITICAL Step {step}: target is all zeros after forward pass!")
                        if torch.cuda.is_available():
                            allocated = torch.cuda.memory_allocated(accelerator.device) / 1024**3
                            reserved = torch.cuda.memory_reserved(accelerator.device) / 1024**3
                            logger.info(f"  GPU memory after forward: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")
                    loss = torch.nn.functional.mse_loss(model_pred.to(network_dtype), target, reduction="none")

                    if weighting is not None:
                        loss = loss * weighting
                    # loss = loss.mean([1, 2, 3])
                    # # min snr gamma, scale v pred loss like noise pred, v pred like loss, debiased estimation etc.
                    # loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

                    loss = loss.mean()  # mean loss over all elements in batch
                    
                    # Debug: Log raw loss value to diagnose zero loss issue
                    raw_loss_value = loss.detach().item()
                    # Always log first 20 steps, then every 50 steps
                    if step < 20 or step % 50 == 0:
                        logger.info(f"Step {step}: Raw loss={raw_loss_value:.6f}, model_pred stats: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}")
                        logger.info(f"Step {step}: Target stats: min={target.min().item():.4f}, max={target.max().item():.4f}, mean={target.mean().item():.4f}")
                    
                    # CRITICAL: Check for NaN/Inf in loss before backward pass
                    # If loss is NaN, skip this step to prevent crash
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.error(f"Step {step}: CRITICAL - Loss is NaN/Inf ({raw_loss_value}). Skipping optimizer step to prevent crash.")
                        logger.error(f"  This is likely due to fp16 numerical instability on ROCm. Consider using bf16 or disabling mixed precision.")
                        # Zero gradients and skip optimizer step
                        optimizer.zero_grad(set_to_none=True)
                        continue  # Skip to next iteration
                    
                    # Warn if loss is suspiciously low or zero
                    if raw_loss_value == 0.0:
                        logger.warning(f"Step {step}: WARNING - Loss is exactly 0.0! This is suspicious. model_pred and target may be identical.")
                    elif raw_loss_value < 1e-6:
                        logger.warning(f"Step {step}: WARNING - Loss is very small ({raw_loss_value:.9f}). This may indicate an issue.")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                        state = accelerate.PartialState()
                        if state.distributed_type != accelerate.DistributedType.NO:
                            for param in network.parameters():
                                if param.grad is not None:
                                    param.grad = accelerator.reduce(param.grad, reduction="mean")

                        if args.max_grad_norm != 0.0:
                            params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                            # For fp16 mixed precision, use direct clipping instead of accelerator's scaler
                            # Accelerator's clip_grad_norm_ tries to unscale fp16 gradients which isn't supported
                            if args.mixed_precision == "fp16":
                                torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                            else:
                                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    # Handle bitsandbytes ROCm binary error during step()
                    try:
                        optimizer.step()
                    except RuntimeError as e:
                        error_msg = str(e)
                        if any(keyword in error_msg for keyword in ["ROCm binary", "libbitsandbytes_rocm", "Forgot to compile", "not found at", "bitsandbytes library load error"]):
                            logger.error(
                                f"Step {step}: bitsandbytes ROCm binary error during optimizer.step(): {error_msg[:300]}\n"
                                f"Please restart training with --optimizer_type AdamW (without 8bit) instead of AdamW8bit.\n"
                                f"To use 8-bit optimizer, compile bitsandbytes from source for ROCm."
                            )
                            raise RuntimeError(
                                "bitsandbytes ROCm binary not available. "
                                "Please restart with --optimizer_type AdamW instead of AdamW8bit."
                            ) from e
                        else:
                            # Some other RuntimeError, re-raise it
                            raise
                    
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    if global_step == 0:
                        progress_bar.reset()  # exclude first step from progress bar, because it may take long due to initializations
                    progress_bar.update(1)
                    global_step += 1

                    # to avoid calling optimizer_eval_fn() too frequently, we call it only when we need to sample images or save the model
                    should_sampling = should_sample_images(args, global_step, epoch=None)
                    should_saving = args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0

                    if should_sampling or should_saving:
                        optimizer_eval_fn()
                        if should_sampling:
                            self.sample_images(accelerator, args, None, global_step, vae, transformer, sample_parameters, dit_dtype)

                        if should_saving:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                ckpt_name = train_utils.get_step_ckpt_name(args.output_name, global_step)
                                save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                                if args.save_state:
                                    train_utils.save_and_remove_state_stepwise(args, accelerator, global_step)

                                remove_step_no = train_utils.get_remove_step_no(args, global_step)
                                if remove_step_no is not None:
                                    remove_ckpt_name = train_utils.get_step_ckpt_name(args.output_name, remove_step_no)
                                    remove_model(remove_ckpt_name)
                        optimizer_train_fn()

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                
                # Debug: Log loss values to diagnose zero loss issue
                # Always log first 20 steps, then every 50 steps
                if global_step < 20 or global_step % 50 == 0:
                    logger.info(f"Step {global_step}: current_loss={current_loss:.6f}, avr_loss={avr_loss:.6f}, valid_count={loss_recorder.valid_count}, loss_total={loss_recorder.loss_total:.6f}")
                
                # Warn if average loss is zero or suspicious
                if avr_loss == 0.0 and loss_recorder.valid_count > 0:
                    logger.warning(f"Step {global_step}: WARNING - Average loss is 0.0 but valid_count={loss_recorder.valid_count}. All losses may be exactly 0.0!")
                elif math.isnan(avr_loss) and loss_recorder.valid_count == 0:
                    logger.warning(f"Step {global_step}: WARNING - No valid losses recorded (all may be NaN/Inf). valid_count=0, loss_total={loss_recorder.loss_total}")
                
                # Debug: Log if loss is NaN/Inf or if average is NaN
                if math.isnan(current_loss) or math.isinf(current_loss):
                    logger.warning(f"Step {step}: Loss is {'NaN' if math.isnan(current_loss) else 'Inf'}: {current_loss}")
                if math.isnan(avr_loss):
                    logger.warning(f"Step {step}: Average loss is NaN (valid_count={loss_recorder.valid_count}, loss_total={loss_recorder.loss_total})")
                
                logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if len(accelerator.trackers) > 0:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions, optimizer, keys_scaled, mean_norm, maximum_norm
                    )
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if len(accelerator.trackers) > 0:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # save model at the end of epoch if needed
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_utils.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_utils.get_epoch_ckpt_name(args.output_name, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_utils.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, vae, transformer, sample_parameters, dit_dtype)
            optimizer_train_fn()

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if is_main_process:
            network = accelerator.unwrap_model(network)

        # Set default output_dir if not specified
        if args.output_dir is None:
            args.output_dir = "."

        # Save model BEFORE end_training to avoid ROCm compatibility issues
        if is_main_process:
            ckpt_name = train_utils.get_last_ckpt_name(args.output_name)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            logger.info("model saved.")

        # Try to clean up, but don't fail if there's a ROCm compatibility issue
        try:
            accelerator.end_training()
        except AttributeError as e:
            if "is_initialized" in str(e) or "torch.distributed" in str(e):
                logger.warning(f"ROCm compatibility issue during cleanup (model was saved): {e}")
            else:
                raise
        optimizer_eval_fn()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            try:
                train_utils.save_state_on_train_end(args, accelerator)
            except AttributeError as e:
                if "is_initialized" in str(e) or "torch.distributed" in str(e):
                    logger.warning(f"ROCm compatibility issue during state save: {e}")
                else:
                    raise


def setup_parser_common() -> argparse.ArgumentParser:
    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(f"Value '{value}' is not a valid percentage")
        try:
            float_value = float(value)
            if float_value >= 1 and float_value.is_integer():
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="using .toml instead of args to pass hyperparameter / ハイパーパラメータを引数ではなく.tomlファイルで渡す",
    )
    parser.add_argument(
        "--dataset_config",
        type=pathlib.Path,
        default=None,
        help="config file for dataset / データセットの設定ファイル",
    )

    # model settings
    parser.add_argument(
        "--sdpa",
        action="store_true",
        help="use sdpa for CrossAttention (requires PyTorch 2.0) / CrossAttentionにsdpaを使う（PyTorch 2.0が必要）",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="use FlashAttention for CrossAttention, requires FlashAttention / CrossAttentionにFlashAttentionを使う、FlashAttentionが必要",
    )
    parser.add_argument(
        "--sage_attn",
        action="store_true",
        help="use SageAttention. requires SageAttention / SageAttentionを使う。SageAttentionが必要",
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="use xformers for CrossAttention, requires xformers / CrossAttentionにxformersを使う、xformersが必要",
    )
    parser.add_argument(
        "--flash3",
        action="store_true",
        help="use FlashAttention 3 for CrossAttention, requires FlashAttention 3, HunyuanVideo does not support this yet"
        " / CrossAttentionにFlashAttention 3を使う、FlashAttention 3が必要。HunyuanVideoは未対応。",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="use split attention for attention calculation (split batch size=1, affects memory usage and speed)"
        " / attentionを分割して計算する（バッチサイズ=1に分割、メモリ使用量と速度に影響）",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (requires Triton) / torch.compileを有効にする（Tritonが必要）",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor) / torch.compileのバックエンド（デフォルト: inductor）",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",  # 学習用のデフォルト
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (default: default) / torch.compileのモード（デフォルト: default）",
    )
    parser.add_argument(
        "--compile_dynamic",
        type=str,
        default=None,
        choices=["true", "false", "auto"],
        help="Dynamic shapes mode for torch.compile (default: None, same as auto)"
        " / torch.compileの動的形状モード（デフォルト: None、autoと同じ動作）",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Enable fullgraph mode in torch.compile / torch.compileでフルグラフモードを有効にする",
    )
    parser.add_argument(
        "--compile_cache_size_limit",
        type=int,
        default=None,
        help="Set torch._dynamo.config.cache_size_limit (default: PyTorch default, typically 8-32) / torch._dynamo.config.cache_size_limitを設定（デフォルト: PyTorchのデフォルト、通常8-32）",
    )
    parser.add_argument(
        "--cuda_allow_tf32",
        action="store_true",
        help="Allow TF32 on Ampere or higher GPUs / Ampere以降のGPUでTF32を許可する",
    )
    parser.add_argument(
        "--cuda_cudnn_benchmark",
        action="store_true",
        help="Enable cudnn benchmark for possibly faster training / cudnnのベンチマークを有効にして学習の高速化を図る",
    )

    # training settings
    parser.add_argument("--max_train_steps", type=int, default=1600, help="training steps / 学習ステップ数")
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=None,
        help="training epochs (overrides max_train_steps) / 学習エポック数（max_train_stepsを上書きします）",
    )
    parser.add_argument(
        "--max_data_loader_n_workers",
        type=int,
        default=8,
        help="max num workers for DataLoader (lower is less main RAM usage, faster epoch start and slower data loading) / DataLoaderの最大プロセス数（小さい値ではメインメモリの使用量が減りエポック間の待ち時間が減りますが、データ読み込みは遅くなります）",
    )
    parser.add_argument(
        "--persistent_data_loader_workers",
        action="store_true",
        help="persistent DataLoader workers (useful for reduce time gap between epoch, but may use more memory) / DataLoader のワーカーを持続させる (エポック間の時間差を少なくするのに有効だが、より多くのメモリを消費する可能性がある)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed for training / 学習時の乱数のseed")
    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="enable gradient checkpointing / gradient checkpointingを有効にする"
    )
    parser.add_argument(
        "--gradient_checkpointing_cpu_offload",
        action="store_true",
        help="enable CPU offloading of activation for gradient checkpointing / gradient checkpointing時に活性化のCPUオフロードを有効にする",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass / 学習時に逆伝播をする前に勾配を合計するステップ数",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help="use mixed precision / 混合精度を使う場合、その精度",
    )

    parser.add_argument(
        "--logging_dir",
        type=str,
        default=None,
        help="enable logging and output TensorBoard log to this directory / ログ出力を有効にしてこのディレクトリにTensorBoard用のログを出力する",
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default=None,
        choices=["tensorboard", "wandb", "all"],
        help="what logging tool(s) to use (if 'all', TensorBoard and WandB are both used) / ログ出力に使用するツール (allを指定するとTensorBoardとWandBの両方が使用される)",
    )
    parser.add_argument(
        "--log_prefix", type=str, default=None, help="add prefix for each log directory / ログディレクトリ名の先頭に追加する文字列"
    )
    parser.add_argument(
        "--log_tracker_name",
        type=str,
        default=None,
        help="name of tracker to use for logging, default is script-specific default name / ログ出力に使用するtrackerの名前、省略時はスクリプトごとのデフォルト名",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="The name of the specific wandb session / wandb ログに表示される特定の実行の名前",
    )
    parser.add_argument(
        "--log_tracker_config",
        type=str,
        default=None,
        help="path to tracker config file to use for logging / ログ出力に使用するtrackerの設定ファイルのパス",
    )
    parser.add_argument(
        "--wandb_api_key",
        type=str,
        default=None,
        help="specify WandB API key to log in before starting training (optional). / WandB APIキーを指定して学習開始前にログインする（オプション）",
    )
    parser.add_argument("--log_config", action="store_true", help="log training configuration / 学習設定をログに出力する")

    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=None,
        help="DDP timeout (min, None for default of accelerate) / DDPのタイムアウト（分、Noneでaccelerateのデフォルト）",
    )
    parser.add_argument(
        "--ddp_gradient_as_bucket_view",
        action="store_true",
        help="enable gradient_as_bucket_view for DDP / DDPでgradient_as_bucket_viewを有効にする",
    )
    parser.add_argument(
        "--ddp_static_graph",
        action="store_true",
        help="enable static_graph for DDP / DDPでstatic_graphを有効にする",
    )

    parser.add_argument(
        "--sample_every_n_steps",
        type=int,
        default=None,
        help="generate sample images every N steps / 学習中のモデルで指定ステップごとにサンプル出力する",
    )
    parser.add_argument(
        "--sample_at_first", action="store_true", help="generate sample images before training / 学習前にサンプル出力する"
    )
    parser.add_argument(
        "--sample_every_n_epochs",
        type=int,
        default=None,
        help="generate sample images every N epochs (overwrites n_steps) / 学習中のモデルで指定エポックごとにサンプル出力する（ステップ数指定を上書きします）",
    )
    parser.add_argument(
        "--sample_prompts",
        type=str,
        default=None,
        help="file for prompts to generate sample images / 学習中モデルのサンプル出力用プロンプトのファイル",
    )

    # optimizer and lr scheduler settings
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, AdaFactor. "
        "Also, you can use any optimizer by specifying the full path to the class, like 'torch.optim.AdamW', 'bitsandbytes.optim.AdEMAMix8bit' or 'bitsandbytes.optim.PagedAdEMAMix8bit' etc. / ",
    )
    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )
    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない",
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor, rex",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps"
        " / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps"
        " / 学習率のスケジューラを減衰させるステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`"
        + " / 逆平方根スケジューラのタイムスケール、デフォルトは`num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler, warmup decay scheduler and rex scheduler"
        + " / 初期学習率の比率としての最小学習率を指定する、cosine with min lr スケジューラ、warmup decay スケジューラ、rex スケジューラ で有効",
    )
    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )

    parser.add_argument("--fp8_base", action="store_true", help="use fp8 for base model / base modelにfp8を使う")
    # parser.add_argument("--full_fp16", action="store_true", help="fp16 training including gradients / 勾配も含めてfp16で学習する")
    # parser.add_argument("--full_bf16", action="store_true", help="bf16 training including gradients / 勾配も含めてbf16で学習する")

    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="NO",
        choices=[e.value for e in DynamoBackend],
        help="dynamo backend type (default is None) / dynamoのbackendの種類（デフォルトは None）",
    )

    parser.add_argument(
        "--dynamo_mode",
        type=str,
        default=None,
        choices=["default", "reduce-overhead", "max-autotune"],
        help="dynamo mode (default is default) / dynamoのモード（デフォルトは default）",
    )

    parser.add_argument(
        "--dynamo_fullgraph",
        action="store_true",
        help="use fullgraph mode for dynamo / dynamoのfullgraphモードを使う",
    )

    parser.add_argument(
        "--dynamo_dynamic",
        action="store_true",
        help="use dynamic mode for dynamo / dynamoのdynamicモードを使う",
    )

    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="number of blocks to swap in the model, max XXX / モデル内のブロックの数、最大XXX",
    )
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swapping, which may speed up data transfer between CPU and GPU but uses more shared GPU memory on Windows"
        " / ブロックスワッピングにピン留めメモリを使用する。これによりCPUとGPU間のデータ転送が高速化される可能性があるが、Windowsではより多くの共有GPUメモリを使用する。",
    )
    parser.add_argument(
        "--img_in_txt_in_offloading",
        action="store_true",
        help="offload img_in and txt_in to cpu / img_inとtxt_inをCPUにオフロードする",
    )
    parser.add_argument(
        "--disable_numpy_memmap",
        action="store_true",
        help="Disable numpy memory mapping for model loading. Only for Wan, FramePack and Qwen-Image. Increases RAM usage but speeds up model loading in some cases."
        " / モデル読み込み時のnumpyメモリマッピングを無効にします。Wan、FramePack、Qwen-Imageで有効です。RAM使用量が増えますが、場合によってはモデルの読み込みが高速化されます。",
    )

    # parser.add_argument("--flow_shift", type=float, default=7.0, help="Shift factor for flow matching schedulers")
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0, help="Embeded classifier free guidance scale (HunyuanVideo only)."
    )
    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "qwen_shift", "logsnr", "qinglong_flux", "qinglong_qwen"],
        default="sigma",
        help="Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal, shift of sigmoid and flux shift."
        " / タイムステップをサンプリングする方法：sigma、random uniform、random normalのsigmoid、sigmoidのシフト、flux shift。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 1.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは1.0。",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help='Scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid" or "shift"). / sigmoidタイムステップサンプリングの倍率（timestep-samplingが"sigmoid"または"shift"の場合のみ有効）。',
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["logit_normal", "mode", "cosmap", "sigma_sqrt", "none"],
        help="weighting scheme for timestep distribution. Default is none / タイムステップ分布の重み付けスキーム、デフォルトはnone",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合の平均",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合のstd",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme` / モード重み付けスキームのスケール",
    )
    parser.add_argument(
        "--min_timestep",
        type=int,
        default=None,
        help="set minimum time step for training (0~999, default is 0) / 学習時のtime stepの最小値を設定する（0~999で指定、省略時はデフォルト値(0)） ",
    )
    parser.add_argument(
        "--max_timestep",
        type=int,
        default=None,
        help="set maximum time step for training (1~1000, default is 1000) / 学習時のtime stepの最大値を設定する（1~1000で指定、省略時はデフォルト値(1000)）",
    )
    parser.add_argument(
        "--preserve_distribution_shape",
        action="store_true",
        help="If specified, constrains timestep sampling to [min_timestep, max_timestep] "
        "using rejection sampling, preserving the original distribution shape. "
        "By default, the [0, 1] range is scaled, which distorts the distribution. Only effective when `timestep_sampling` is not 'sigma'."
        " / 指定すると、タイムステップのサンプリングを[最小タイムステップ、最大タイムステップ]に制約し、元の分布形状を保持します。"
        "デフォルトでは、[0, 1]の範囲がスケーリングされ、分布が歪むことがあります。timestep_samplingがsigma以外で有効です。",
    )
    parser.add_argument(
        "--num_timestep_buckets",
        type=int,
        default=None,
        help=(
            "Number of buckets for timestep sampling. Default is None, which disables bucketing. "
            "Set to 2 or more to enable stratified sampling. This forces timesteps to be sampled "
            "uniformly from the [0, 1] range, which can improve training stability, especially for small datasets."
            " / timestepサンプリングのバケット数。デフォルトはNoneで、バケット化を無効にします。"
            "2以上に設定すると、層化抽出が有効になり、タイムステップが[0, 1]の範囲から均等にサンプリングされるようになります。"
            "これは、特に小規模なデータセットでの学習の安定性向上が期待できます。"
        ),
    )

    parser.add_argument(
        "--show_timesteps",
        type=str,
        default=None,
        choices=["image", "console"],
        help="show timesteps in image or console, and return to console / タイムステップを画像またはコンソールに表示し、コンソールに戻る",
    )

    # network settings
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        default=False,
        help="Use Hugging Face PEFT for LoRA training (avoids ROCm tensor transfer bug on Windows) / Hugging Face PEFTを使用してLoRA学習を行う（WindowsでのROCmテンソル転送バグを回避）",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )

    # save and load settings
    parser.add_argument(
        "--output_dir", type=str, default=None, help="directory to output trained model / 学習後のモデル出力先ディレクトリ"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="base name of trained model file / 学習後のモデルの拡張子を除くファイル名",
    )
    parser.add_argument("--resume", type=str, default=None, help="saved state to resume training / 学習再開するモデルのstate")

    parser.add_argument(
        "--save_every_n_epochs",
        type=int,
        default=None,
        help="save checkpoint every N epochs / 学習中のモデルを指定エポックごとに保存する",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=None,
        help="save checkpoint every N steps / 学習中のモデルを指定ステップごとに保存する",
    )
    parser.add_argument(
        "--save_last_n_epochs",
        type=int,
        default=None,
        help="save last N checkpoints when saving every N epochs (remove older checkpoints) / 指定エポックごとにモデルを保存するとき最大Nエポック保存する（古いチェックポイントは削除する）",
    )
    parser.add_argument(
        "--save_last_n_epochs_state",
        type=int,
        default=None,
        help="save last N checkpoints of state (overrides the value of --save_last_n_epochs)/ 最大Nエポックstateを保存する（--save_last_n_epochsの指定を上書きする）",
    )
    parser.add_argument(
        "--save_last_n_steps",
        type=int,
        default=None,
        help="save checkpoints until N steps elapsed (remove older checkpoints if N steps elapsed) / 指定ステップごとにモデルを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する）",
    )
    parser.add_argument(
        "--save_last_n_steps_state",
        type=int,
        default=None,
        help="save states until N steps elapsed (remove older states if N steps elapsed, overrides --save_last_n_steps) / 指定ステップごとにstateを保存するとき、このステップ数経過するまで保存する（このステップ数経過したら削除する。--save_last_n_stepsを上書きする）",
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="save training state additionally (including optimizer states etc.) when saving model / optimizerなど学習状態も含めたstateをモデル保存時に追加で保存する",
    )
    parser.add_argument(
        "--save_state_on_train_end",
        action="store_true",
        help="save training state (including optimizer states etc.) on train end even if --save_state is not specified"
        " / --save_stateが未指定時にもoptimizerなど学習状態も含めたstateを学習終了時に保存する",
    )

    # SAI Model spec
    parser.add_argument(
        "--metadata_title",
        type=str,
        default=None,
        help="title for model metadata (default is output_name) / メタデータに書き込まれるモデルタイトル、省略時はoutput_name",
    )
    parser.add_argument(
        "--metadata_author",
        type=str,
        default=None,
        help="author name for model metadata / メタデータに書き込まれるモデル作者名",
    )
    parser.add_argument(
        "--metadata_description",
        type=str,
        default=None,
        help="description for model metadata / メタデータに書き込まれるモデル説明",
    )
    parser.add_argument(
        "--metadata_license",
        type=str,
        default=None,
        help="license for model metadata / メタデータに書き込まれるモデルライセンス",
    )
    parser.add_argument(
        "--metadata_tags",
        type=str,
        default=None,
        help="tags for model metadata, separated by comma / メタデータに書き込まれるモデルタグ、カンマ区切り",
    )
    parser.add_argument(
        "--metadata_reso",
        type=str,
        default=None,
        help="resolution for model metadata (e.g., `1024,1024`) / メタデータに書き込まれるモデル解像度（例: `1024,1024`）",
    )
    parser.add_argument(
        "--metadata_arch",
        type=str,
        default=None,
        help="architecture for model metadata / メタデータに書き込まれるモデルアーキテクチャ",
    )

    # huggingface settings
    parser.add_argument(
        "--huggingface_repo_id",
        type=str,
        default=None,
        help="huggingface repo name to upload / huggingfaceにアップロードするリポジトリ名",
    )
    parser.add_argument(
        "--huggingface_repo_type",
        type=str,
        default=None,
        help="huggingface repo type to upload / huggingfaceにアップロードするリポジトリの種類",
    )
    parser.add_argument(
        "--huggingface_path_in_repo",
        type=str,
        default=None,
        help="huggingface model path to upload files / huggingfaceにアップロードするファイルのパス",
    )
    parser.add_argument("--huggingface_token", type=str, default=None, help="huggingface token / huggingfaceのトークン")
    parser.add_argument(
        "--huggingface_repo_visibility",
        type=str,
        default=None,
        help="huggingface repository visibility ('public' for public, 'private' or None for private) / huggingfaceにアップロードするリポジトリの公開設定（'public'で公開、'private'またはNoneで非公開）",
    )
    parser.add_argument(
        "--save_state_to_huggingface", action="store_true", help="save state to huggingface / huggingfaceにstateを保存する"
    )
    parser.add_argument(
        "--resume_from_huggingface",
        action="store_true",
        help="resume from huggingface (ex: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type}) / huggingfaceから学習を再開する(例: --resume {repo_id}/{path_in_repo}:{revision}:{repo_type})",
    )
    parser.add_argument(
        "--async_upload",
        action="store_true",
        help="upload to huggingface asynchronously / huggingfaceに非同期でアップロードする",
    )

    parser.add_argument("--dit", type=str, help="DiT checkpoint path / DiTのチェックポイントのパス")
    parser.add_argument("--vae", type=str, help="VAE checkpoint path / VAEのチェックポイントのパス")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is float16")

    return parser


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    if not os.path.exists(config_path):
        logger.info(f"{config_path} not found.")
        exit(1)

    logger.info(f"Loading settings from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]
    logger.info(args.config_file)

    return args


def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """HunyuanVideo specific parser setup"""
    # model settings
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for DiT, default is bfloat16")
    parser.add_argument("--dit_in_channels", type=int, default=16, help="input channels for DiT, default is 16, skyreels I2V is 32")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for LLM / LLMにfp8を使う")
    parser.add_argument("--text_encoder1", type=str, help="Text Encoder 1 directory / テキストエンコーダ1のディレクトリ")
    parser.add_argument("--text_encoder2", type=str, help="Text Encoder 2 directory / テキストエンコーダ2のディレクトリ")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encoder, default is float16")
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled."
        " / VAEの空間タイリングを有効にする、デフォルトはFalse。vae_spatial_tile_sample_min_sizeが設定されている場合、自動的に有効になります。",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.fp8_scaled = False  # HunyuanVideo does not support this yet

    trainer = NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
