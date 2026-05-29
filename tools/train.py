# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import argparse
from datetime import datetime
import math
import os
import random
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from geraf.datasets.transforms.common import to_device
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "PyTorch is not available in this shell. Activate your training environment first, "
            "then run `python train.py <config.py>`."
        ) from exc
    raise


def configure_runtime_env() -> None:
    cache_root = Path(".cache").resolve()
    matplotlib_cache = cache_root / "matplotlib"
    fontconfig_cache = cache_root / "fontconfig"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    fontconfig_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one of the two supported GeRaF configs.")
    parser.add_argument("config", type=str, help="Path to a Python config file")
    parser.add_argument("--work-dir", type=str, default=None, help="Directory for logs and checkpoints")
    parser.add_argument("--resume", action="store_true", help="Resume from work_dir/latest.pth")
    parser.add_argument("--resume-from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--max-iters", type=int, default=None, help="Override max training iterations")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count")
    return parser.parse_args()


def build_work_dir(config_path: str, cfg: Dict, cli_work_dir: Optional[str]) -> Path:
    if cli_work_dir:
        return Path(cli_work_dir).resolve()
    cfg_work_dir = cfg.get("work_dir")
    if cfg_work_dir:
        return Path(cfg_work_dir).resolve()
    return (Path("work_dirs") / Path(config_path).stem).resolve()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def setup_logging(work_dir: Path) -> Path:
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_{timestamp}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_path


def log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{timestamp} - {message}")


def format_eta(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def set_random_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def build_dataloader(dataset, dataloader_cfg: Dict, cli_num_workers: Optional[int]) -> DataLoader:
    num_workers = cli_num_workers if cli_num_workers is not None else dataloader_cfg.get("num_workers", 0)
    persistent_workers = dataloader_cfg.get("persistent_workers", False) and num_workers > 0
    sampler_cfg = dataloader_cfg.get("sampler", {}) or {}
    sampler_type = sampler_cfg.get("type", "")
    shuffle = sampler_cfg.get("shuffle", False)
    if sampler_type == "InfiniteSampler":
        shuffle = sampler_cfg.get("shuffle", True)

    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def build_optimizer(model: torch.nn.Module, optim_wrapper_cfg: Dict) -> torch.optim.Optimizer:
    optimizer_cfg = (optim_wrapper_cfg or {}).get("optimizer", {}).copy()
    optimizer_type = optimizer_cfg.pop("type", "AdamW")
    paramwise_cfg = (optim_wrapper_cfg or {}).get("paramwise_cfg", {}) or {}
    custom_keys = paramwise_cfg.get("custom_keys", {}) or {}

    named_params = list(model.named_parameters())
    used = set()
    param_groups = []

    for key, key_cfg in custom_keys.items():
        matched_params = []
        lr_mult = key_cfg.get("lr_mult", 1.0)
        decay_mult = key_cfg.get("decay_mult", 1.0)
        for name, param in named_params:
            if not param.requires_grad or name in used:
                continue
            if name.startswith(f"{key}.") or name == key:
                matched_params.append(param)
                used.add(name)
        if matched_params:
            group = {"params": matched_params}
            if "lr" in optimizer_cfg:
                group["lr"] = optimizer_cfg["lr"] * lr_mult
            if "weight_decay" in optimizer_cfg:
                group["weight_decay"] = optimizer_cfg.get("weight_decay", 0.0) * decay_mult
            param_groups.append(group)

    base_params = [param for name, param in named_params if param.requires_grad and name not in used]
    if base_params:
        param_groups.append({"params": base_params})

    optimizer_cls = getattr(torch.optim, optimizer_type, None)
    if optimizer_cls is None:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    return optimizer_cls(param_groups, **optimizer_cfg)


def get_global_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        raise RuntimeError("Optimizer has no parameter groups.")
    return max(float(group["lr"]) for group in optimizer.param_groups)


class IterLRScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler_cfgs: List[Dict], total_iters: int):
        self.optimizer = optimizer
        self.scheduler_cfgs = scheduler_cfgs or []
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.total_iters = total_iters

    def step(self, current_iter: int) -> None:
        multipliers = [1.0 for _ in self.base_lrs]
        for cfg in self.scheduler_cfgs:
            if cfg.get("by_epoch", False):
                continue
            begin = int(cfg.get("begin", 0))
            end = cfg.get("end")
            end = self.total_iters if end is None else int(end)
            if current_iter < begin:
                continue
            if current_iter >= end:
                progress = 1.0
            else:
                span = max(end - begin, 1)
                progress = min(max((current_iter - begin) / span, 0.0), 1.0)

            sched_type = cfg.get("type")
            factor = 1.0
            if sched_type == "LinearLR":
                start_factor = float(cfg.get("start_factor", 1.0))
                factor = start_factor + (1.0 - start_factor) * progress
            elif sched_type == "ExponentialLR":
                gamma = cfg.get("gamma")
                if gamma is None:
                    eta_min = float(cfg.get("eta_min", 0.0))
                    for group_idx, base_lr in enumerate(self.base_lrs):
                        if base_lr == 0:
                            continue
                        if eta_min <= 0.0:
                            lr = base_lr if progress == 0.0 else 0.0
                        else:
                            lr = base_lr * ((eta_min / base_lr) ** progress)
                        multipliers[group_idx] *= lr / base_lr
                    continue
                factor = float(gamma) ** max(current_iter - begin, 0)
            elif sched_type == "CosineAnnealingLR":
                eta_min = float(cfg.get("eta_min", 0.0))
                for group_idx, base_lr in enumerate(self.base_lrs):
                    cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
                    lr = eta_min + (base_lr - eta_min) * cosine
                    multipliers[group_idx] *= lr / base_lr if base_lr != 0 else 1.0
                continue
            else:
                continue

            for group_idx in range(len(multipliers)):
                multipliers[group_idx] *= factor

        for group_idx, group in enumerate(self.optimizer.param_groups):
            group["lr"] = self.base_lrs[group_idx] * multipliers[group_idx]

def summarize_losses(losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    log_vars = {}
    total_loss = None
    for key, value in losses.items():
        if not torch.is_tensor(value):
            continue
        scalar = value.mean()
        log_vars[key] = float(scalar.detach().item())
        total_loss = scalar if total_loss is None else total_loss + scalar
    if total_loss is None:
        raise RuntimeError("Model returned no tensor losses.")
    log_vars["loss"] = float(total_loss.detach().item())
    return total_loss, log_vars


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    work_dir: Path,
    cfg_path: str,
    scheduler_state: Optional[Dict] = None,
) -> Path:
    checkpoint = {
        "iter": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg_path,
    }
    if scheduler_state is not None:
        checkpoint["scheduler_state"] = scheduler_state

    ckpt_path = work_dir / f"iter_{iteration}.pth"
    latest_path = work_dir / "latest.pth"
    torch.save(checkpoint, ckpt_path)
    torch.save(checkpoint, latest_path)
    return ckpt_path


def load_checkpoint_file(path: str, map_location: str = "cpu") -> Dict:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}")
    return checkpoint


def load_model_weights(model: torch.nn.Module, checkpoint_path: str, strict: bool = False) -> None:
    checkpoint = load_checkpoint_file(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    if strict:
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
    else:
        model_state = model.state_dict()
        filtered_state = {}
        skipped_mismatch = []
        for key, value in state_dict.items():
            target_value = model_state.get(key)
            if target_value is None:
                continue
            if not isinstance(value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
                filtered_state[key] = value
                continue
            if target_value.shape != value.shape:
                skipped_mismatch.append((key, tuple(value.shape), tuple(target_value.shape)))
                continue
            filtered_state[key] = value
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        if skipped_mismatch:
            log_message(f"Skipped mismatched keys when loading {checkpoint_path}: {len(skipped_mismatch)}")
            for key, src_shape, dst_shape in skipped_mismatch:
                log_message(f"  {key}: checkpoint{src_shape} != model{dst_shape}")
    if missing:
        log_message(f"Missing keys when loading {checkpoint_path}: {len(missing)}")
        for key in missing:
            log_message(f"  {key}")
    if unexpected:
        log_message(f"Unexpected keys when loading {checkpoint_path}: {len(unexpected)}")
        for key in unexpected:
            log_message(f"  {key}")


def maybe_load_pretrained(model: torch.nn.Module, model_cfg: Dict) -> None:
    from geraf.utils import resolve_ckpt

    init_cfg = (model_cfg or {}).get("init_cfg")
    if not isinstance(init_cfg, dict):
        return
    if init_cfg.get("type") != "Pretrained":
        return
    checkpoint = init_cfg.get("checkpoint")
    if not checkpoint:
        return
    checkpoint = resolve_ckpt(checkpoint)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        log_message(f"Pretrained checkpoint not found, skipping init load: {checkpoint}")
        return
    log_message(f"Loading pretrained weights from {checkpoint_path}...")
    load_model_weights(model, str(checkpoint_path), strict=False)


def maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Optional[str],
) -> int:
    if not resume_path:
        return 0
    checkpoint = load_checkpoint_file(resume_path, map_location="cpu")
    state_dict = checkpoint.get("model") or checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"No model state found in checkpoint: {resume_path}")
    model.load_state_dict(state_dict, strict=False)
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    start_iter = int(checkpoint.get("iter", 0))
    log_message(f"Resumed training from {resume_path} at iter {start_iter}.")
    return start_iter


def resolve_resume_path(args: argparse.Namespace, cfg: Dict, work_dir: Path) -> Optional[str]:
    if args.resume_from:
        return args.resume_from
    if args.resume or cfg.get("resume"):
        latest = work_dir / "latest.pth"
        if latest.exists():
            return str(latest)
    return None


def build_train_components(cfg: Dict):
    from geraf.utils import DATASETS, MODELS

    import geraf.datasets  # noqa: F401
    import geraf.datasets.transforms  # noqa: F401
    import geraf.models.networks  # noqa: F401
    import geraf.models.rendering  # noqa: F401

    model = MODELS.build(cfg["model"])

    dataloader_cfg = (cfg.get("train_dataloader") or {}).copy()
    dataset_cfg = dataloader_cfg.pop("dataset", None)
    if dataset_cfg is None:
        dataset_cfg = cfg.get("data")
    if dataset_cfg is None:
        raise KeyError("Config must define train_dataloader.dataset or data.")
    dataset = DATASETS.build(dataset_cfg)
    return model, dataset, dataloader_cfg


def train() -> None:
    args = parse_args()
    configure_runtime_env()

    from geraf.utils import load_py_config

    config_path = Path(args.config).resolve()
    cfg = load_py_config(str(config_path))
    work_dir = build_work_dir(str(config_path), cfg, args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(work_dir)

    device = choose_device()
    seed_cfg = (cfg.get("randomness") or {}).get("seed")
    set_random_seed(seed_cfg)

    model, dataset, dataloader_cfg = build_train_components(cfg)
    maybe_load_pretrained(model, cfg.get("model", {}))
    model.to(device)
    model.train()

    dataloader = build_dataloader(dataset, dataloader_cfg, args.num_workers)
    optimizer = build_optimizer(model, cfg.get("optim_wrapper", {}))
    train_cfg = cfg.get("train_cfg", {}) or {}
    max_iters = args.max_iters if args.max_iters is not None else int(train_cfg.get("max_iters", cfg.get("max_iters", 0)))
    if max_iters <= 0:
        raise ValueError("max_iters must be set in the config or by --max-iters.")
    scheduler = IterLRScheduler(optimizer, cfg.get("param_scheduler", []), max_iters)

    logger_cfg = ((cfg.get("default_hooks") or {}).get("logger") or {})
    checkpoint_cfg = ((cfg.get("default_hooks") or {}).get("checkpoint") or {})
    log_interval = int(logger_cfg.get("interval", 10))
    save_interval = int(checkpoint_cfg.get("interval", max_iters))
    clip_grad_cfg = (cfg.get("optim_wrapper") or {}).get("clip_grad") or {}

    resume_path = resolve_resume_path(args, cfg, work_dir)
    start_iter = maybe_resume(model, optimizer, resume_path)

    log_message(f"Config: {config_path}")
    log_message(f"Work dir: {work_dir}")
    log_message(f"Log file: {log_path}")
    log_message(f"Device: {device}")
    log_message(f"Dataset size: {len(dataset)}")
    log_message(f"Max iters: {max_iters}")

    data_iter = iter(dataloader)
    train_start_time = time.time()
    for iteration in range(start_iter + 1, max_iters + 1):
        try:
            data_sample = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            data_sample = next(data_iter)

        to_device(data_sample, device, non_blocking=torch.cuda.is_available())

        optimizer.zero_grad(set_to_none=True)
        losses = model(data_sample, mode="loss")
        total_loss, log_vars = summarize_losses(losses)
        total_loss.backward()

        if clip_grad_cfg:
            max_norm = clip_grad_cfg.get("max_norm")
            norm_type = clip_grad_cfg.get("norm_type", 2.0)
            if max_norm is not None:
                clip_grad_norm_(model.parameters(), max_norm=max_norm, norm_type=norm_type)

        optimizer.step()
        scheduler.step(iteration)

        if iteration % log_interval == 0 or iteration == 1 or iteration == max_iters:
            lr = get_global_lr(optimizer)
            elapsed = time.time() - train_start_time
            completed_iters = max(iteration - start_iter, 1)
            avg_iter_time = elapsed / completed_iters
            remaining_iters = max_iters - iteration
            eta_time = format_eta(avg_iter_time * remaining_iters)
            metrics = " ".join(f"{key}={value:.6f}" for key, value in sorted(log_vars.items()))
            log_message(f"[Iter {iteration}/{max_iters}] lr={lr:.6e} eta_time={eta_time} {metrics}")

        if save_interval > 0 and (iteration % save_interval == 0 or iteration == max_iters):
            ckpt_path = save_checkpoint(model, optimizer, iteration, work_dir, str(config_path))
            log_message(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
