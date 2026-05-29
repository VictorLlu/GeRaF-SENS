# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
    from geraf.datasets.transforms.common import to_device
except ModuleNotFoundError as exc:
    if exc.name == "torch":
        raise SystemExit(
            "PyTorch is not available in this shell. Activate your environment first, "
            "then run `python test.py <config.py> <checkpoint.pth>`."
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
    parser = argparse.ArgumentParser(description="Run prediction and visualization from a config and checkpoint.")
    parser.add_argument("config", type=str, help="Path to a Python config file")
    parser.add_argument("checkpoint", type=str, help="Path to a checkpoint file")
    parser.add_argument("--work-dir", type=str, default=None, help="Directory for test outputs")
    parser.add_argument(
        "--save-mesh",
        nargs="?",
        const="",
        default=None,
        help="Save mesh.ply. Pass a path to override, or omit the value to save into the work dir.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count")
    return parser.parse_args()


def build_dataloader(dataset, dataloader_cfg: Dict, cli_num_workers: Optional[int]) -> DataLoader:
    num_workers = cli_num_workers if cli_num_workers is not None else dataloader_cfg.get("num_workers", 0)
    persistent_workers = dataloader_cfg.get("persistent_workers", False) and num_workers > 0
    return DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_work_dir(config_path: str, cfg: Dict, cli_work_dir: Optional[str]) -> Path:
    if cli_work_dir:
        return Path(cli_work_dir).absolute()
    cfg_work_dir = cfg.get("work_dir")
    if cfg_work_dir:
        return Path(cfg_work_dir).absolute()
    return (Path("work_dirs") / Path(config_path).stem).absolute()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Missing keys when loading {checkpoint_path}: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys when loading {checkpoint_path}: {len(unexpected)}")


def build_test_components(cfg: Dict):
    from geraf.utils import DATASETS, MODELS

    import geraf.datasets  # noqa: F401
    import geraf.datasets.transforms  # noqa: F401
    import geraf.models.networks  # noqa: F401
    import geraf.models.rendering  # noqa: F401

    model = MODELS.build(cfg["model"])

    dataloader_cfg = (
        cfg.get("test_dataloader")
        or cfg.get("val_dataloader")
        or cfg.get("train_dataloader")
        or {}
    ).copy()
    dataset_cfg = dataloader_cfg.pop("dataset", None)
    if dataset_cfg is None:
        raise KeyError("Config must define test_dataloader.dataset, val_dataloader.dataset, or train_dataloader.dataset.")
    dataset = DATASETS.build(dataset_cfg)
    return model, dataset, dataloader_cfg


def main() -> None:
    args = parse_args()
    configure_runtime_env()

    from geraf.hooks import RFMeshVisualizationHook, RGBMeshVisualizationHook
    from geraf.utils import load_py_config

    config_path = Path(args.config).absolute()
    checkpoint_path = Path(args.checkpoint).absolute()
    cfg = load_py_config(str(config_path))
    work_dir = build_work_dir(str(config_path), cfg, args.work_dir)
    out_dir = None
    if args.save_mesh is not None:
        out_dir = Path(args.save_mesh).absolute() if args.save_mesh else work_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    model, dataset, dataloader_cfg = build_test_components(cfg)
    load_model_weights(model, str(checkpoint_path), strict=False)
    model.to(device)
    model.eval()

    dataloader = build_dataloader(dataset, dataloader_cfg, args.num_workers)

    vis_cfg = ((cfg.get("default_hooks") or {}).get("visualization") or {}).copy()
    vis_type = vis_cfg.pop("type", None)
    if vis_type not in {None, "RFMeshVisualizationHook", "RGBMeshVisualizationHook"}:
        raise ValueError(f"Unsupported visualization hook type: {vis_type}")
    vis_cfg.setdefault("draw", True)
    if out_dir is not None:
        vis_cfg["test_out_dir"] = str(out_dir)
    hook_cls = RGBMeshVisualizationHook if vis_type == "RGBMeshVisualizationHook" else RFMeshVisualizationHook
    hook = hook_cls(**vis_cfg)

    print(f"Config: {os.path.relpath(config_path)}")
    print(f"Checkpoint: {os.path.relpath(checkpoint_path)}")
    print(f"Device: {device}")
    if out_dir is not None:
        print(f"Save mesh: {os.path.relpath(out_dir / 'mesh.ply')}")
    else:
        print("Save mesh: disabled")
    print(f"Dataset size: {len(dataset)}")

    with torch.no_grad():
        for idx, data_sample in enumerate(dataloader):
            to_device(data_sample, device, non_blocking=torch.cuda.is_available())
            outputs = model(data_sample, mode="predict")
            hook.after_test_iter(idx, outputs, out_dir)
            print(f"Processed {idx + 1}/{len(dataloader)}")


if __name__ == "__main__":
    main()
