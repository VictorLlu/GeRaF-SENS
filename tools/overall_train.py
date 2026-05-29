# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import argparse
from datetime import datetime
import importlib.util
from pathlib import Path
import pprint
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRAIN_ENTRYPOINT = REPO_ROOT / "tools" / "train.py"


def load_py_config(config_path: str):
    config_module_path = REPO_ROOT / "geraf" / "utils" / "config.py"
    spec = importlib.util.spec_from_file_location("overall_config_loader", config_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load config helper from {config_module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_py_config(config_path)


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{timestamp} - {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full GeRaF multi-stage training pipeline.")
    parser.add_argument("config", type=str, help="Path to the overall pipeline config")
    parser.add_argument("--work-dir", type=str, default=None, help="Overall work directory")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader worker count")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Explicitly resume incomplete stages from their latest checkpoints. "
        "Finished stages are still skipped automatically.",
    )
    return parser.parse_args()


def import_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "PyTorch is not available in this shell. Activate your training environment first, "
                "then run `python tools/overall_train.py <overall_config.py>`."
            ) from exc
        raise
    return torch


def setup_logging(work_dir: Path) -> Path:
    log_dir = work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"overall_train_{timestamp}.log"
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_path


def build_overall_work_dir(config_path: Path, pipeline_cfg: Dict[str, Any], cli_work_dir: Optional[str]) -> Path:
    if cli_work_dir:
        return Path(cli_work_dir).resolve()
    cfg_work_dir = pipeline_cfg.get("work_dir")
    if cfg_work_dir:
        return Path(cfg_work_dir).resolve()
    pipeline_name = pipeline_cfg.get("name", config_path.stem)
    return (REPO_ROOT / "work_dirs" / pipeline_name).resolve()


def load_checkpoint_file(path: Path) -> Dict[str, Any]:
    torch = import_torch()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}")
    return checkpoint


def get_checkpoint_iter(path: Path) -> int:
    checkpoint = load_checkpoint_file(path)
    return int(checkpoint.get("iter", 0))


def get_stage_max_iters(config_path: Path) -> int:
    cfg = load_py_config(str(config_path))
    train_cfg = cfg.get("train_cfg", {}) or {}
    max_iters = int(train_cfg.get("max_iters", cfg.get("max_iters", 0)))
    if max_iters <= 0:
        raise ValueError(f"max_iters must be positive in {config_path}")
    return max_iters


def stage_generated_config_dir(work_dir: Path) -> Path:
    generated_dir = work_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)
    return generated_dir


def create_stage_alias(alias_path: Path, target_path: Path) -> None:
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    if alias_path.exists() or alias_path.is_symlink():
        alias_path.unlink()
    try:
        alias_path.symlink_to(target_path.resolve())
    except OSError:
        shutil.copy2(target_path, alias_path)


def sync_stage_aliases(overall_work_dir: Path, stage_id: str, stage_work_dir: Path) -> int:
    latest_path = stage_work_dir / "latest.pth"
    if not latest_path.exists():
        raise FileNotFoundError(f"Latest checkpoint not found for stage '{stage_id}': {latest_path}")
    current_iter = get_checkpoint_iter(latest_path)
    iter_path = stage_work_dir / f"iter_{current_iter}.pth"
    if not iter_path.exists():
        iter_path = latest_path

    latest_alias = overall_work_dir / f"{stage_id}_latest.pth"
    iter_alias = overall_work_dir / f"{stage_id}_iter_{current_iter}.pth"
    create_stage_alias(latest_alias, latest_path)
    create_stage_alias(iter_alias, iter_path)
    return current_iter


def build_stage_overrides(stage_cfg: Dict[str, Any], overall_work_dir: Path) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    checkpoint_sources = stage_cfg.get("checkpoint_sources") or {}
    if not checkpoint_sources:
        return overrides

    model_overrides: Dict[str, Any] = {}
    tgt_sdf_stage = checkpoint_sources.get("tgt_sdf_ckpt_path")
    if tgt_sdf_stage:
        model_overrides["tgt_sdf_ckpt_path"] = str((overall_work_dir / f"{tgt_sdf_stage}_latest.pth").resolve())

    pretrained_stage = checkpoint_sources.get("pretrained")
    if pretrained_stage:
        model_overrides["init_cfg"] = dict(
            type="Pretrained",
            checkpoint=str((overall_work_dir / f"{pretrained_stage}_latest.pth").resolve()),
        )

    if model_overrides:
        overrides["model"] = model_overrides
    return overrides


def write_generated_stage_config(
    source_config_path: Path,
    generated_config_path: Path,
    overrides: Dict[str, Any],
) -> None:
    lines = [f"_base_ = [{str(source_config_path.resolve())!r}]\n"]
    for key, value in overrides.items():
        formatted = pprint.pformat(value, width=100, sort_dicts=False)
        lines.append(f"{key} = {formatted}\n")
    generated_config_path.write_text("".join(lines), encoding="utf-8")


def run_stage(
    stage_id: str,
    stage_config_path: Path,
    generated_config_path: Path,
    stage_work_dir: Path,
    max_iters: int,
    num_workers: Optional[int],
    resume: bool,
) -> None:
    latest_checkpoint = stage_work_dir / "latest.pth"
    current_iter = get_checkpoint_iter(latest_checkpoint) if latest_checkpoint.exists() else 0

    if current_iter >= max_iters:
        log_message(f"Stage '{stage_id}' already reached iter {current_iter}/{max_iters}; skipping.")
        return

    command = [
        sys.executable,
        str(TRAIN_ENTRYPOINT),
        str(generated_config_path),
        "--work-dir",
        str(stage_work_dir),
    ]
    if resume and latest_checkpoint.exists():
        command.append("--resume")
    if num_workers is not None:
        command.extend(["--num-workers", str(num_workers)])

    log_message(f"Launching stage '{stage_id}' with config {stage_config_path}")
    log_message(f"Stage '{stage_id}' work dir: {stage_work_dir}")
    if resume and latest_checkpoint.exists():
        log_message(f"Stage '{stage_id}' resuming from {latest_checkpoint}")

    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = parse_args()
    overall_config_path = Path(args.config).resolve()
    overall_cfg = load_py_config(str(overall_config_path))
    pipeline_cfg = overall_cfg.get("pipeline")
    if not isinstance(pipeline_cfg, dict):
        raise KeyError("Overall config must define a top-level `pipeline` dict.")

    overall_work_dir = build_overall_work_dir(overall_config_path, pipeline_cfg, args.work_dir)
    overall_work_dir.mkdir(parents=True, exist_ok=True)
    log_path = setup_logging(overall_work_dir)

    log_message(f"Overall config: {overall_config_path}")
    log_message(f"Overall work dir: {overall_work_dir}")
    log_message(f"Overall log file: {log_path}")

    generated_dir = stage_generated_config_dir(overall_work_dir)
    stages = pipeline_cfg.get("stages") or []
    if not stages:
        raise ValueError("Pipeline config must define at least one stage.")

    for stage in stages:
        stage_id = stage["id"]
        stage_config_path = (REPO_ROOT / stage["config"]).resolve()
        max_iters = get_stage_max_iters(stage_config_path)
        stage_work_dir = overall_work_dir / stage_id
        stage_work_dir.mkdir(parents=True, exist_ok=True)

        overrides = build_stage_overrides(stage, overall_work_dir)
        generated_config_path = generated_dir / f"{stage_id}.py"
        write_generated_stage_config(stage_config_path, generated_config_path, overrides)

        latest_checkpoint = stage_work_dir / "latest.pth"
        current_iter = get_checkpoint_iter(latest_checkpoint) if latest_checkpoint.exists() else 0
        if current_iter >= max_iters:
            log_message(f"Stage '{stage_id}' already complete at iter {current_iter}/{max_iters}.")
            synced_iter = sync_stage_aliases(overall_work_dir, stage_id, stage_work_dir)
            log_message(f"Stage '{stage_id}' aliases updated at iter {synced_iter}.")
            continue

        run_stage(
            stage_id=stage_id,
            stage_config_path=stage_config_path,
            generated_config_path=generated_config_path,
            stage_work_dir=stage_work_dir,
            max_iters=max_iters,
            num_workers=args.num_workers,
            resume=args.resume,
        )
        synced_iter = sync_stage_aliases(overall_work_dir, stage_id, stage_work_dir)
        log_message(f"Stage '{stage_id}' completed or updated to iter {synced_iter}.")

    log_message("Overall pipeline finished.")


if __name__ == "__main__":
    main()
