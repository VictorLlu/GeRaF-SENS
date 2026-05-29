# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path
import collections.abc


def resolve_ckpt(path):
    p = Path(path)
    if p.suffix == ".py":
        return f"work_dirs/{p.stem}/latest.pth"
    return str(path)


def deep_update(dst, src):
    for key, value in src.items():
        if isinstance(value, collections.abc.Mapping) and isinstance(dst.get(key), dict):
            dst[key] = deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_py_config(config_path):
    config_path = Path(config_path).resolve()
    namespace = {"__file__": str(config_path), "__name__": "__main__"}
    with config_path.open("r", encoding="utf-8") as handle:
        code = compile(handle.read(), str(config_path), "exec")
        exec(code, namespace, namespace)

    config = {
        key: value
        for key, value in namespace.items()
        if not key.startswith("__")
    }

    base_paths = config.pop("_base_", None)
    if base_paths:
        if isinstance(base_paths, str):
            base_paths = [base_paths]
        merged = {}
        for base_path in base_paths:
            deep_update(merged, load_py_config(config_path.parent / base_path))
        deep_update(merged, config)
        config = merged

    if "data" not in config and "train_dataloader" in config:
        train_dataloader = config.get("train_dataloader") or {}
        if isinstance(train_dataloader, dict) and "dataset" in train_dataloader:
            config["data"] = train_dataloader["dataset"]

    return config
