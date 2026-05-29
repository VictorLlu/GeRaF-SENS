# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import numpy as np
import torch


def to_device(data, device, non_blocking: bool = False):
    """Move all torch.Tensor values in a Mapping to device, in-place. Returns data."""
    for key in list(data.keys()):
        value = data[key]
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device, non_blocking=non_blocking)
    return data


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, (list, tuple)):
        return torch.tensor(data)
    if isinstance(data, (int, float, bool)):
        return torch.tensor(data)
    raise TypeError(f"Cannot convert {type(data)!r} to tensor")
