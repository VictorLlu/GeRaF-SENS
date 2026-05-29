# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import torch
import torch.nn as nn

from geraf.utils.registry import MODELS


@MODELS.register_module()
class FixedPowerNetwork(nn.Module):
    default_cfg = {
        "refelective_act": "sigmoid",
        "light_power": 0.0,
    }

    def __init__(self, cfg, feats_dim=256):
        super().__init__()
        self.cfg = {**self.default_cfg, **cfg}
        self.register_parameter("light_power", nn.Parameter(torch.tensor(self.cfg["light_power"])))

    def forward(self, points, feature_vectors, trans_prob):
        color = torch.exp(self.light_power) * trans_prob
        return torch.clamp(color, min=1e-6)

    def get_refelectivness(self, points, feature_vectors):
        return torch.ones(points.shape[0], points.shape[1], device=points.device, dtype=points.dtype)
