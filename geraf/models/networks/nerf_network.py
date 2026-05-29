# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from geraf.models.networks.sdf_network import get_embedder
from geraf.utils.registry import MODELS


@MODELS.register_module()
class RenderingNetwork(nn.Module):
    def __init__(
        self,
        d_feature,
        mode,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        weight_norm=True,
        multires_view=0,
        squeeze_out=True,
    ):
        super().__init__()
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += input_ch - 3

        self.num_layers = len(dims)
        for layer_idx in range(self.num_layers - 1):
            linear = nn.Linear(dims[layer_idx], dims[layer_idx + 1])
            if weight_norm:
                linear = nn.utils.weight_norm(linear)
            setattr(self, f"lin{layer_idx}", linear)
        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == "idr":
            x = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == "no_view_dir":
            x = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == "no_normal":
            x = torch.cat([points, view_dirs, feature_vectors], dim=-1)
        else:
            raise ValueError(f"Unsupported rendering mode: {self.mode}")

        for layer_idx in range(self.num_layers - 1):
            linear = getattr(self, f"lin{layer_idx}")
            x = linear(x)
            if layer_idx < self.num_layers - 2:
                x = self.relu(x)
        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x


@MODELS.register_module()
class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        d_in=3,
        d_in_view=3,
        multires=0,
        multires_view=0,
        output_ch=4,
        skips=[4],
        use_viewdirs=False,
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            self.embed_fn, self.input_ch = get_embedder(multires, input_dims=d_in)
        if multires_view > 0:
            self.embed_fn_view, self.input_ch_view = get_embedder(multires_view, input_dims=d_in_view)

        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)]
            + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)]
        )
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for layer_idx, linear in enumerate(self.pts_linears):
            h = F.relu(linear(h))
            if layer_idx in self.skips:
                h = torch.cat([input_pts, h], dim=-1)

        if not self.use_viewdirs:
            raise RuntimeError("This standalone NeRF path expects use_viewdirs=True.")

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        h = torch.cat([feature, input_views], dim=-1)
        for linear in self.views_linears:
            h = F.relu(linear(h))
        rgb = self.rgb_linear(h)
        return alpha, rgb
