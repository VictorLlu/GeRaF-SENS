# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
# removed mmengine import
from torch import Tensor
import numpy as np

from geraf.utils.registry import MODELS
from geraf.utils import OptConfigType, OptMultiConfig

from geraf.models.model_utils.field import make_predictor

# Positional encoding embedding. Code was taken from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class NGPEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super(NGPEmbedder, self).__init__()
        raise RuntimeError(
            "NGPEmbedder depends on RFGridEncoder, which is not retained in the standalone lite build."
        )

    def forward(self, inputs):
        raise RuntimeError(
            "NGPEmbedder is unavailable in the standalone lite build because RFGridEncoder was removed."
        )

def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim
    

@MODELS.register_module()
class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 sdf_activation='none',
                 layer_activation='softplus'):
        super(SDFNetwork, self).__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if layer_activation=='softplus':
            self.activation = nn.Softplus(beta=100)
        elif layer_activation=='relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def sdf_normal(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return y[...,:1].detach(), gradients.detach()
    

@MODELS.register_module()
class SDFNGPNetwork(nn.Module):
    grid_level_interval: int = 2
    grid_level_dim: int = 4
    grid_base_resolution: int = 16
    grid_log2_hashmap_size: int = 19
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 grid_num_levels = 10,
                 bias=0.5,
                 scale=1,
                 init_value=1e0,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 sdf_activation='none',
                 layer_activation='softplus'):
        super(SDFNGPNetwork, self).__init__()
        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.pos_embed = None

        if grid_num_levels > 0:
            self.pos_embed = NGPEmbedder(input_dim=3,
                                   num_levels=grid_num_levels,
                                   level_dim=self.grid_level_dim,
                                   base_resolution=self.grid_base_resolution,
                                   per_level_scale=2,
                                   desired_resolution=None,
                                   log2_hashmap_size=self.grid_log2_hashmap_size,
                                   gridtype='hash',
                                   align_corners=False, 
                                   init_value=init_value)
            dims[0] = self.pos_embed.output_dim

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif grid_num_levels > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif grid_num_levels > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if layer_activation=='softplus':
            self.activation = nn.Softplus(beta=100)
        elif layer_activation=='relu':
            self.activation = nn.ReLU()
        else:
            raise NotImplementedError
    
    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.pos_embed is not None:
            inputs = self.pos_embed(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

    def sdf(self, x):
        return self.forward(x)[..., :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients

    def sdf_normal(self, x):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return y[...,:1].detach(), gradients.detach()


@MODELS.register_module()
class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val, activation='exp'):
        super(SingleVarianceNetwork, self).__init__()
        self.act = activation
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        device = x.device
        if self.act=='exp':
            return torch.ones([*x.shape[:-1], 1], device=device) * torch.exp(self.variance * 10.0)
        elif self.act=='linear':
            return torch.ones([*x.shape[:-1], 1], device=device) * self.variance * 10.0
        elif self.act=='square':
            return torch.ones([*x.shape[:-1], 1], device=device) * (self.variance * 10.0) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        device = x.device
        return torch.ones([*x.shape[:-1], 1], device=device) * inv_s


@MODELS.register_module()
class SingleVarianceNetworkDecay(nn.Module):
    def __init__(self, init_val, scale=0.01, activation='exp'):
        super(SingleVarianceNetworkDecay, self).__init__()
        self.act = activation
        self.scale = scale
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        device = x.device
        if self.act=='exp':
            return torch.ones([*x.shape[:-1], 1], device=device) * torch.exp(self.variance * self.scale)
        elif self.act=='linear':
            return torch.ones([*x.shape[:-1], 1], device=device) * self.variance * self.scale
        elif self.act=='square':
            return torch.ones([*x.shape[:-1], 1], device=device) * (self.variance * self.scale) ** 2
        else:
            raise NotImplementedError

    def warp(self, x, inv_s):
        device = x.device
        return torch.ones([*x.shape[:-1], 1], device=device) * inv_s
   

@MODELS.register_module()
class ReflectivePowerNetworkInit(nn.Module):
    default_cfg={
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
        'reflective_init': 0.5,
        'light_act': 'exp',
        'refelective_act': 'sigmoid',
        'light_power': 0.0
    }
    def __init__(self, cfg, feats_dim=256):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}

        # self.roughness_predictor = make_predictor(feats_dim+3, 1)
        # if self.cfg['roughness_init']!=0:
        #     nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.refelective_predictor = make_predictor(feats_dim + 3, 1, activation=self.cfg['refelective_act'], exp_max=self.cfg['refelective_exp_max'])
        self.register_parameter('light_power', nn.Parameter(torch.tensor(self.cfg['light_power'])))
        # === Initialization to output ~1 ===
        self._init_reflective_all_ones(self.cfg['reflective_init'])

    def _init_reflective_all_ones(self, target: float = 0.999):
        """Force refelective predictor to start with outputs ≈ target for any input."""
        eps = 1e-6
        t = max(min(target, 1 - eps), eps)
        seq = self.refelective_predictor
        last_linear_idx = len(seq) - 2
        assert isinstance(seq[last_linear_idx], nn.Linear), \
            "Expected last module before activation to be Linear."

        for i, m in enumerate(seq):
            if isinstance(m, nn.Linear):
                is_last = (i == last_linear_idx)
                nn.init.constant_(m.weight, 0.0)
                nn.init.constant_(m.bias, 0.0)
                if is_last:
                    act = seq[last_linear_idx + 1]
                    if isinstance(act, nn.Sigmoid):
                        bias_val = np.log(t / (1 - t))
                    elif isinstance(act, nn.Softplus):
                        bias_val = np.log(np.exp(t) - 1.0)
                    else:
                        bias_val = t
                    nn.init.constant_(m.bias, bias_val)

    def forward(self, points, feature_vectors, trans_prob):
        # roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        refelective = self.refelective_predictor(torch.cat([feature_vectors, points], -1))

        color = refelective.squeeze(-1) * torch.exp(self.light_power) * trans_prob
        color = torch.clamp(color, min=1e-6)
        return color
    
    def get_refelectivness(self, points, feature_vectors):
        refelective = self.refelective_predictor(torch.cat([feature_vectors, points], -1))
        return refelective


@MODELS.register_module()
class ReflectivePowerNetwork(nn.Module):
    default_cfg={
        'human_light': False,
        'sphere_direction': False,
        'light_pos_freq': 8,
        'inner_init': -0.95,
        'roughness_init': 0.0,
        'metallic_init': 0.0,
        'light_exp_max': 0.0,
        'light_act': 'exp',
        'refelective_act': 'sigmoid',
        'light_power': 0.0
    }
    def __init__(self, cfg, feats_dim=256):
        super().__init__()
        self.cfg={**self.default_cfg, **cfg}

        # self.roughness_predictor = make_predictor(feats_dim+3, 1)
        # if self.cfg['roughness_init']!=0:
        #     nn.init.constant_(self.roughness_predictor[-2].bias, self.cfg['roughness_init'])
        self.refelective_predictor = make_predictor(feats_dim + 3, 1, activation=self.cfg['refelective_act'], exp_max=self.cfg['refelective_exp_max'])
        self.register_parameter('light_power', nn.Parameter(torch.tensor(self.cfg['light_power'])))
        # === Initialization to output ~1 ===

    def forward(self, points, feature_vectors, trans_prob):
        # roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        refelective = self.refelective_predictor(torch.cat([feature_vectors, points], -1))

        color = refelective.squeeze(-1) * torch.exp(self.light_power) * trans_prob
        color = torch.clamp(color, min=1e-6)
        return color
    
    def get_refelectivness(self, points, feature_vectors):
        refelective = self.refelective_predictor(torch.cat([feature_vectors, points], -1))
        return refelective


@MODELS.register_module()
class ReflectivePowerNetworkInitV2(ReflectivePowerNetworkInit):
    def __init__(self, cfg, feats_dim=256):
        super().__init__(
            cfg=cfg,
            feats_dim=feats_dim,
        )

    def forward(self, points, feature_vectors, trans_prob, cfg, step):
        # roughness = self.roughness_predictor(torch.cat([feature_vectors, points], -1))
        refelective = self.refelective_predictor(torch.cat([feature_vectors, points], -1))
        if cfg['freeze_ref_step'] is not None and step < cfg['freeze_ref_step']:
            refelective = refelective.detach()

        color = refelective.squeeze(-1) * torch.exp(self.light_power) * trans_prob
        color = torch.clamp(color, min=1e-6)
        return color
