# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from geraf.models.model_utils.field import sample_pdf
from geraf.utils import ConfigType, OptConfigType, OptMultiConfig
from geraf.utils.registry import MODELS


ForwardResults = Union[Dict[str, torch.Tensor], Tuple[torch.Tensor], torch.Tensor]


class BaseRendering(nn.Module, metaclass=ABCMeta):
    def __init__(self, eval_mode: str = "mesh", data_preprocessor: OptConfigType = None, init_cfg: OptMultiConfig = None):
        super().__init__()
        self.eval_mode = eval_mode

    def forward(self, data_sample, mode: str = "tensor") -> ForwardResults:
        if mode == "loss":
            return self.loss(data_sample)
        if mode == "predict":
            return self.predict(data_sample)
        if mode == "tensor":
            return self._forward(data_sample)
        raise RuntimeError(f'Invalid mode "{mode}".')

    @abstractmethod
    def loss(self, data_sample):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data_sample):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data_sample):
        raise NotImplementedError


@MODELS.register_module()
class NeusRendering(BaseRendering, metaclass=ABCMeta):
    def __init__(
        self,
        nerf: ConfigType,
        sdf_network: ConfigType,
        deviation_network: ConfigType,
        color_network: ConfigType,
        eval_mode: str = "mesh",
        data_preprocessor: OptConfigType = None,
        with_sigma_loss: float = None,
        with_grad_regression: float = None,
        loss_mode: str = "charbonier",
        norm_sdf: bool = True,
        loss_scale: float = 1.0,
        sample_cfg=None,
        radar_cfg=None,
        predict_chunk: int = 1024,
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(eval_mode=eval_mode, data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.nerf = MODELS.build(nerf)
        self.sdf_network = MODELS.build(sdf_network)
        self.deviation_network = MODELS.build(deviation_network)
        self.color_network = MODELS.build(color_network)

        self.with_sigma_loss = with_sigma_loss
        self.with_grad_regression = with_grad_regression
        self.loss_scale = loss_scale
        self.loss_mode = loss_mode
        self.norm_sdf = norm_sdf
        self.cfg = sample_cfg
        self.n_samples = sample_cfg["N_samples"]
        self.n_outside = sample_cfg["n_outside"]
        self.perturb = sample_cfg["perturb"]
        self.n_importance = sample_cfg["n_importance"]
        self.up_sample_steps = sample_cfg["up_sample_steps"]
        self.radar_cfg = radar_cfg
        self.step = 0
        self.predict_chunk = predict_chunk

    def _forward(self, data_sample: dict):
        raise NotImplementedError("Tensor mode is not used by the standalone runner.")

    def update_step(self, step: int):
        self.step = step

    def get_anneal_val(self, step: int) -> float:
        anneal_end = self.cfg.get("anneal_end", -1)
        if anneal_end < 0:
            return 1.0
        return float(np.min([1.0, step / anneal_end]))

    def render_core(
        self,
        rays_o,
        rays_d,
        z_vals,
        sample_dist,
        sdf_network,
        deviation_network,
        color_network,
        background_alpha=None,
        background_sampled_color=None,
        background_rgb=None,
        cos_anneal_ratio=0.0,
    ):
        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device=dists.device).expand(dists[..., :1].shape)], dim=-1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dirs = rays_d[:, None, :].expand(pts.shape)
        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        sdf_nn_output = sdf_network(pts)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]
        gradients = sdf_network.gradient(pts).squeeze()
        sampled_color = color_network(pts, gradients, dirs, feature_vector).reshape(batch_size, n_samples, 3)

        inv_s = deviation_network(torch.zeros([1, 3], device=pts.device))[:, :1].clip(1e-6, 1e6)
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        true_cos = (dirs * gradients).sum(-1, keepdim=True)
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )

        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] + background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)

        trans = torch.cumprod(torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1.0 - alpha + 1e-7], dim=-1), dim=-1)[:, :-1]
        weights = alpha * trans
        weights_sum = weights.sum(dim=-1, keepdim=True)

        color = (sampled_color * weights[:, :, None]).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights_sum)

        depth = (weights[:, :n_samples] * mid_z_vals).sum(dim=1)
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        return {
            "color": color,
            "depth": depth,
            "sdf": sdf,
            "gradients": gradients.reshape(batch_size, n_samples, 3),
            "s_val": 1.0 / inv_s,
            "mid_z_vals": mid_z_vals,
            "alpha": alpha,
            "trans": trans,
            "weights": weights,
            "gradient_error": gradient_error,
        }

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size, device=index.device)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            sdf = sdf[(xx, index.reshape(-1))].reshape(batch_size, n_samples + n_importance)
        return z_vals, sdf

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        batch_size, n_samples = z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        radius = torch.linalg.norm(pts, ord=2, dim=-1)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        prev_cos_val = torch.cat([torch.zeros([batch_size, 1], device=cos_val.device), cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = next_z_vals - prev_z_vals
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1.0 - alpha + 1e-7], dim=-1),
            dim=-1,
        )[:, :-1]
        return sample_pdf(z_vals, weights, n_importance, det=True).detach()

    def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, nerf, background_rgb=None):
        batch_size, n_samples = z_vals.shape
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.tensor([sample_dist], device=dists.device).expand(dists[..., :1].shape)], dim=-1)
        mid_z_vals = z_vals + dists * 0.5

        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
        pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)
        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)
        density, sampled_color = nerf(pts, dirs)

        sampled_color = torch.sigmoid(sampled_color)
        alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones([batch_size, 1], device=alpha.device), 1.0 - alpha + 1e-7], dim=-1),
            dim=-1,
        )[:, :-1]
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {"color": color, "sampled_color": sampled_color, "alpha": alpha, "weights": weights}

    def loss(self, data_sample: dict):
        hit_mask = data_sample["hit_mask"]
        true_rgb = data_sample["color"][hit_mask]
        near = data_sample["near"][hit_mask, None]
        far = data_sample["far"][hit_mask, None]
        rays_o = data_sample["ray_o"][hit_mask, :]
        rays_d = data_sample["ray_d"][hit_mask, :]
        n_rays, _ = true_rgb.shape
        device = true_rgb.device

        sample_dist = 2.0 / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=device)

        if self.perturb > 0:
            t_rand = torch.rand([n_rays, 1], device=device) - 0.5
            z_vals = z_vals + t_rand * 2.0 / self.n_samples
            if self.n_outside > 0:
                mids = 0.5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], dim=-1)
                lower = torch.cat([z_vals_outside[..., :1], mids], dim=-1)
                t_rand = torch.rand([n_rays, z_vals_outside.shape[-1]], device=device)
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        background_alpha = None
        background_sampled_color = None
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(n_rays, self.n_samples)
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o, rays_d, z_vals, sdf, self.n_importance // self.up_sample_steps, 64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o, rays_d, z_vals, new_z_vals, sdf, last=(i + 1 == self.up_sample_steps))

        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, self.nerf)
            background_sampled_color = ret_outside["sampled_color"]
            background_alpha = ret_outside["alpha"]

        ret_fine = self.render_core(
            rays_o,
            rays_d,
            z_vals,
            sample_dist,
            self.sdf_network,
            self.deviation_network,
            self.color_network,
            background_rgb=None,
            background_alpha=background_alpha,
            background_sampled_color=background_sampled_color,
            cos_anneal_ratio=self.get_anneal_val(self.step),
        )
        color_error = ret_fine["color"] - true_rgb

        losses = {"color_loss": F.l1_loss(color_error, torch.zeros_like(color_error), reduction="sum") / n_rays}
        if self.with_grad_regression:
            losses["grad_loss"] = ret_fine["gradient_error"] * self.with_grad_regression
        return losses

    def predict(self, data_sample: dict):
        if self.eval_mode == "mesh":
            return self.mesh_predict(data_sample)
        if self.eval_mode == "nvs":
            return self.rgb_predict(data_sample)
        raise ValueError(f"Unsupported eval mode: {self.eval_mode}")

    def mesh_predict(self, data_sample: dict):
        tot_poses = data_sample["sampled_poses_norm"]
        nx, ny, nz = data_sample["sample_cfg"]["shape"]
        tot_poses = tot_poses.reshape(nx, ny, nz, 3).permute(2, 1, 0, 3).flatten(0, 1)
        num_chunk_pts = data_sample["sample_cfg"]["n_ray"]
        normglb_scale = data_sample["normglb_scale"]
        iters = tot_poses.shape[0] // num_chunk_pts + int(tot_poses.shape[0] % num_chunk_pts != 0)
        all_pred_sdfs = []

        for i in range(iters):
            start_idx = i * num_chunk_pts
            end_idx = min((i + 1) * num_chunk_pts, tot_poses.shape[0])
            chunk_poses = tot_poses[start_idx:end_idx]
            with torch.no_grad():
                sdf_nn_output = self.sdf_network(chunk_poses)
                sdf = sdf_nn_output[..., 0] / normglb_scale
            all_pred_sdfs.append(sdf.squeeze(-1).detach().cpu().numpy())

        pred_sdfs = np.concatenate(all_pred_sdfs, axis=0).reshape(nz, ny, nx).transpose(2, 1, 0)
        return {"sdfs": pred_sdfs}

    def rgb_predict(self, data_sample: dict):
        chunk_size = self.predict_chunk
        rays_d = data_sample["ray_d"]
        hs, ws = data_sample["sample_img_shape"]
        near = data_sample["near"]
        far = data_sample["far"]
        rays_o = data_sample["ray_o"]
        hit_mask = data_sample["hit_mask"]
        normglb_scale = data_sample["normglb_scale"]

        num_rays = rays_d.shape[0]
        color_fine_all = torch.zeros(num_rays, 3, device=rays_d.device)
        depth_all = torch.zeros(num_rays, device=rays_d.device)
        hit_idx = torch.nonzero(hit_mask, as_tuple=False).squeeze(-1)
        if hit_idx.numel() == 0:
            return {"color": color_fine_all.reshape(hs, ws, 3).cpu().numpy()}

        near = near[hit_mask, None]
        far = far[hit_mask, None]
        rays_o = rays_o[hit_mask, :]
        rays_d = rays_d[hit_mask, :]
        sample_dist = 2.0 / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_d.device)
        z_vals = near + (far - near) * z_vals[None, :]

        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=rays_d.device)
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples

        for start in range(0, rays_d.shape[0], chunk_size):
            end = min(start + chunk_size, rays_d.shape[0])
            local_idx = torch.arange(start, end, device=rays_d.device)
            idx = hit_idx[start:end]
            rays_d_chunk = rays_d[local_idx]
            rays_o_chunk = rays_o[local_idx]
            z_vals_chunk = z_vals[local_idx]

            background_alpha = None
            background_sampled_color = None
            if self.n_outside > 0:
                z_vals_feed = torch.cat([z_vals_chunk, z_vals_outside[local_idx]], dim=-1)
                z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
                ret_outside = self.render_core_outside(rays_o_chunk, rays_d_chunk, z_vals_feed, sample_dist, self.nerf)
                background_sampled_color = ret_outside["sampled_color"]
                background_alpha = ret_outside["alpha"]

            ret_fine = self.render_core(
                rays_o_chunk,
                rays_d_chunk,
                z_vals_chunk,
                sample_dist,
                self.sdf_network,
                self.deviation_network,
                self.color_network,
                background_rgb=None,
                background_alpha=background_alpha,
                background_sampled_color=background_sampled_color,
                cos_anneal_ratio=self.get_anneal_val(self.step),
            )
            color_fine_all[idx] = ret_fine["color"]
            depth_all[idx] = ret_fine["depth"] / normglb_scale

        return {
            "color": color_fine_all.reshape(hs, ws, 3).detach().cpu().numpy(),
            "depth": depth_all.reshape(hs, ws).detach().cpu().numpy(),
            "id": data_sample.get("id", 0),
        }
