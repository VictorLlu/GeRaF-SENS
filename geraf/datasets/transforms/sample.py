# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .base_transform import BaseTransform
from geraf.utils.registry import TRANSFORMS

np.random.seed(42)


@TRANSFORMS.register_module()
class GetPrimeRay(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        rot = np.load(os.path.join(results["data_path"], "rotation.npy"))
        results["ant_dir"] = rot

        x_axis = (rot @ np.array([[1.0, 0.0, 0.0]]).T).squeeze()
        y_axis = (rot @ np.array([[0.0, 1.0, 0.0]]).T).squeeze()
        z_axis = (rot @ np.array([[0.0, 0.0, 1.0]]).T).squeeze()
        results["ant_coord"] = (x_axis, y_axis, z_axis)

        local_dir = x_axis.reshape(1, 3)
        local_dir /= np.linalg.norm(local_dir, axis=-1, keepdims=True)
        results["ray_d"] = local_dir
        return results


@TRANSFORMS.register_module()
class UniformRaySampler(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        sample_cfg = results["sample_cfg"]
        ray_d = results["ray_d"].copy()
        _, y_axis, z_axis = results["ant_coord"]

        n_aperture = sample_cfg["N_aperture"]
        lin = np.linspace(-1, 1, n_aperture, endpoint=False) + 1.0 / n_aperture / 2
        grid_y, grid_z = np.meshgrid(lin, lin)
        grid_points = np.stack([grid_y.ravel(), grid_z.ravel()], axis=1)

        valid_cells = grid_points[np.sum(grid_points**2, axis=1) <= 1.0]
        cell_size = 2.0 / n_aperture
        offsets = (np.random.rand(len(valid_cells), 2) - 0.5) * cell_size
        sampled_2d = valid_cells + offsets

        y_v = sampled_2d[:, 0]
        z_v = sampled_2d[:, 1]
        ray_o = np.outer(y_v, y_axis) + np.outer(z_v, z_axis)

        o_dot_d = np.sum(ray_o * ray_d, axis=1)
        o_norm_sq = np.sum(ray_o**2, axis=1)
        discriminant = (2 * o_dot_d) ** 2 - 4 * (o_norm_sq - 1.0)
        sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
        t_near = (-2 * o_dot_d - sqrt_disc) / 2
        t_far = (-2 * o_dot_d + sqrt_disc) / 2

        valid_mask = (discriminant >= 0) & ((t_far - t_near) > 1e-1)
        ray_o = ray_o[valid_mask]
        t_near = t_near[valid_mask]
        t_far = t_far[valid_mask]

        n_rays = ray_o.shape[0]
        ray_d = ray_d.repeat(n_rays, axis=0)
        results["near"] = t_near
        results["far"] = t_far
        results["ray_d"] = ray_d
        results["ray_o"] = ray_o

        n_samples = sample_cfg["N_samples"]
        t_vals = np.linspace(0, 1, n_samples + 1)[None, :]
        z_vals = t_near[:, None] * (1 - t_vals) + t_far[:, None] * t_vals
        rand_offsets = np.random.rand(n_rays, n_samples) * (z_vals[:, 1:] - z_vals[:, :-1])
        z_vals = z_vals[:, :-1] + rand_offsets

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = np.concatenate([dists, dists[:, -1:]], axis=-1)
        mid_z_vals = z_vals + 0.5 * dists
        results["z_vals"] = mid_z_vals
        results["dists"] = dists

        pts = ray_o[:, None, :] + mid_z_vals[:, :, None] * ray_d[:, None, :]
        results["sampled_poses_norm"] = pts
        normglb2glb = results["normglb2glb"]
        h_pts = np.concatenate([pts, np.ones_like(pts[..., :1])], axis=-1) @ normglb2glb.T
        results["sampled_poses_glb"] = h_pts[..., :3] / h_pts[..., 3:4]
        return results


@TRANSFORMS.register_module()
class TargetRaySampler(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        sample_cfg = results["sample_cfg"]
        ray_d = results["ray_d"]
        ray_o = results["ray_o"]

        n_ray = ray_o.shape[0]
        n_samples = sample_cfg["N_samples_tgt"]
        z_vals = np.linspace(-1, 1, n_samples, dtype=np.float32)[None, :].repeat(n_ray, axis=0)

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = np.concatenate([dists, dists[:, -1:]], axis=-1)
        mid_z_vals = z_vals + 0.5 * dists
        results["tgt_z_vals"] = mid_z_vals
        results["tgt_dists"] = dists

        pts = ray_o[:, None, :] + mid_z_vals[:, :, None] * ray_d[:, None, :]
        results["tgt_sampled_poses_norm"] = pts
        normglb2glb = results["normglb2glb"]
        h_pts = np.concatenate([pts, np.ones_like(pts[..., :1])], axis=-1) @ normglb2glb.T
        results["tgt_sampled_poses_glb"] = h_pts[..., :3] / h_pts[..., 3:4]
        return results


@TRANSFORMS.register_module()
class DynamicLossMask(BaseTransform):
    def __init__(self, thres_rate=0.05, tot_thres_rate=0.1) -> None:
        self.thres_rate = thres_rate
        self.tot_thres_rate = tot_thres_rate

    def transform(self, results: dict) -> Optional[dict]:
        n_arr, n_sample, _ = results["tgt_sampled_poses_norm"].shape
        mf = results["mf_sampled_value"].reshape(n_arr, n_sample)
        accmf = results["accmf_sampled_value"].reshape(n_arr, n_sample)

        threshold = mf.max() * self.thres_rate
        accmf_threshold = accmf.max() * self.tot_thres_rate
        inside_arr_mask = (accmf > accmf_threshold).sum(axis=-1).astype(bool)
        high_arr_mask = (mf > threshold).sum(axis=-1).astype(bool)
        low_arr_mask = np.logical_not(high_arr_mask)

        delete_array_mask = np.logical_and(low_arr_mask, inside_arr_mask)
        results["loss_mask"] = np.logical_not(delete_array_mask)
        return results
