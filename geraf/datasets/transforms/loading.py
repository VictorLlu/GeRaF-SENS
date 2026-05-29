# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.image as mpimg

from .base_transform import BaseTransform
from geraf.utils import grid_num
from geraf.utils.registry import TRANSFORMS

np.random.seed(42)


def _sample_volume(volume: np.ndarray, points: np.ndarray, mode: str, align_corners: bool) -> np.ndarray:
    normalized_poses = np.stack([points[:, 2], points[:, 1], points[:, 0]], axis=-1)
    grid = torch.tensor(normalized_poses, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(volume_tensor, grid, mode=mode, align_corners=align_corners)
    return sampled.squeeze().detach().cpu().numpy()


@TRANSFORMS.register_module()
class LoadRF(BaseTransform):
    def __init__(self, normalization: bool = False) -> None:
        self.normalization = normalization

    def transform(self, results: dict) -> Optional[dict]:
        radar_cfg = results["radar_cfg"]
        sample_cfg = results["sample_cfg"]
        data_path = results["data_path"]
        results["data_name"] = os.path.basename(data_path)

        bound = sample_cfg["bound"]
        mf_res = sample_cfg["mf_res"]
        nx = grid_num(bound[1], bound[0], mf_res[0])
        ny = grid_num(bound[3], bound[2], mf_res[1])
        nz = grid_num(bound[5], bound[4], mf_res[2])
        trans_power = radar_cfg["trans_power"]

        mf_image_path = os.path.join(data_path, "sarimage.bin")
        accmf_image_path = results["accmf_path"]

        if os.path.exists(mf_image_path):
            mf_image = np.fromfile(mf_image_path, dtype=np.float32).reshape(nx, ny, nz)
            if self.normalization:
                mf_image /= trans_power
            results["mf_image"] = mf_image

        if os.path.exists(accmf_image_path):
            accmf_image = np.load(accmf_image_path).astype(np.float32).reshape(nx, ny, nz)
            if self.normalization:
                accmf_image /= trans_power
            results["accmf_image"] = accmf_image

        return results


@TRANSFORMS.register_module()
class LoadAntennaPositions(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        radar_cfg = results["radar_cfg"]
        data_path = results["data_path"]
        r_pos = np.load(os.path.join(data_path, "rpos.npy")).astype(np.float32).reshape(-1, 3)
        t_pos = np.load(os.path.join(data_path, "tpos.npy")).astype(np.float32).reshape(-1, 3)

        radar_cfg["total_ant_size"] = r_pos.shape[0]
        results["r_pos"] = r_pos
        results["t_pos"] = t_pos
        return results


@TRANSFORMS.register_module()
class InterpolateMFAtTargets(BaseTransform):
    def __init__(self, mode: str = "bilinear", align_corners: bool = True) -> None:
        self.sample_mode = mode
        self.align_corners = align_corners

    def transform(self, results: dict) -> Optional[dict]:
        selected_points = results["tgt_sampled_poses_norm"].reshape(-1, 3).copy()
        if "mf_image" in results:
            results["mf_sampled_value"] = _sample_volume(
                results["mf_image"],
                selected_points,
                self.sample_mode,
                self.align_corners,
            )
        if "accmf_image" in results:
            results["accmf_sampled_value"] = _sample_volume(
                results["accmf_image"],
                selected_points,
                self.sample_mode,
                self.align_corners,
            )
        return results


@TRANSFORMS.register_module()
class LoadImageFromFile(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        image = mpimg.imread(results["img_path"])
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        results["img"] = image
        results["img_shape"] = image.shape
        return results


@TRANSFORMS.register_module()
class LoadIntrinsic(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        intrinsic_dict = results["intrinsic_dict"]
        orig_w = intrinsic_dict["w"]
        orig_h = intrinsic_dict["h"]
        fl_x = intrinsic_dict["fl_x"]
        fl_y = intrinsic_dict["fl_y"]
        cx = intrinsic_dict["cx"]
        cy = intrinsic_dict["cy"]

        if "img_shape" in results:
            new_h, new_w = results["img_shape"][:2]
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
        else:
            scale_x = 1.0
            scale_y = 1.0

        intrinsic = np.array(
            [
                [fl_x * scale_x, 0, cx * scale_x, 0],
                [0, fl_y * scale_y, cy * scale_y, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        results["intrinsic"] = intrinsic
        results["focal"] = fl_x
        return results


@TRANSFORMS.register_module()
class LoadRandomRays(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        num_rays = int(results["sample_cfg"]["n_ray"])
        img = results["img"]
        height, width = img.shape[:2]
        intrinsic = np.asarray(results["intrinsic"], dtype=np.float32)[:3, :3]
        pose = np.asarray(results["extrinsic"], dtype=np.float32)

        px = np.random.randint(0, width, size=(num_rays,), dtype=np.int64)
        py = np.random.randint(0, height, size=(num_rays,), dtype=np.int64)
        rgb = img[py, px]

        pixels = np.stack(
            [
                px.astype(np.float32) + 0.5,
                py.astype(np.float32) + 0.5,
                np.ones_like(px, dtype=np.float32),
            ],
            axis=-1,
        )
        dirs_cam = (np.linalg.inv(intrinsic) @ pixels.T).T
        dirs_cam = dirs_cam / np.maximum(np.linalg.norm(dirs_cam, axis=-1, keepdims=True), 1e-8)

        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        dirs_world = (rotation @ dirs_cam.T).T.astype(np.float32)
        ray_o = np.broadcast_to(translation, dirs_world.shape).copy().astype(np.float32)

        results["ray_o"] = ray_o
        results["ray_d"] = dirs_world
        results["color"] = rgb.astype(np.float32)
        results["xy"] = np.stack([px, py], axis=-1).astype(np.float32)
        return results


@TRANSFORMS.register_module()
class GetNearFarFromSphere(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        ray_o = results["ray_o"]
        ray_d = results["ray_d"]

        a = np.einsum("ij,ij->i", ray_d, ray_d)
        b = 2.0 * np.einsum("ij,ij->i", ray_o, ray_d)
        c = np.einsum("ij,ij->i", ray_o, ray_o) - 1.0
        disc = b * b - 4 * a * c
        safe = a > 1e-8
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))

        t0 = np.full_like(b, np.inf)
        t1 = np.full_like(b, np.inf)
        t0[safe] = (-b[safe] - sqrt_disc[safe]) / (2 * a[safe])
        t1[safe] = (-b[safe] + sqrt_disc[safe]) / (2 * a[safe])
        t_enter = np.minimum(t0, t1)
        t_exit = np.maximum(t0, t1)
        valid = (disc >= -1e-8) & safe & (t_exit >= 0)

        results["near"] = np.where(valid, np.maximum(t_enter, 0.0), np.nan).astype(np.float32)
        results["far"] = np.where(valid, t_exit, np.nan).astype(np.float32)
        results["hit_mask"] = valid
        results["inside_mask"] = (np.einsum("ij,ij->i", ray_o, ray_o) < 1.0 + 1e-8)
        return results


@TRANSFORMS.register_module()
class ExtractFields(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        sample_cfg = results["sample_cfg"]
        res = np.asarray(sample_cfg["sample_res"], dtype=np.float32)
        radius = sample_cfg["radius"]
        res_norm = res / radius

        half_side = 1.0 / np.sqrt(3.0)
        nx = grid_num(half_side, -half_side, res_norm[0])
        ny = grid_num(half_side, -half_side, res_norm[1])
        nz = grid_num(half_side, -half_side, res_norm[2])
        x_lin = np.linspace(-half_side, half_side, nx)
        y_lin = np.linspace(-half_side, half_side, ny)
        z_lin = np.linspace(-half_side, half_side, nz)
        sample_cfg["shape"] = (nx, ny, nz)

        x_grid, y_grid, z_grid = np.meshgrid(x_lin, y_lin, z_lin, indexing="ij")
        sampled_poses_norm = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).astype(np.float32)
        results["sampled_poses_norm"] = sampled_poses_norm

        normglb2glb = results["normglb2glb"]
        h_pts = np.concatenate([sampled_poses_norm, np.ones_like(sampled_poses_norm[..., :1])], axis=-1)
        h_pts = h_pts @ normglb2glb.T
        results["sampled_poses_glb"] = h_pts[..., :3] / h_pts[..., 3:4]
        return results
