# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

from typing import Optional

import numpy as np

from .base_transform import BaseTransform
from geraf.utils.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SceneToUnitSphere(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        bound = results["sample_cfg"]["bound"]
        radius = results["sample_cfg"]["radius"]

        x_c = (bound[0] + bound[1]) / 2.0
        y_c = (bound[2] + bound[3]) / 2.0
        z_c = (bound[4] + bound[5]) / 2.0
        scale = 1.0 / radius

        results["glb2normglb"] = np.array(
            [
                [scale, 0, 0, -scale * x_c],
                [0, scale, 0, -scale * y_c],
                [0, 0, scale, -scale * z_c],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        results["normglb2glb"] = np.array(
            [
                [radius, 0, 0, x_c],
                [0, radius, 0, y_c],
                [0, 0, radius, z_c],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        results["normglb_scale"] = scale
        return results


@TRANSFORMS.register_module()
class AntennasToUnitSphere(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        t_pos = results["t_pos"].copy()
        r_pos = results["r_pos"].copy()
        glb2normglb = results["glb2normglb"]

        t_pos_norm = np.concatenate([t_pos, np.ones_like(t_pos[:, :1])], axis=1) @ glb2normglb.T
        r_pos_norm = np.concatenate([r_pos, np.ones_like(r_pos[:, :1])], axis=1) @ glb2normglb.T

        results["t_pos_norm"] = t_pos_norm[:, :3] / t_pos_norm[:, 3:4]
        results["r_pos_norm"] = r_pos_norm[:, :3] / r_pos_norm[:, 3:4]
        return results


@TRANSFORMS.register_module()
class RGBTo01(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        image = results["img"]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)
            if image.max() > 1.0:
                image /= 255.0
        results["img"] = image
        return results


@TRANSFORMS.register_module()
class RGBSceneToUnitSphere(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        sample_cfg = results["sample_cfg"]
        bound = sample_cfg["bound"]
        radius = sample_cfg["radius"]

        x_c = (bound[0] + bound[1]) / 2.0
        y_c = (bound[2] + bound[3]) / 2.0
        z_c = (bound[4] + bound[5]) / 2.0
        scale = 1.0 / radius

        results["glb2normglb"] = np.array(
            [
                [scale, 0, 0, -scale * x_c],
                [0, scale, 0, -scale * y_c],
                [0, 0, scale, -scale * z_c],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        results["normglb2glb"] = np.array(
            [
                [radius, 0, 0, x_c],
                [0, radius, 0, y_c],
                [0, 0, radius, z_c],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        results["normglb_scale"] = scale
        return results


@TRANSFORMS.register_module()
class CameraToUnitSphere(BaseTransform):
    def transform(self, results: dict) -> Optional[dict]:
        results["org_extrinsic"] = results["extrinsic"].copy()
        results["extrinsic"] = results["glb2normglb"] @ results["extrinsic"]
        return results
