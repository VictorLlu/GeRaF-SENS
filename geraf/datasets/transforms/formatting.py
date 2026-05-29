# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import numpy as np

from .base_transform import BaseTransform
from .common import to_tensor
from geraf.utils.registry import TRANSFORMS


DEFAULT_RF_META_KEYS = (
    "sampled_poses_glb",
    "sampled_poses_norm",
    "tgt_sampled_poses_norm",
    "tgt_sampled_poses_glb",
    "z_vals",
    "tgt_z_vals",
    "tgt_dists",
    "ray_d",
    "dists",
    "mf_sampled_value",
    "t_pos",
    "r_pos",
    "t_pos_norm",
    "r_pos_norm",
    "ant_dir",
    "radar_cfg",
    "loss_mask",
    "near",
    "far",
    "sample_cfg",
    "data_name",
    "glb2normglb",
    "normglb2glb",
    "normglb_scale",
)


def _pack_into_metainfo(results, meta_keys, tensor_meta_keys):
    img_meta = {}
    for key in meta_keys:
        if key not in results:
            continue
        value = results[key]
        if key in tensor_meta_keys:
            if not value.flags.c_contiguous:
                value = np.ascontiguousarray(value)
            if value.dtype == np.float64:
                value = value.astype(np.float32)
            value = to_tensor(value).contiguous()
        img_meta[key] = value
    return img_meta


@TRANSFORMS.register_module()
class PackRFInputs(BaseTransform):
    tensor_meta_keys = {
        "sampled_poses_norm",
        "sampled_poses_glb",
        "tgt_sampled_poses_norm",
        "tgt_sampled_poses_glb",
        "sampled_normals",
        "mf_sampled_value",
        "z_vals",
        "tgt_z_vals",
        "tgt_dists",
        "ray_o",
        "ray_d",
        "dists",
        "glb2normglb",
        "normglb2glb",
        "t_pos",
        "r_pos",
        "r_pos_norm",
        "t_pos_norm",
        "ant_dir",
        "loss_mask",
        "sph_sampled_value",
        "inner_mask",
        "tgt_inner_mask",
        "near",
        "far",
    }

    def __init__(self, meta_keys=DEFAULT_RF_META_KEYS):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        return _pack_into_metainfo(results, self.meta_keys, self.tensor_meta_keys)


@TRANSFORMS.register_module()
class PackRGBInputs(BaseTransform):
    tensor_meta_keys = {"img", "color", "ray_o", "ray_d", "near", "far", "depth_mask", "hit_mask", "depth", "xy", "extrinsic", "intrinsic"}

    def __init__(self, meta_keys=("color", "ray_o", "ray_d", "near", "far")):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        return _pack_into_metainfo(results, self.meta_keys, self.tensor_meta_keys)


@TRANSFORMS.register_module()
class PackMeshInputs(BaseTransform):
    tensor_meta_keys = {"sampled_poses_norm"}

    def __init__(self, meta_keys=("sampled_poses_norm", "sample_cfg", "normglb_scale")):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        return _pack_into_metainfo(results, self.meta_keys, self.tensor_meta_keys)
