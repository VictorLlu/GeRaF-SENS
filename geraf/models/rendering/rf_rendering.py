# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from geraf.models.model_utils.matched_filter import base_matched_filter
from geraf.models.model_utils.ray_sampler import bilinear_ray_sampler
from geraf.models.model_utils.ray_tracer import lensless_ray_tracer
from geraf.utils import ConfigType, OptConfigType, OptMultiConfig
from geraf.utils.registry import MODELS

from .base import BaseRendering


EPS = 1e-5
TRANS_EPS = 1e-7
INF_CLIP_MIN = 1e-6
INF_CLIP_MAX = 1e6
SPHERE_RADIUS = 1.0


def _load_checkpoint_state_dict(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """Load model weights from either repo-native or MMEngine-style checkpoints."""
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}")
    return checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint


def _log_message(message: str) -> None:
    timestamp = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"{timestamp} - {message}")


def _compute_transmission(alpha: torch.Tensor) -> torch.Tensor:
    """Return cumulative transmission probabilities used by volume rendering."""
    leading_ones = torch.ones([alpha.shape[0], 1]).to(alpha)
    return torch.cumprod(
        torch.cat([leading_ones, 1.0 - alpha + TRANS_EPS], dim=-1),
        dim=-1,
    )[..., :-1]


def _reshape_masked(mask: torch.Tensor, tensor: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """Apply a boolean mask and reshape to the expected downstream shape."""
    return tensor[mask].reshape(*shape)


def _detach_cache_to_cpu(cache_dict: MutableMapping[Any, List[torch.Tensor]]) -> None:
    """Move cached antenna tensors to CPU in-place for checkpoint compatibility."""
    for key, cache_items in cache_dict.items():
        for index, item in enumerate(cache_items):
            cache_items[index] = item.detach().cpu()


class GeRaFStageBase(BaseRendering, metaclass=ABCMeta):
    """Shared base for GeRaF stage renderers.

    The implementation intentionally preserves the existing rendering math and
    training behavior. The refactor here focuses on readability and grouping
    repeated steps into small helpers.
    """

    def __init__(
        self,
        sdf_network: ConfigType,
        deviation_network: ConfigType,
        signal_network: ConfigType,
        data_preprocessor: OptConfigType = dict,
        rt_backend: str = "cuda",
        mf_backend: str = "cuda",
        with_sigma_loss: float = None,
        with_grad_regression: float = None,
        sign_penalty_weight: float = None,
        use_diff: bool = True,
        test_mf: bool = False,
        pretrain: bool = True,
        freeze_sigma: bool = True,
        sdf_scale_cfg: Dict[str, float] = dict(inside_min=0.05, outside_max=0.25),
        loss_mode: str = "sum",
        mf_loss: str = "mf_loss",
        norm_sdf: bool = True,
        loss_scale: float = 1.0,
        up_sample: bool = True,
        sample_cfg=None,
        radar_cfg=None,
        predict_chunk: int = 1024,
        init_cfg: OptMultiConfig = dict,
    ) -> None:
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.rt_backend = rt_backend.lower()
        self.mf_backend = mf_backend.lower()

        self.sdf_network = MODELS.build(sdf_network)
        self.deviation_network = MODELS.build(deviation_network)
        self.signal_network = MODELS.build(signal_network)
        self.sdf_inter_fun = lambda x: self.sdf_network.sdf(x)

        self.with_sigma_loss = with_sigma_loss
        self.use_diff = use_diff
        self.test_mf = test_mf
        self.with_grad_regression = with_grad_regression
        self.pretrain = pretrain
        self.loss_scale = loss_scale
        self.inside_min = sdf_scale_cfg["inside_min"]
        self.outside_max = sdf_scale_cfg["outside_max"]
        self.up_sample = up_sample
        self.loss_mode = loss_mode
        self.mf_loss = mf_loss
        self.norm_sdf = norm_sdf
        self.cfg = sample_cfg
        self.radar_cfg = radar_cfg
        self.step = 0
        self.predict_chunk = predict_chunk

        self.ant_real_list: Dict[str, List[torch.Tensor]] = {}
        self.ant_imag_list: Dict[str, List[torch.Tensor]] = {}
        self.ant_chunk_id: Dict[str, int] = {}

    def _forward(self, data_sample: dict) -> torch.Tensor:
        raise NotImplementedError("Tensor mode is not used by GeRaF stage runners.")

    def update_step(self, step: int) -> None:
        self.step = step

    def get_anneal_val(self, step: int) -> float:
        if self.cfg["anneal_end"] < 0:
            return 1.0
        return np.min([1.0, step / self.cfg["anneal_end"]])

    def _clip_inverse_s(self, values: torch.Tensor) -> torch.Tensor:
        return values.clip(INF_CLIP_MIN, INF_CLIP_MAX)

    def _compute_iter_cos(
        self,
        gradients: torch.Tensor,
        directions: torch.Tensor,
        cos_anneal_ratio: float,
    ) -> torch.Tensor:
        true_cos = (directions * gradients).sum(-1)
        return -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )

    def _compute_alpha_from_sdf(
        self,
        sdf: torch.Tensor,
        inv_s: torch.Tensor,
        dists: torch.Tensor,
        iter_cos: torch.Tensor,
    ) -> torch.Tensor:
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        probability = prev_cdf - next_cdf
        return ((probability + EPS) / (prev_cdf + EPS)).clip(0.0, 1.0)

    def compute_sdf_alpha(
        self,
        points: torch.Tensor,
        dists: torch.Tensor,
        dirs: torch.Tensor,
        cos_anneal_ratio: float,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sdf_nn_output = self.sdf_network(points)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]
        gradients = self.sdf_network.gradient(points)

        inv_s = self._clip_inverse_s(self.deviation_network(points))[..., 0]
        if self.cfg["freeze_inv_s_step"] is not None and step < self.cfg["freeze_inv_s_step"]:
            inv_s = inv_s.detach()

        iter_cos = self._compute_iter_cos(gradients, dirs, cos_anneal_ratio)
        alpha = self._compute_alpha_from_sdf(sdf, inv_s, dists, iter_cos)
        return alpha, gradients, feature_vector, inv_s, sdf

    def compute_cdf(self, points: torch.Tensor) -> torch.Tensor:
        sdf = self.sdf_network(points)[..., 0]
        inv_s = self._clip_inverse_s(self.deviation_network(points))[..., 0]
        return torch.sigmoid(sdf * inv_s)

    def compute_mf_loss(self, rgb_pr: torch.Tensor, rgb_gt: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "l2":
            return torch.sum((rgb_pr - rgb_gt) ** 2, dim=-1)
        if self.loss_mode == "l1":
            return torch.sum(F.l1_loss(rgb_pr, rgb_gt, reduction="none"), dim=-1)
        if self.loss_mode == "smooth_l1":
            return torch.sum(
                F.smooth_l1_loss(rgb_pr, rgb_gt, reduction="none", beta=0.25),
                dim=-1,
            )
        if self.loss_mode == "charbonier":
            epsilon = 0.001
            return torch.sqrt(torch.sum((rgb_gt - rgb_pr) ** 2, dim=-1) + epsilon)
        raise NotImplementedError

    def state_dict(
        self,
        destination=None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        state[prefix + "ant_real_list"] = self.ant_real_list
        state[prefix + "ant_imag_list"] = self.ant_imag_list
        state[prefix + "ant_chunk_id"] = self.ant_chunk_id
        return state

    def _load_cache_from_state(
        self,
        state_dict: MutableMapping[str, Any],
        key: str,
        strict: bool,
        missing_keys: List[str],
    ) -> Any:
        if key in state_dict:
            return state_dict.pop(key)
        if strict:
            missing_keys.append(key)
        return None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        ant_real_key = prefix + "ant_real_list"
        ant_imag_key = prefix + "ant_imag_list"
        ant_chunk_key = prefix + "ant_chunk_id"

        ant_real_list = self._load_cache_from_state(state_dict, ant_real_key, strict, missing_keys)
        if ant_real_list is not None:
            self.ant_real_list = ant_real_list
            _detach_cache_to_cpu(self.ant_real_list)

        ant_imag_list = self._load_cache_from_state(state_dict, ant_imag_key, strict, missing_keys)
        if ant_imag_list is not None:
            self.ant_imag_list = ant_imag_list
            _detach_cache_to_cpu(self.ant_imag_list)

        ant_chunk_id = self._load_cache_from_state(state_dict, ant_chunk_key, strict, missing_keys)
        if ant_chunk_id is not None:
            self.ant_chunk_id = ant_chunk_id

        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def generate_series(self, num_antennas: int, num_groups: int) -> List[torch.Tensor]:
        return [torch.arange(index, num_antennas, step=num_groups) for index in range(num_groups)]

    def _prepare_predict_grid(
        self,
        total_poses: torch.Tensor,
        data_sample: dict,
    ) -> Tuple[torch.Tensor, int, int, int, torch.Tensor]:
        nx, ny, nz = data_sample["sample_cfg"]["shape"]
        total_poses = total_poses.reshape(nx, ny, nz, 3).permute(2, 1, 0, 3).flatten(0, 1)

        half_side = 1.0 / np.sqrt(3.0)
        near, far = -half_side, half_side
        z_vals = torch.linspace(0.0, 1.0, nx).to(total_poses)
        z_vals = near + (far - near) * z_vals[None, :]
        return total_poses, nx, ny, nz, z_vals

    def _predict_chunk_outputs(
        self,
        chunk_poses: torch.Tensor,
        z_vals: torch.Tensor,
        normglb_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            num_chunk_poses = chunk_poses.shape[0]
            chunk_z_vals = z_vals.repeat(num_chunk_poses, 1)
            chunk_dists = chunk_z_vals[..., 1:] - chunk_z_vals[..., :-1]
            chunk_dists = torch.cat([chunk_dists, chunk_dists[..., -1:]], dim=-1)

            chunk_dirs = torch.zeros_like(chunk_poses)
            chunk_dirs[..., 0] = 1.0

            sdf_nn_output = self.sdf_network(chunk_poses)
            sdf = self.sdf_network.sdf(chunk_poses)[..., 0]
            feature_vector = sdf_nn_output[..., 1:]
            gradients = self.sdf_network.gradient(chunk_poses)
            normals = F.normalize(gradients, dim=-1)
            inv_s = self._clip_inverse_s(self.deviation_network(chunk_poses))[..., 0]

            true_cos = (chunk_dirs * gradients).sum(-1)
            iter_cos = -F.relu(-true_cos)
            alpha = self._compute_alpha_from_sdf(sdf, inv_s, chunk_dists, iter_cos)

            trans_prob = _compute_transmission(alpha)
            weights = alpha * trans_prob
            chunk_occ = self.signal_network(chunk_poses, feature_vector, trans_prob)
            chunk_ref = self.signal_network.get_refelectivness(chunk_poses, feature_vector)
            chunk_sigmas = chunk_occ * weights
            sdf /= normglb_scale

        return (
            sdf.squeeze(-1).detach().cpu().numpy(),
            chunk_sigmas.squeeze(-1).detach().cpu().numpy(),
            weights.squeeze(-1).detach().cpu().numpy(),
            normals.detach().cpu().numpy(),
            chunk_ref.detach().cpu().numpy(),
        )

    def predict(self, data_sample: dict) -> Dict[str, np.ndarray]:
        total_poses, nx, ny, nz, z_vals = self._prepare_predict_grid(data_sample["sampled_poses_norm"], data_sample)

        num_apr = total_poses.shape[0]
        num_chunk_points = self.predict_chunk
        normglb_scale = data_sample["normglb_scale"]
        num_iters = num_apr // num_chunk_points + (1 if num_apr % num_chunk_points != 0 else 0)

        all_pred_sdfs = []
        all_pred_sigmas = []
        all_normal_vecs = []
        all_pred_weights = []
        all_pred_refs = []

        for iteration in tqdm(range(num_iters)):
            start_idx = iteration * num_chunk_points
            end_idx = min((iteration + 1) * num_chunk_points, num_apr)
            chunk_poses = total_poses[start_idx:end_idx, :, :]

            (
                chunk_sdf,
                chunk_sigmas,
                chunk_weights,
                chunk_normals,
                chunk_refs,
            ) = self._predict_chunk_outputs(chunk_poses, z_vals, normglb_scale)

            all_pred_sdfs.append(chunk_sdf)
            all_pred_sigmas.append(chunk_sigmas)
            all_pred_weights.append(chunk_weights)
            all_normal_vecs.append(chunk_normals)
            all_pred_refs.append(chunk_refs)

        pred_sdfs = np.concatenate(all_pred_sdfs, axis=0).reshape(nz, ny, nx).transpose(2, 1, 0)
        pred_sigmas = np.concatenate(all_pred_sigmas, axis=0).reshape(nz, ny, nx).transpose(2, 1, 0)
        pred_weights = np.concatenate(all_pred_weights, axis=0).reshape(nz, ny, nx).transpose(2, 1, 0)
        pred_normals = (
            np.concatenate(all_normal_vecs, axis=0)
            .reshape(nz, ny, nx, 3)
            .transpose(2, 1, 0, 3)
        )
        pred_refs = np.concatenate(all_pred_refs, axis=0).reshape(nz, ny, nx).transpose(2, 1, 0)

        return dict(
            sdfs=pred_sdfs,
            sigmas=pred_sigmas,
            weights=pred_weights,
            normals=pred_normals,
            refs=pred_refs,
        )

    @abstractmethod
    def loss(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


@MODELS.register_module()
class GeRaFStage1(GeRaFStageBase):
    """GeRaF stage-1 renderer."""

    def __init__(
        self,
        sdf_network: ConfigType,
        deviation_network: ConfigType,
        signal_network: ConfigType,
        data_preprocessor: OptConfigType = dict,
        rt_backend: str = "cuda",
        mf_backend: str = "cuda",
        with_sigma_loss: float = None,
        with_grad_regression: float = None,
        sign_penalty_weight: float = None,
        use_diff: bool = True,
        test_mf: bool = False,
        pretrain: bool = True,
        freeze_sigma: bool = True,
        sdf_scale_cfg: Dict[str, float] = dict(inside_min=0.05, outside_max=0.25),
        loss_mode: str = "sum",
        mf_loss: str = "mf_loss",
        norm_sdf: bool = True,
        loss_scale: float = 1.0,
        up_sample: bool = True,
        sample_cfg=None,
        radar_cfg=None,
        predict_chunk: int = 1024,
        fix_sdf: bool = False,
        fix_power: bool = False,
        init_cfg: OptMultiConfig = dict,
    ) -> None:
        super().__init__(
            sdf_network=sdf_network,
            deviation_network=deviation_network,
            signal_network=signal_network,
            data_preprocessor=data_preprocessor,
            rt_backend=rt_backend,
            mf_backend=mf_backend,
            with_sigma_loss=with_sigma_loss,
            with_grad_regression=with_grad_regression,
            sign_penalty_weight=sign_penalty_weight,
            use_diff=use_diff,
            test_mf=test_mf,
            pretrain=pretrain,
            freeze_sigma=freeze_sigma,
            sdf_scale_cfg=sdf_scale_cfg,
            loss_mode=loss_mode,
            mf_loss=mf_loss,
            norm_sdf=norm_sdf,
            loss_scale=loss_scale,
            up_sample=up_sample,
            sample_cfg=sample_cfg,
            radar_cfg=radar_cfg,
            predict_chunk=predict_chunk,
            init_cfg=init_cfg,
        )
        self._freeze_requested_parameters(fix_sdf=fix_sdf, fix_power=fix_power)

    def _freeze_requested_parameters(self, fix_sdf: bool, fix_power: bool) -> None:
        if fix_sdf:
            for parameter in self.sdf_network.parameters():
                parameter.requires_grad = False
        if fix_power:
            self.signal_network.light_power.requires_grad = False

    def intersect_sphere(
        self,
        r_pos_norm: torch.Tensor,
        norm_pts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        center = torch.mean(r_pos_norm, dim=0, keepdim=True)
        n_ant = center.shape[0]
        n_apr, n_sample, _ = norm_pts.shape

        ant_pts_v = norm_pts[None, :, :, :] - center[:, None, None, :]
        ant_pts_dir = ant_pts_v / (
            torch.linalg.norm(ant_pts_v, dim=-1).unsqueeze(-1) + 1e-8
        )

        center_expanded = (
            center.reshape(n_ant, 1, 1, 3)
            .expand(-1, n_apr, n_sample, -1)
            .reshape(-1, 3)
        )
        ant_pts_dir = ant_pts_dir.reshape(-1, 3)

        b_val = 2.0 * torch.sum(center_expanded * ant_pts_dir, dim=-1)
        c_val = torch.sum(center_expanded * center_expanded, dim=-1) - SPHERE_RADIUS**2
        discriminant = b_val**2 - 4 * c_val

        valid = discriminant >= 0
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
        t1 = (-b_val - sqrt_disc) / 2
        t2 = (-b_val + sqrt_disc) / 2
        t_val = torch.where(
            t1 > 1e-5,
            t1,
            torch.where(t2 > 1e-5, t2, torch.full_like(t1, float("inf"))),
        )

        final_valid = valid & (t_val != float("inf"))
        intersect_pts = center_expanded + t_val.unsqueeze(-1) * ant_pts_dir
        return (
            final_valid.reshape(n_ant, n_apr, n_sample),
            intersect_pts.reshape(n_ant, n_apr, n_sample, 3),
        )

    def _extract_batch_inputs(self, data_sample: dict) -> Dict[str, Any]:
        sampled_poses_norm = data_sample["sampled_poses_norm"]
        n_apr, n_sample, _ = sampled_poses_norm.shape
        ray_d = data_sample["ray_d"]
        tgt_sampled_poses_norm = data_sample["tgt_sampled_poses_norm"]
        _, tgt_n_sample, _ = tgt_sampled_poses_norm.shape

        return {
            "data_name": data_sample["data_name"],
            "sampled_poses_norm": sampled_poses_norm,
            "t_pos": data_sample["t_pos"],
            "r_pos": data_sample["r_pos"],
            "sampled_poses_glb": data_sample["sampled_poses_glb"],
            "ray_d": ray_d,
            "z_vals": data_sample["z_vals"],
            "dists": data_sample["dists"],
            "dirs": ray_d.reshape(n_apr, 1, 3).expand(-1, n_sample, -1),
            "tgt_sampled_poses_norm": tgt_sampled_poses_norm,
            "gt_sar_value": data_sample["mf_sampled_value"].reshape(n_apr, tgt_n_sample),
            "tgt_dirs": ray_d.reshape(n_apr, 1, 3).expand(-1, tgt_n_sample, -1),
            "tgt_sampled_poses_glb": data_sample["tgt_sampled_poses_glb"],
            "tgt_z_vals": data_sample["tgt_z_vals"],
            "tgt_dists": data_sample["tgt_dists"],
            "r_pos_norm": data_sample["r_pos_norm"],
        }

    def _apply_loss_mask(
        self,
        batch_inputs: Dict[str, Any],
        loss_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        sampled_poses_norm = batch_inputs["sampled_poses_norm"]
        tgt_sampled_poses_norm = batch_inputs["tgt_sampled_poses_norm"]
        n_apr = int(loss_mask.sum())
        n_sample = sampled_poses_norm.shape[1]
        tgt_n_sample = tgt_sampled_poses_norm.shape[1]

        batch_inputs["n_apr"] = n_apr
        batch_inputs["n_pts"] = n_apr * n_sample
        batch_inputs["tgt_n_pts"] = n_apr * tgt_n_sample
        batch_inputs["n_ant"] = batch_inputs["t_pos"].shape[0]

        batch_inputs["sampled_poses_glb"] = _reshape_masked(
            loss_mask,
            batch_inputs["sampled_poses_glb"],
            (batch_inputs["n_pts"], 3),
        )
        batch_inputs["sampled_poses_norm"] = batch_inputs["sampled_poses_norm"][loss_mask, :, :]
        batch_inputs["dirs"] = batch_inputs["dirs"][loss_mask, :, :]
        batch_inputs["dists"] = batch_inputs["dists"][loss_mask, :]
        batch_inputs["z_vals"] = batch_inputs["z_vals"][loss_mask, :]
        batch_inputs["gt_sar_value"] = batch_inputs["gt_sar_value"][loss_mask, :]
        batch_inputs["tgt_z_vals"] = batch_inputs["tgt_z_vals"][loss_mask, :]
        batch_inputs["tgt_dirs"] = batch_inputs["tgt_dirs"][loss_mask, :, :]
        batch_inputs["tgt_dists"] = batch_inputs["tgt_dists"][loss_mask]
        batch_inputs["tgt_sampled_poses_norm"] = batch_inputs["tgt_sampled_poses_norm"][loss_mask, :, :]
        batch_inputs["tgt_sampled_poses_glb"] = batch_inputs["tgt_sampled_poses_glb"][loss_mask, :, :]
        return batch_inputs

    def _ensure_cached_adc(
        self,
        batch_inputs: Dict[str, Any],
        ant_indices_groups: Sequence[torch.Tensor],
    ) -> None:
        data_name = batch_inputs["data_name"]
        if (data_name in self.ant_real_list) and (data_name in self.ant_imag_list):
            return

        self.ant_real_list[data_name] = []
        self.ant_imag_list[data_name] = []

        for ant_id in tqdm(range(self.cfg["bank_size"]), desc=data_name):
            ant_indices = ant_indices_groups[ant_id]
            t_pos_sel = batch_inputs["t_pos"][ant_indices]
            r_pos_sel = batch_inputs["r_pos"][ant_indices]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                alpha_sel, gradients_sel, feature_vector_sel, _, _ = self.compute_sdf_alpha(
                    batch_inputs["tgt_sampled_poses_norm"],
                    batch_inputs["tgt_dists"],
                    batch_inputs["tgt_dirs"],
                    self.get_anneal_val(self.step),
                    self.step,
                )

            trans_prob_sel = _compute_transmission(alpha_sel)
            weights_sel = alpha_sel * trans_prob_sel
            sigmas_sel = (
                self.signal_network(
                    batch_inputs["tgt_sampled_poses_norm"],
                    feature_vector_sel,
                    trans_prob_sel,
                )
                * weights_sel
            )

            n_chunk_ant = ant_indices.shape[0]
            normals_sel = F.normalize(gradients_sel, dim=-1)
            sim_real_sel, sim_imag_sel = lensless_ray_tracer(
                normals=normals_sel.reshape(batch_inputs["tgt_n_pts"], 3),
                sigmas=sigmas_sel.reshape(1, batch_inputs["tgt_n_pts"]).repeat(n_chunk_ant, 1),
                points=batch_inputs["tgt_sampled_poses_glb"].reshape(batch_inputs["tgt_n_pts"], 3),
                antenna_t=t_pos_sel,
                antenna_r=r_pos_sel,
                radar_cfg=self.radar_cfg,
                backend=self.rt_backend,
                use_diff=self.use_diff,
                nomralization=True,
            )
            self.ant_real_list[data_name].append(sim_real_sel.detach().cpu())
            self.ant_imag_list[data_name].append(sim_imag_sel.detach().cpu())

        self.ant_chunk_id[data_name] = 0

    def _calibrate_transmission(
        self,
        sampled_poses_norm: torch.Tensor,
        selected_indices: torch.Tensor,
        r_pos_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dyntrans_mask, intersect_pts = self.intersect_sphere(r_pos_norm[selected_indices], sampled_poses_norm)

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                calib_cdf = self.compute_cdf(intersect_pts)
                prev_cdf = self.compute_cdf(sampled_poses_norm[:, 0:1, :])
                prev_cdf_expanded = prev_cdf[None, :, 0, None]
                new_cdf = torch.where(dyntrans_mask, calib_cdf, prev_cdf_expanded)

        return prev_cdf, new_cdf

    def _sample_tgt_normals_and_sigmas(
        self,
        gradients: torch.Tensor,
        sigmas: torch.Tensor,
        z_vals: torch.Tensor,
        tgt_z_vals: torch.Tensor,
        n_chunk_ant: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normals = F.normalize(gradients, dim=-1)
        tgt_normals, tgt_sigmas, inbounds = bilinear_ray_sampler(
            normals=normals,
            sigmas=sigmas,
            z_vals=z_vals,
            tgt_z_vals=tgt_z_vals,
        )
        tgt_normals = F.normalize(tgt_normals, dim=-1)
        tgt_sigmas = tgt_sigmas.repeat(n_chunk_ant, 1, 1)
        return tgt_normals, tgt_sigmas, inbounds

    def _compute_grad_regression_loss(self, gradients: torch.Tensor) -> torch.Tensor:
        """Compute the eikonal-style gradient regularizer used by the RF stages."""
        gradient_error = (torch.linalg.norm(gradients.float(), ord=2, dim=-1) - 1.0) ** 2
        return torch.mean(gradient_error.reshape(-1))

    def _gather_unselected_adc(
        self,
        data_name: str,
        ant_indices_groups: Sequence[torch.Tensor],
        current_chunk_id: int,
        t_pos: torch.Tensor,
        r_pos: torch.Tensor,
        sim_real: torch.Tensor,
        sim_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_pos_unselected = []
        r_pos_unselected = []
        adc_r_unselected = []
        adc_i_unselected = []

        for idx in range(self.cfg["bank_size"]):
            if idx == current_chunk_id:
                self.ant_real_list[data_name][idx] = sim_real.detach().cpu()
                self.ant_imag_list[data_name][idx] = sim_imag.detach().cpu()
                continue

            unsel_indices = ant_indices_groups[idx]
            t_pos_unselected.append(t_pos[unsel_indices])
            r_pos_unselected.append(r_pos[unsel_indices])
            adc_r_unselected.append(self.ant_real_list[data_name][idx])
            adc_i_unselected.append(self.ant_imag_list[data_name][idx])

        return (
            torch.cat(t_pos_unselected, dim=0),
            torch.cat(r_pos_unselected, dim=0),
            torch.cat(adc_r_unselected, dim=0).to(sim_real),
            torch.cat(adc_i_unselected, dim=0).to(sim_imag),
        )

    def _matched_filter_prediction(
        self,
        mix_real: torch.Tensor,
        mix_imag: torch.Tensor,
        mix_t_pos: torch.Tensor,
        mix_r_pos: torch.Tensor,
        tgt_sampled_poses_glb: torch.Tensor,
        gt_sar_value: torch.Tensor,
        inbounds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inb_tgt_sampled_poses_glb = tgt_sampled_poses_glb[inbounds, :]
        gt_sar_value = gt_sar_value[inbounds]
        n_inb_pts = gt_sar_value.shape[0]

        sar_real, sar_imag = base_matched_filter(
            real=mix_real,
            image=mix_imag,
            t_pos=mix_t_pos,
            r_pos=mix_r_pos,
            points=inb_tgt_sampled_poses_glb,
            radar_cfg=self.radar_cfg,
            backend=self.mf_backend,
        )
        pred_sar_value = torch.hypot(sar_real, sar_imag).reshape(n_inb_pts, 1)

        if "sub_adc" in self.radar_cfg:
            gt_sar_value = gt_sar_value.reshape(n_inb_pts, 1) / self.radar_cfg["sub_adc"]
        else:
            gt_sar_value = gt_sar_value.reshape(n_inb_pts, 1)

        return pred_sar_value, gt_sar_value

    def _update_chunk_pointer(self, data_name: str) -> None:
        self.ant_chunk_id[data_name] = (self.ant_chunk_id[data_name] + 1) % self.cfg["bank_size"]

    def loss(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        frame = self._extract_batch_inputs(data_sample)
        frame = self._apply_loss_mask(frame, data_sample["loss_mask"])

        ant_indices_groups = self.generate_series(frame["n_ant"], self.cfg["bank_size"])
        self._ensure_cached_adc(frame, ant_indices_groups)

        current_chunk_id = self.ant_chunk_id[frame["data_name"]]
        selected_indices = ant_indices_groups[current_chunk_id]
        n_chunk_ant = selected_indices.shape[0]
        t_pos_selected = frame["t_pos"][selected_indices]
        r_pos_selected = frame["r_pos"][selected_indices]

        prev_cdf, new_cdf = self._calibrate_transmission(
            frame["sampled_poses_norm"],
            selected_indices,
            frame["r_pos_norm"],
        )

        with torch.amp.autocast("cuda", dtype=torch.float16):
            alpha, gradients, feature_vector, _, _ = self.compute_sdf_alpha(
                frame["sampled_poses_norm"],
                frame["dists"],
                frame["dirs"],
                self.get_anneal_val(self.step),
                self.step,
            )

        trans_prob = _compute_transmission(alpha)
        trans_prob_calib = (trans_prob - prev_cdf) + new_cdf
        weights = alpha * trans_prob_calib
        sigmas = (
            self.signal_network(
                frame["sampled_poses_norm"],
                feature_vector,
                trans_prob_calib,
            )
            * weights
        )

        tgt_normals, tgt_sigmas, inbounds = self._sample_tgt_normals_and_sigmas(
            gradients,
            sigmas,
            frame["z_vals"],
            frame["tgt_z_vals"],
            n_chunk_ant,
        )

        sim_real, sim_imag = lensless_ray_tracer(
            normals=tgt_normals.reshape(frame["tgt_n_pts"], 3),
            sigmas=tgt_sigmas.reshape(n_chunk_ant, frame["tgt_n_pts"]),
            points=frame["tgt_sampled_poses_glb"].reshape(frame["tgt_n_pts"], 3),
            antenna_t=t_pos_selected,
            antenna_r=r_pos_selected,
            inbounds=inbounds.reshape(frame["tgt_n_pts"]),
            radar_cfg=self.radar_cfg,
            backend=self.rt_backend,
            use_diff=self.use_diff,
            nomralization=True,
        )

        (
            t_pos_unselected,
            r_pos_unselected,
            adc_r_unselected,
            adc_i_unselected,
        ) = self._gather_unselected_adc(
            frame["data_name"],
            ant_indices_groups,
            current_chunk_id,
            frame["t_pos"],
            frame["r_pos"],
            sim_real,
            sim_imag,
        )

        mix_real = torch.cat([sim_real, adc_r_unselected], dim=0)
        mix_imag = torch.cat([sim_imag, adc_i_unselected], dim=0)
        mix_t_pos = torch.cat([t_pos_selected, t_pos_unselected], dim=0)
        mix_r_pos = torch.cat([r_pos_selected, r_pos_unselected], dim=0)

        pred_sar_value, gt_sar_value = self._matched_filter_prediction(
            mix_real,
            mix_imag,
            mix_t_pos,
            mix_r_pos,
            frame["tgt_sampled_poses_glb"],
            frame["gt_sar_value"],
            inbounds,
        )

        self._update_chunk_pointer(frame["data_name"])

        losses: Dict[str, torch.Tensor] = {}
        losses["mf_loss"] = torch.mean(
            self.compute_mf_loss(pred_sar_value, gt_sar_value * self.loss_scale)
        )
        if self.with_grad_regression:
            losses["grad_loss"] = self._compute_grad_regression_loss(gradients) * self.with_grad_regression
        return losses


@MODELS.register_module()
class GeRaFStage2(GeRaFStage1):
    """GeRaF stage-2 renderer with depth supervision."""

    def __init__(
        self,
        sdf_network: ConfigType,
        deviation_network: ConfigType,
        signal_network: ConfigType,
        data_preprocessor: OptConfigType = dict,
        rt_backend: str = "cuda",
        mf_backend: str = "cuda",
        with_sigma_loss: float = None,
        with_grad_regression: float = None,
        with_depth_regression: float = None,
        sign_penalty_weight: float = None,
        use_diff: bool = True,
        test_mf: bool = False,
        pretrain: bool = True,
        freeze_sigma: bool = True,
        sdf_scale_cfg: Dict[str, float] = dict(inside_min=0.05, outside_max=0.25),
        loss_mode: str = "sum",
        mf_loss: str = "mf_loss",
        norm_sdf: bool = True,
        loss_scale: float = 1.0,
        up_sample: bool = True,
        sample_cfg=None,
        radar_cfg=None,
        predict_chunk: int = 1024,
        fix_sdf: bool = False,
        fix_power: bool = False,
        tgt_sdf_ckpt_path=None,
        init_cfg: OptMultiConfig = dict,
    ) -> None:
        super().__init__(
            sdf_network=sdf_network,
            deviation_network=deviation_network,
            signal_network=signal_network,
            data_preprocessor=data_preprocessor,
            rt_backend=rt_backend,
            mf_backend=mf_backend,
            with_sigma_loss=with_sigma_loss,
            with_grad_regression=with_grad_regression,
            sign_penalty_weight=sign_penalty_weight,
            use_diff=use_diff,
            test_mf=test_mf,
            pretrain=pretrain,
            freeze_sigma=freeze_sigma,
            sdf_scale_cfg=sdf_scale_cfg,
            loss_mode=loss_mode,
            mf_loss=mf_loss,
            norm_sdf=norm_sdf,
            loss_scale=loss_scale,
            up_sample=up_sample,
            sample_cfg=sample_cfg,
            radar_cfg=radar_cfg,
            predict_chunk=predict_chunk,
            fix_sdf=fix_sdf,
            fix_power=fix_power,
            init_cfg=init_cfg,
        )
        from geraf.utils import resolve_ckpt

        self.with_depth_regression = with_depth_regression
        self.tgt_sdf_network = MODELS.build(sdf_network)
        self.tgt_deviation_network = MODELS.build(deviation_network)
        self.tgt_sdf_ckpt_path = resolve_ckpt(tgt_sdf_ckpt_path) if tgt_sdf_ckpt_path else tgt_sdf_ckpt_path

        for parameter in self.tgt_sdf_network.parameters():
            parameter.requires_grad = False
        for parameter in self.tgt_deviation_network.parameters():
            parameter.requires_grad = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        _log_message(f"Loading target SDF weights from {self.tgt_sdf_ckpt_path}...")
        tgt_state_dict = _load_checkpoint_state_dict(self.tgt_sdf_ckpt_path, map_location="cpu")

        remapped_state_dict = {
            key.replace("sdf_network.", "tgt_sdf_network."): value
            for key, value in tgt_state_dict.items()
            if key.startswith("sdf_network.")
        }
        remapped_state_dict.update(
            {
                key.replace("deviation_network.", "tgt_deviation_network."): value
                for key, value in tgt_state_dict.items()
                if key.startswith("deviation_network.")
            }
        )

        target_state_dict = {}
        target_state_dict.update(
            {
                f"tgt_sdf_network.{key}": value
                for key, value in self.tgt_sdf_network.state_dict().items()
            }
        )
        target_state_dict.update(
            {
                f"tgt_deviation_network.{key}": value
                for key, value in self.tgt_deviation_network.state_dict().items()
            }
        )

        filtered_state_dict = {}
        missing_target_keys = []
        unexpected_target_keys = []
        skipped_mismatch = []

        for key, value in remapped_state_dict.items():
            target_value = target_state_dict.get(key)
            if target_value is None:
                unexpected_target_keys.append(key)
                continue
            if not isinstance(value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
                continue
            if value.shape != target_value.shape:
                skipped_mismatch.append((key, tuple(value.shape), tuple(target_value.shape)))
                continue
            filtered_state_dict[key] = value

        for key in target_state_dict:
            if key not in filtered_state_dict:
                missing_target_keys.append(key)

        if skipped_mismatch:
            _log_message(
                f"Skipped mismatched target-SDF keys when loading {self.tgt_sdf_ckpt_path}: {len(skipped_mismatch)}"
            )
            for key, src_shape, dst_shape in skipped_mismatch:
                _log_message(f"  {key}: checkpoint{src_shape} != model{dst_shape}")

        if missing_target_keys:
            _log_message(
                f"Missing target-SDF keys when loading {self.tgt_sdf_ckpt_path}: {len(missing_target_keys)}"
            )
            for key in missing_target_keys:
                _log_message(f"  {key}")

        if unexpected_target_keys:
            _log_message(
                f"Unexpected target-SDF keys when loading {self.tgt_sdf_ckpt_path}: {len(unexpected_target_keys)}"
            )
            for key in unexpected_target_keys:
                _log_message(f"  {key}")

        state_dict.update(filtered_state_dict)
        super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=strict,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )

    def compute_sdf_alpha(
        self,
        sdf_network,
        deviation_network,
        points: torch.Tensor,
        dists: torch.Tensor,
        dirs: torch.Tensor,
        cos_anneal_ratio: float,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sdf_nn_output = sdf_network(points)
        sdf = sdf_nn_output[..., 0]
        feature_vector = sdf_nn_output[..., 1:]
        gradients = sdf_network.gradient(points)

        inv_s = self._clip_inverse_s(deviation_network(points))[..., 0]
        if self.cfg["freeze_inv_s_step"] is not None and step < self.cfg["freeze_inv_s_step"]:
            inv_s = inv_s.detach()

        iter_cos = self._compute_iter_cos(gradients, dirs, cos_anneal_ratio)
        alpha = self._compute_alpha_from_sdf(sdf, inv_s, dists, iter_cos)
        return alpha, gradients, feature_vector, inv_s, sdf

    def compute_dpt_loss(self, dpt: torch.Tensor, tgt_dpt: torch.Tensor) -> torch.Tensor:
        return torch.abs(dpt - tgt_dpt)

    def _ensure_cached_adc(
        self,
        batch_inputs: Dict[str, Any],
        ant_indices_groups: Sequence[torch.Tensor],
    ) -> None:
        data_name = batch_inputs["data_name"]
        if (data_name in self.ant_real_list) and (data_name in self.ant_imag_list):
            return

        self.ant_real_list[data_name] = []
        self.ant_imag_list[data_name] = []

        for ant_id in tqdm(range(self.cfg["bank_size"]), desc=data_name):
            ant_indices = ant_indices_groups[ant_id]
            t_pos_sel = batch_inputs["t_pos"][ant_indices]
            r_pos_sel = batch_inputs["r_pos"][ant_indices]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                alpha_sel, gradients_sel, feature_vector_sel, _, _ = self.compute_sdf_alpha(
                    self.sdf_network,
                    self.deviation_network,
                    batch_inputs["tgt_sampled_poses_norm"],
                    batch_inputs["tgt_dists"],
                    batch_inputs["tgt_dirs"],
                    self.get_anneal_val(self.step),
                    self.step,
                )

            trans_prob_sel = _compute_transmission(alpha_sel)
            weights_sel = alpha_sel * trans_prob_sel
            sigmas_sel = (
                self.signal_network(
                    batch_inputs["tgt_sampled_poses_norm"],
                    feature_vector_sel,
                    trans_prob_sel,
                )
                * weights_sel
            )

            n_chunk_ant = ant_indices.shape[0]
            normals_sel = F.normalize(gradients_sel, dim=-1)
            sim_real_sel, sim_imag_sel = lensless_ray_tracer(
                normals=normals_sel.reshape(batch_inputs["tgt_n_pts"], 3),
                sigmas=sigmas_sel.reshape(1, batch_inputs["tgt_n_pts"]).repeat(n_chunk_ant, 1),
                points=batch_inputs["tgt_sampled_poses_glb"].reshape(batch_inputs["tgt_n_pts"], 3),
                antenna_t=t_pos_sel,
                antenna_r=r_pos_sel,
                radar_cfg=self.radar_cfg,
                backend=self.rt_backend,
                use_diff=self.use_diff,
                nomralization=True,
            )
            self.ant_real_list[data_name].append(sim_real_sel.detach().cpu())
            self.ant_imag_list[data_name].append(sim_imag_sel.detach().cpu())

        self.ant_chunk_id[data_name] = 0

    def loss(self, data_sample: dict) -> Dict[str, torch.Tensor]:
        frame = self._extract_batch_inputs(data_sample)
        frame = self._apply_loss_mask(frame, data_sample["loss_mask"])

        ant_indices_groups = self.generate_series(frame["n_ant"], self.cfg["bank_size"])
        self._ensure_cached_adc(frame, ant_indices_groups)

        current_chunk_id = self.ant_chunk_id[frame["data_name"]]
        selected_indices = ant_indices_groups[current_chunk_id]
        n_chunk_ant = selected_indices.shape[0]
        t_pos_selected = frame["t_pos"][selected_indices]
        r_pos_selected = frame["r_pos"][selected_indices]

        dyntrans_mask, intersect_pts = self.intersect_sphere(
            frame["r_pos_norm"][selected_indices],
            frame["sampled_poses_norm"],
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.float16):
                calib_cdf = self.compute_cdf(intersect_pts)
                prev_cdf = self.compute_cdf(frame["sampled_poses_norm"][:, 0:1, :])
                prev_cdf_expanded = prev_cdf[None, :, 0, None]
                new_cdf = torch.where(dyntrans_mask, calib_cdf, prev_cdf_expanded)
                tgt_alpha, _, _, _, _ = self.compute_sdf_alpha(
                    self.tgt_sdf_network,
                    self.tgt_deviation_network,
                    frame["sampled_poses_norm"],
                    frame["dists"],
                    frame["dirs"],
                    self.get_anneal_val(self.step),
                    self.step,
                )

        with torch.amp.autocast("cuda", dtype=torch.float16):
            alpha, gradients, feature_vector, _, _ = self.compute_sdf_alpha(
                self.sdf_network,
                self.deviation_network,
                frame["sampled_poses_norm"],
                frame["dists"],
                frame["dirs"],
                self.get_anneal_val(self.step),
                self.step,
            )

        trans_prob = _compute_transmission(alpha)
        tgt_trans_prob = _compute_transmission(tgt_alpha)

        pre_weights = trans_prob * alpha
        pre_depth = torch.sum(pre_weights * frame["dists"], dim=-1)
        tgt_pre_weights = tgt_trans_prob * tgt_alpha
        tgt_depth = torch.sum(tgt_pre_weights * frame["dists"], dim=-1)

        trans_prob_calib = (trans_prob - prev_cdf) + new_cdf
        weights = alpha * trans_prob_calib
        sigmas = (
            self.signal_network(
                frame["sampled_poses_norm"],
                feature_vector,
                trans_prob_calib,
            )
            * weights
        )

        tgt_normals, tgt_sigmas, inbounds = self._sample_tgt_normals_and_sigmas(
            gradients,
            sigmas,
            frame["z_vals"],
            frame["tgt_z_vals"],
            n_chunk_ant,
        )

        sim_real, sim_imag = lensless_ray_tracer(
            normals=tgt_normals.reshape(frame["tgt_n_pts"], 3),
            sigmas=tgt_sigmas.reshape(n_chunk_ant, frame["tgt_n_pts"]),
            points=frame["tgt_sampled_poses_glb"].reshape(frame["tgt_n_pts"], 3),
            antenna_t=t_pos_selected,
            antenna_r=r_pos_selected,
            inbounds=inbounds.reshape(frame["tgt_n_pts"]),
            radar_cfg=self.radar_cfg,
            backend=self.rt_backend,
            use_diff=self.use_diff,
            nomralization=True,
        )

        (
            t_pos_unselected,
            r_pos_unselected,
            adc_r_unselected,
            adc_i_unselected,
        ) = self._gather_unselected_adc(
            frame["data_name"],
            ant_indices_groups,
            current_chunk_id,
            frame["t_pos"],
            frame["r_pos"],
            sim_real,
            sim_imag,
        )

        mix_real = torch.cat([sim_real, adc_r_unselected], dim=0)
        mix_imag = torch.cat([sim_imag, adc_i_unselected], dim=0)
        mix_t_pos = torch.cat([t_pos_selected, t_pos_unselected], dim=0)
        mix_r_pos = torch.cat([r_pos_selected, r_pos_unselected], dim=0)

        pred_sar_value, gt_sar_value = self._matched_filter_prediction(
            mix_real,
            mix_imag,
            mix_t_pos,
            mix_r_pos,
            frame["tgt_sampled_poses_glb"],
            frame["gt_sar_value"],
            inbounds,
        )

        self._update_chunk_pointer(frame["data_name"])

        losses: Dict[str, torch.Tensor] = {}
        losses["mf_loss"] = torch.mean(
            self.compute_mf_loss(pred_sar_value, gt_sar_value * self.loss_scale)
        )
        if self.with_grad_regression:
            losses["grad_loss"] = self._compute_grad_regression_loss(gradients) * self.with_grad_regression
        if self.with_depth_regression:
            losses["depth_loss"] = torch.mean(self.compute_dpt_loss(pre_depth, tgt_depth)) * self.with_depth_regression
        return losses
