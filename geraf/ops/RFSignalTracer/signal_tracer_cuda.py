# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import _C


def _ensure_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


class _SignalTracerFunction(Function):
    @staticmethod
    def forward(
        ctx,
        normals: torch.Tensor,
        sigmas: torch.Tensor,
        points: torch.Tensor,
        antenna_t: torch.Tensor,
        antenna_r: torch.Tensor,
        inbounds: torch.Tensor,
        numADCSample: int,
        adcSampleRate: float,
        fc_start: float,
        As_sci: float,
        total_ant_size: int,
        total_point_size: int,
        use_diff: bool,
        nomralization: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not use_diff:
            raise ValueError("The standalone signal tracer retains only the differentiable CUDA path.")

        if normals.dim() != 2 or normals.size(1) != 3:
            raise ValueError(f"`normals` must have shape [num_points, 3], got {tuple(normals.shape)}.")
        if points.dim() != 2 or points.size(1) != 3:
            raise ValueError(f"`points` must have shape [num_points, 3], got {tuple(points.shape)}.")
        if sigmas.dim() != 2:
            raise ValueError(f"`sigmas` must have shape [num_antennas, num_points], got {tuple(sigmas.shape)}.")
        if antenna_t.dim() != 2 or antenna_t.size(1) != 3:
            raise ValueError(f"`antenna_t` must have shape [num_antennas, 3], got {tuple(antenna_t.shape)}.")
        if antenna_r.dim() != 2 or antenna_r.size(1) != 3:
            raise ValueError(f"`antenna_r` must have shape [num_antennas, 3], got {tuple(antenna_r.shape)}.")
        if inbounds.dim() != 1:
            raise ValueError(f"`inbounds` must have shape [num_points], got {tuple(inbounds.shape)}.")

        normals = _ensure_contiguous(normals)
        sigmas = _ensure_contiguous(sigmas)
        points = _ensure_contiguous(points)
        antenna_t = _ensure_contiguous(antenna_t)
        antenna_r = _ensure_contiguous(antenna_r)
        inbounds = _ensure_contiguous(inbounds)

        output_real = torch.empty(
            (total_ant_size, numADCSample), dtype=normals.dtype, device=normals.device
        )
        output_imag = torch.empty_like(output_real)

        _C.SignalTracer_forward(
            normals,
            sigmas,
            points,
            antenna_t,
            antenna_r,
            inbounds,
            output_real,
            output_imag,
            numADCSample,
            adcSampleRate,
            fc_start,
            As_sci,
            total_ant_size,
            total_point_size,
        )

        ctx.save_for_backward(normals, sigmas, points, antenna_t, antenna_r, inbounds)
        ctx.numADCSample = numADCSample
        ctx.adcSampleRate = adcSampleRate
        ctx.fc_start = fc_start
        ctx.As_sci = As_sci
        ctx.total_ant_size = total_ant_size
        ctx.total_point_size = total_point_size
        ctx.nomralization = nomralization

        return output_real, output_imag

    @staticmethod
    @once_differentiable
    def backward(
        ctx,
        grad_output_real: torch.Tensor,
        grad_output_imag: torch.Tensor,
    ) -> tuple[torch.Tensor | None, ...]:
        normals, sigmas, points, antenna_t, antenna_r, inbounds = ctx.saved_tensors

        grad_output_real = _ensure_contiguous(grad_output_real)
        grad_output_imag = _ensure_contiguous(grad_output_imag)

        grad_normals = None
        grad_sigmas = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            full_grad_normals = torch.empty_like(normals)
            full_grad_sigmas = torch.empty_like(sigmas)

            _C.SignalTracer_backward_normal_sigma(
                grad_output_real,
                grad_output_imag,
                normals,
                sigmas,
                points,
                antenna_t,
                antenna_r,
                inbounds,
                full_grad_normals,
                full_grad_sigmas,
                ctx.numADCSample,
                ctx.adcSampleRate,
                ctx.fc_start,
                ctx.As_sci,
                ctx.total_ant_size,
                ctx.total_point_size,
            )

            if ctx.needs_input_grad[0]:
                grad_normals = full_grad_normals
            if ctx.needs_input_grad[1]:
                grad_sigmas = full_grad_sigmas

        return (
            grad_normals,
            grad_sigmas,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def signal_tracer_cuda(
    normals: torch.Tensor,
    sigmas: torch.Tensor,
    points: torch.Tensor,
    antenna_t: torch.Tensor,
    antenna_r: torch.Tensor,
    inbounds: torch.Tensor,
    numADCSample: int,
    adcSampleRate: float,
    fc_start: float,
    As_sci: float,
    total_ant_size: int,
    total_point_size: int,
    use_diff: bool = True,
    nomralization: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not all(
        tensor.is_cuda for tensor in (normals, sigmas, points, antenna_t, antenna_r, inbounds)
    ):
        raise ValueError("All signal tracer inputs must be CUDA tensors.")

    return _SignalTracerFunction.apply(
        normals,
        sigmas,
        points,
        antenna_t,
        antenna_r,
        inbounds,
        numADCSample,
        adcSampleRate,
        fc_start,
        As_sci,
        total_ant_size,
        total_point_size,
        use_diff,
        nomralization,
    )
