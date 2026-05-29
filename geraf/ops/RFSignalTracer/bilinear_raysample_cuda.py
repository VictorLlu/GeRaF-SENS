# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from typing import Tuple

from . import _C  # C++/CUDA extension module


class BilinearRaysampleKernel(Function):
    @staticmethod
    def forward(ctx,
                normals: torch.Tensor,         # [N_ray, Nz, 3]
                sigmas: torch.Tensor,          # [N_ant, N_ray, Nz]
                z_vals: torch.Tensor,          # [N_ray, Nz]
                tgt_z_vals: torch.Tensor,      # [N_ray, Nz_tgt]
                N_ant: int,
                N_ray: int,
                Nz: int,
                Nz_tgt: int
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # Ensure contiguous memory
        normals = normals.contiguous()
        sigmas = sigmas.contiguous()
        z_vals = z_vals.contiguous()
        tgt_z_vals = tgt_z_vals.contiguous()

        output_normals = torch.empty((N_ray, Nz_tgt, 3), dtype=normals.dtype, device=normals.device)
        output_sigmas = torch.empty((N_ant, N_ray, Nz_tgt), dtype=sigmas.dtype, device=sigmas.device)
        output_inbounds = torch.empty((N_ray, Nz_tgt), dtype=torch.bool, device=sigmas.device)

        # Call the CUDA kernel
        _C.BilinearRaysample_forward(
            normals,
            sigmas,
            z_vals,
            tgt_z_vals,
            output_normals,
            output_sigmas,
            output_inbounds,
            N_ant,
            N_ray,
            Nz,
            Nz_tgt
        )

        # Save necessary context for backward
        ctx.save_for_backward(z_vals, tgt_z_vals)
        ctx.N_ant = N_ant
        ctx.N_ray = N_ray
        ctx.Nz = Nz
        ctx.Nz_tgt = Nz_tgt

        return output_normals, output_sigmas, output_inbounds

    @staticmethod
    @once_differentiable
    def backward(ctx,
                 grad_output_normals: torch.Tensor,
                 grad_output_sigmas: torch.Tensor,
                 _  # grad_output_inbounds (bool, not differentiable)
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None, None, None, None]:

        z_vals, tgt_z_vals = ctx.saved_tensors
        N_ant = ctx.N_ant
        N_ray = ctx.N_ray
        Nz = ctx.Nz
        Nz_tgt = ctx.Nz_tgt

        grad_output_normals = grad_output_normals.contiguous()
        grad_output_sigmas = grad_output_sigmas.contiguous()

        # Allocate gradient outputs
        normals_grad = torch.zeros((N_ray, Nz, 3), dtype=grad_output_normals.dtype, device=grad_output_normals.device)
        sigmas_grad = torch.zeros((N_ant, N_ray, Nz), dtype=grad_output_sigmas.dtype, device=grad_output_sigmas.device)

        _C.BilinearRaysample_backward(
            grad_output_normals,
            grad_output_sigmas,
            z_vals,
            tgt_z_vals,
            normals_grad,
            sigmas_grad,
            N_ant,
            N_ray,
            Nz,
            Nz_tgt
        )

        return normals_grad, sigmas_grad, None, None, None, None, None, None


def bilinear_raysample_cuda(
        normals: torch.Tensor,         # [N_ray, Nz, 3]
        sigmas: torch.Tensor,          # [N_ant, N_ray, Nz]
        z_vals: torch.Tensor,          # [N_ray, Nz]
        tgt_z_vals: torch.Tensor,      # [N_ray, Nz_tgt]
        N_ant: int,
        N_ray: int,
        Nz: int,
        Nz_tgt: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CUDA wrapper for the Bilinear Ray Sample Kernel.

    Applies bilinear interpolation along the z-axis to interpolate
    normals and sigmas at target z-values using a CUDA kernel.

    Parameters
    ----------
    normals : torch.Tensor
        Surface normals per ray and depth sample. Shape: [N_ray, Nz, 3]

    sigmas : torch.Tensor
        Density values per antenna, ray, and depth. Shape: [N_ant, N_ray, Nz]

    z_vals : torch.Tensor
        Original z-coordinates along rays. Shape: [N_ray, Nz]

    tgt_z_vals : torch.Tensor
        Target z-values to interpolate at. Shape: [N_ray, Nz_tgt]

    N_ant : int
        Number of antennas

    N_ray : int
        Number of rays

    Nz : int
        Number of samples along each ray

    Nz_tgt : int
        Number of target interpolation depths

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - Interpolated normals: [N_ray, Nz_tgt, 3]
        - Interpolated sigmas: [N_ant, N_ray, Nz_tgt]
        - In-bounds mask: [N_ray, Nz_tgt] (bool)
    """
    return BilinearRaysampleKernel.apply(normals, sigmas, z_vals, tgt_z_vals, N_ant, N_ray, Nz, Nz_tgt)
