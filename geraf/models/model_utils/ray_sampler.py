# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import torch
from typing import Tuple


def bilinear_ray_sampler(
    normals: torch.Tensor,          # [N_ray, Nz, 3]
    sigmas: torch.Tensor,           # [N_ant, N_ray, Nz]
    z_vals: torch.Tensor,           # [N_ray, Nz]
    tgt_z_vals: torch.Tensor,       # [N_ray, Nz_tgt]
    backend: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs bilinear ray sampling (e.g., for radar rendering or beamforming)
    using either a CUDA or PyTorch backend.

    Parameters
    ----------
    normals : torch.Tensor
        Surface normals per ray and depth sample. Shape: [N_ray, Nz, 3]

    sigmas : torch.Tensor
        Density or reflection strength per antenna, ray, and depth. Shape: [N_ant, N_ray, Nz]

    z_vals : torch.Tensor
        Original z-coordinates along rays. Shape: [N_ray, Nz]

    tgt_z_vals : torch.Tensor
        Target z-values to interpolate at. Shape: [N_ray, Nz_tgt]

    backend : str
        Backend to use: 'cuda' or 'torch'. Default is 'cuda'.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - Interpolated normals at target z-values. Shape: [N_ray, Nz_tgt, 3]
        - Interpolated sigmas at target z-values. Shape: [N_ant, N_ray, Nz_tgt]
        - In-bounds mask indicating valid interpolation. Shape: [N_ray, Nz_tgt] (bool)
    """

    N_ray, Nz, _ = normals.shape
    N_ant = sigmas.shape[0]
    Nz_tgt = tgt_z_vals.shape[1]

    if backend == "cuda":
        from geraf.ops.RFSignalTracer.bilinear_raysample_cuda import (
            bilinear_raysample_cuda as ray_sampler,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Use 'cuda' or 'torch'.")

    # Run ray sampling kernel
    tgt_normals, tgt_sigmas, inbounds = ray_sampler(
        normals=normals,
        sigmas=sigmas,
        z_vals=z_vals,
        tgt_z_vals=tgt_z_vals,
        N_ant=N_ant,
        N_ray=N_ray,
        Nz=Nz,
        Nz_tgt=Nz_tgt
    )

    return tgt_normals, tgt_sigmas, inbounds
