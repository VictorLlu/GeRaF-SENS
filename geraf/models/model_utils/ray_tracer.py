# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import torch
from torch import nn
from typing import Dict

def lensless_ray_tracer(normals: torch.Tensor, 
                    sigmas: torch.Tensor,
                    points: torch.Tensor,  
                    antenna_t: torch.Tensor, 
                    antenna_r: torch.Tensor, 
                    radar_cfg: Dict[str, float],
                    backend: str = "cuda", 
                    inbounds: torch.Tensor = None,
                    use_diff: bool = True,
                    nomralization: bool = True):
    """
    Function to perform ray tracing using CUDA kernel with provided inputs.

    Parameters:
    normals (torch.Tensor): Tensor representing the normal vectors at each point.
    points (torch.Tensor): Tensor representing the 3D positions of the points to trace.
    antenna_t (torch.Tensor): Tensor representing the transmitting antenna parameters.
    antenna_r (torch.Tensor): Tensor representing the receiving antenna parameters.
    radar_cfg (Dict[str, float]): Dictionary containing radar parameters like ADC sample rate, frequency, and antenna size.

    Returns:
    torch.Tensor: Output from the CUDA ray tracer kernel.
    """

    # Define the pointing direction (assumed to be along the x-axis). This is a fixed vector with no gradients.
    num_points = normals.shape[0]
    if inbounds is None:
        inbounds = torch.ones(num_points, dtype=torch.bool, device=points.device)
    # assert antenna_t.shape[0] == antenna_r.shape[0] == radar_cfg['total_ant_size']

    if backend != "cuda":
        raise ValueError(f"Unsupported ray tracer backend: {backend!r}. Only 'cuda' is retained.")

    from geraf.ops.RFSignalTracer.signal_tracer_cuda import signal_tracer_cuda as ray_tracer_func

    sim_real, sim_image = ray_tracer_func(
        normals=normals,
        sigmas=sigmas,
        points=points,
        antenna_t=antenna_t,
        antenna_r=antenna_r,
        inbounds=inbounds,
        numADCSample=radar_cfg['numADCSample'],
        adcSampleRate=radar_cfg['adcSampleRate'],
        fc_start=radar_cfg['fc_start'],
        As_sci=radar_cfg['As_sci'],
        total_ant_size=antenna_t.shape[0],
        total_point_size=num_points,
        use_diff=use_diff,
        nomralization=nomralization,
    )
    return sim_real, sim_image
