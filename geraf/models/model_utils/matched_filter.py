# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import torch
from torch import nn
from typing import Dict

def base_matched_filter(real: torch.Tensor, 
                        image: torch.Tensor,
                        t_pos: torch.Tensor,
                        r_pos: torch.Tensor,
                        points: torch.Tensor, 
                        radar_cfg: Dict[str, float],  # Configuration dictionary for radar parameters
                        backend: str = "cuda",
                       ):
    """
    Function to apply matched filtering using CUDA kernel with provided inputs.

    Parameters:
    real (torch.Tensor): Real part of the input data (e.g., SAR data).
    image (torch.Tensor): Imaginary part of the input data (e.g., SAR data).
    t_pos (torch.Tensor): Tensor representing the transmitting antenna positions.
    r_pos (torch.Tensor): Tensor representing the receiving antenna positions.
    radar_cfg (Dict[str, float]): Configuration dictionary for the radar parameters.

    Returns:
    torch.Tensor: sar_real - Real part of the filtered output.
    torch.Tensor: sar_image - Imaginary part of the filtered output.
    """
    if backend != "cuda":
        raise ValueError(
            f"Unsupported matched filter backend: {backend!r}. "
            "Only 'cuda' is retained."
        )
    from geraf.ops.RFMatchedFilter.matched_filter_cuda import matched_filter_cuda as matched_filter_func

    total_ant_size = t_pos.shape[0]
    numADCSample = radar_cfg['numADCSample']
    adcSampleRate = radar_cfg['adcSampleRate']
    As_sci = radar_cfg['As_sci']
    num_points = points.shape[0]
    sar_real, sar_image = matched_filter_func(
        real, image, t_pos, r_pos, points,
        total_ant_size, numADCSample,
        adcSampleRate, As_sci,
        num_points
    )

    return sar_real, sar_image

def faster_matched_filter(real_sel: torch.Tensor, 
                        image_sel: torch.Tensor, 
                        real_unsel: torch.Tensor, 
                        image_unsel: torch.Tensor,
                        tpos_sel: torch.Tensor,
                        rpos_sel: torch.Tensor,
                        tpos_unsel: torch.Tensor,
                        rpos_unsel: torch.Tensor,
                        points: torch.Tensor, 
                        radar_cfg: Dict[str, float],  # Configuration dictionary for radar parameters
                        backend: str = "cuda",
                       ):
    """
    Function to apply matched filtering using CUDA kernel with provided inputs.

    Parameters:
    real (torch.Tensor): Real part of the input data (e.g., SAR data).
    image (torch.Tensor): Imaginary part of the input data (e.g., SAR data).
    t_pos (torch.Tensor): Tensor representing the transmitting antenna positions.
    r_pos (torch.Tensor): Tensor representing the receiving antenna positions.
    radar_cfg (Dict[str, float]): Configuration dictionary for the radar parameters.

    Returns:
    torch.Tensor: sar_real - Real part of the filtered output.
    torch.Tensor: sar_image - Imaginary part of the filtered output.
    """
    if backend != "cuda":
        raise ValueError(
            f"Unsupported faster matched filter backend: {backend!r}. Only 'cuda' is retained."
        )

    from geraf.ops.RFMatchedFilter.faster_matched_filter_cuda import (
        faster_matched_filter_cuda as matched_filter_func,
    )


    # Extract radar configuration parameters from radar_cfg
    num_sel = tpos_sel.shape[0]
    num_unsel = tpos_unsel.shape[0]
    total_ant_size = num_sel + num_unsel
    numADCSample = radar_cfg['numADCSample']  # Number of ADC samples in radar

    adcSampleRate = radar_cfg['adcSampleRate']  # Sampling rate of the ADC
    As_sci = radar_cfg['As_sci']  # Some radar parameter (e.g., signal strength)

    # Extract grid size for matched filtering
    num_points = points.shape[0]

    # Call the CUDA-based matched filter function with all necessary inputs
    sar_real, sar_image = matched_filter_func(
        real_sel, image_sel, 
        real_unsel, image_unsel, 
        tpos_sel, rpos_sel, tpos_unsel, rpos_unsel, points,
        total_ant_size, num_sel, numADCSample,
        adcSampleRate, As_sci,
        num_points
    )

    return sar_real, sar_image
