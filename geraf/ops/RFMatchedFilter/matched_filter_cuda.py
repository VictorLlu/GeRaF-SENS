# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from typing import Tuple
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from einops import rearrange
from . import _C  # Assumed C++/CUDA extension import
from tqdm import tqdm

class _MatchedFilterKernel(Function):
    @staticmethod
    def forward(
        ctx, 
        real: torch.Tensor, 
        image: torch.Tensor, 
        tpos: torch.Tensor,
        rpos: torch.Tensor,
        voxel: torch.Tensor,
        total_ant_size: int, 
        numADCSample: int,
        adcSampleRate: float, 
        As_sci: float,
        N: int
    ):
        """
        Forward pass for matched filter kernel.

        Parameters:
        ----------
        real : torch.Tensor
            Real part of the input tensor (must be 3D and on CUDA).
        image : torch.Tensor
            Imaginary part of the input tensor (must be 3D and on CUDA).
        pos : torch.Tensor
            Position tensor (must be contiguous and 3D).
        xmin, ymin, zmin : float
            Minimum coordinates for x, y, and z.
        resx, resy, resz : float
            Resolution in x, y (azimuth), and z (elevation).
        total_ant_size, numADCSample : int
            Number of antenna, and number of ADC samples.
        adcSampleRate : float
            ADC sample rate.
        As_sci : float
            Chirp slope scaled value.
        Nx, Ny, Nz : int
            Number of grid points in x, y, and z directions.

        Returns:
        -------
        torch.Tensor
            Output tensors for real and imaginary parts.
        """
        # Check dimensions
        assert real.dim() == 2, f"Expected 2D tensor (total_ant_size, numADCSample) for real part, but got: {real.dim()}D"
        assert image.dim() == 2, f"Expected 2D tensor(total_ant_size, numADCSample) for image part, but got: {image.dim()}D"
        assert tpos.dim() == 2, f"Expected 2D tensor (total_ant_size, 3) for position data, but got: {tpos.dim()}D"
        assert rpos.dim() == 2, f"Expected 2D tensor (total_ant_size, 3) for position data, but got: {rpos.dim()}D"
        assert voxel.dim() == 2, f"Expected 3D tensor (numPoints, 3) for voxel data, but got: {voxel.dim()}D"

        # Check sizes match target sizes
        assert real.size(0) == total_ant_size and real.size(1) == numADCSample, (
            f"Size mismatch for real tensor: expected ({total_ant_size}, {numADCSample}), "
            f"but got: ({real.size(0)}, {real.size(1)})"
        )

        assert image.size(0) == total_ant_size and image.size(1) == numADCSample, (
            f"Size mismatch for image tensor: expected ({total_ant_size}, {numADCSample}), "
            f"but got: ({image.size(0)}, {image.size(1)})"
        )

        assert tpos.size(0) == total_ant_size and tpos.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({total_ant_size}, 3), "
            f"but got: ({tpos.size(0)}, {tpos.size(1)})"
        )

        assert rpos.size(0) == total_ant_size and rpos.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({total_ant_size}, 3), "
            f"but got: ({rpos.size(0)}, {rpos.size(1)})"
        )

        assert voxel.size(0) == N and voxel.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({N}, 3), "
            f"but got: ({voxel.size(0)}, {voxel.size(1)})"
        )

        voxel = voxel.transpose(0, 1)

        # Ensure tensors are contiguous for efficiency
        if not real.is_contiguous():
            real = real.contiguous()
        if not image.is_contiguous():
            image = image.contiguous()
        if not rpos.is_contiguous():
            rpos = rpos.contiguous()
        if not tpos.is_contiguous():
            tpos = tpos.contiguous()
        if not voxel.is_contiguous():
            voxel = voxel.contiguous()

        # Initialize the output tensors
        output_real = torch.empty((N), dtype=real.dtype, device=real.device)
        output_image = torch.empty((N), dtype=real.dtype, device=real.device)

        # Call the CUDA kernel via C++/CUDA extension
        _C.MatchedFilter_forward(
            real, image, tpos, rpos, output_real, output_image,
            voxel,
            total_ant_size, numADCSample,
            adcSampleRate, As_sci,
            N
        )

        # Save tensors for the backward pass
        ctx.save_for_backward(real, image, tpos, rpos, voxel)
        
        # Save non-tensor parameters for backward pass
        ctx.total_ant_size = total_ant_size 
        ctx.numADCSample = numADCSample
        ctx.adcSampleRate = adcSampleRate
        ctx.As_sci = As_sci
        ctx.N = N

        return output_real, output_image

    @staticmethod
    @once_differentiable
    def backward(
        ctx, 
        grad_output_real: torch.Tensor, 
        grad_output_image: torch.Tensor
    ):
        """
        Backward pass for matched filter kernel.

        Parameters:
        ----------
        grad_output_real : torch.Tensor
            Gradient of the real part of the output tensor.
        grad_output_image : torch.Tensor
            Gradient of the imaginary part of the output tensor.

        Returns:
        -------
        tuple
            Gradients of the inputs (real, image, pos) and other parameters (if necessary).
        """
        # Retrieve saved tensors from the forward pass
        real, image, tpos, rpos, voxel = ctx.saved_tensors

        # Initialize gradients for real, image, and pos
        grad_real = grad_image = None

        # Use saved non-tensor parameters
        total_ant_size = ctx.total_ant_size
        numADCSample = ctx.numADCSample
        adcSampleRate = ctx.adcSampleRate
        As_sci = ctx.As_sci
        N = ctx.N

        # Ensure that the gradient tensors are on the GPU
        assert grad_output_real.is_cuda and grad_output_image.is_cuda, \
            "grad_output_real and grad_output_image must be CUDA tensors."

        # If the gradient tensors are not contiguous in memory, make them contiguous
        if not grad_output_real.is_contiguous():
            grad_output_real = grad_output_real.contiguous()

        if not grad_output_image.is_contiguous():
            grad_output_image = grad_output_image.contiguous()

        # Compute the gradient for real if needed
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            grad_real = torch.empty_like(real)
            grad_image = torch.empty_like(image)
            _C.MatchedFilter_backward_realimag(
                grad_output_real, grad_output_image,
                real, image, tpos, rpos,
                grad_real, grad_image, voxel,
                total_ant_size, numADCSample,
                adcSampleRate, As_sci,
                N
            )

        # grad_real = torch.nan_to_num(grad_real, nan=0.0)  # Replaces NaNs with zero
        # grad_image = torch.nan_to_num(grad_image, nan=0.0)  # Replaces NaNs with zero

        # Return gradients for real, image, pos, and None for the non-tensor parameters
        return (
        grad_real,    # Gradient for real
        grad_image,   # Gradient for image
        None,         # Gradient for tpos (currently None)
        None,         # Gradient for rpos (currently None)
        None,  # Gradients for voxel
        None, None, None, None,  # Gradients for total_ant_size, numADCSample, adcSampleRate, As_sci
        None                   # Gradients for N
    )


# Wrapper function for the matched filter kernel
def matched_filter_cuda(
    real: torch.Tensor, 
    image: torch.Tensor, 
    tpos: torch.Tensor,
    rpos: torch.Tensor,
    voxel: torch.Tensor,
    total_ant_size: int, 
    numADCSample: int,
    adcSampleRate: float, 
    As_sci: float,
    N: int 
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for _MatchedFilterKernel.

    Parameters:
    ----------
    real : torch.Tensor
        The real part of the input tensor (must be 3D and on CUDA).
    image : torch.Tensor
        The imaginary part of the input tensor (must be 3D and on CUDA).
    pos : torch.Tensor
        The position tensor (must be 3D and contiguous).
    xmin, ymin, zmin : float
        Minimum coordinates for x, y (azimuth), and z (elevation).
    resx, resy, resz : float
        Resolution for x, y, and z directions.
    num_x_stp, num_z_stp, numADCSample : int
        Number of steps in x, z, and number of ADC samples.
    adcSampleRate : float
        ADC sample rate.
    As_sci : float
        Chirp slope scaled value.
    x_start, x_end, z_start, z_end : int
        Start and end indices for x and z.
    Nx, Ny, Nz : int
        Number of grid points in x, y, and z directions.

    Returns:
    -------
    tuple
        Output tensors (real and imaginary) after applying the matched filter kernel.
    """
    # Ensure inputs are on CUDA
    assert real.is_cuda and image.is_cuda and tpos.is_cuda and rpos.is_cuda and voxel.is_cuda, "All tensors must be on CUDA."

    return _MatchedFilterKernel.apply(
        real, image, tpos, rpos, 
        voxel,
        total_ant_size, numADCSample, 
        adcSampleRate, As_sci,
        N
    )
