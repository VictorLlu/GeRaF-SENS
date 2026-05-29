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

class _FasterMatchedFilterKernel(Function):
    @staticmethod
    def forward(
        ctx, 
        real_sel: torch.Tensor, 
        image_sel: torch.Tensor, 
        real_unsel: torch.Tensor, 
        image_unsel: torch.Tensor,
        tpos_sel: torch.Tensor,
        rpos_sel: torch.Tensor,
        tpos_unsel: torch.Tensor,
        rpos_unsel: torch.Tensor,
        voxel: torch.Tensor,
        total_ant_size: int, 
        num_sel: int,
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
        assert real_sel.dim() == 2, f"Expected 2D tensor (total_ant_size, numADCSample) for real part, but got: {real_sel.dim()}D"
        assert image_sel.dim() == 2, f"Expected 2D tensor(total_ant_size, numADCSample) for image part, but got: {image_sel.dim()}D"
        assert tpos_sel.dim() == 2, f"Expected 2D tensor (total_ant_size, 3) for position data, but got: {tpos.dim()}D"
        assert rpos_sel.dim() == 2, f"Expected 2D tensor (total_ant_size, 3) for position data, but got: {rpos.dim()}D"
        assert real_unsel.dim() == 2, f"Expected 2D tensor (total_ant_size, numADCSample) for real part, but got: {real_sel.dim()}D"
        assert image_unsel.dim() == 2, f"Expected 2D tensor(total_ant_size, numADCSample) for image part, but got: {image_sel.dim()}D"
        assert tpos_unsel.dim() == 2, f"Expected 2D tensor (total_ant_size, 3) for position data, but got: {tpos.dim()}D"
        assert rpos_unsel.dim() == 2, f"Expected 2D tensor (total_ant_size, 3) for position data, but got: {rpos.dim()}D"
        assert voxel.dim() == 2, f"Expected 3D tensor (numPoints, 3) for voxel data, but got: {voxel.dim()}D"

        num_unsel = total_ant_size - num_sel
        # Check sizes match target sizes
        assert real_sel.size(0) == num_sel and real_sel.size(1) == numADCSample, (
            f"Size mismatch for real tensor: expected ({num_sel}, {numADCSample}), "
            f"but got: ({real_sel.size(0)}, {real_sel.size(1)})"
        )

        assert image_sel.size(0) == num_sel and image_sel.size(1) == numADCSample, (
            f"Size mismatch for image tensor: expected ({num_sel}, {numADCSample}), "
            f"but got: ({image_sel.size(0)}, {image_sel.size(1)})"
        )

        assert tpos_sel.size(0) == num_sel and tpos_sel.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({num_sel}, 3), "
            f"but got: ({tpos_sel.size(0)}, {tpos_sel.size(1)})"
        )

        assert rpos_sel.size(0) == num_sel and rpos_sel.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({num_sel}, 3), "
            f"but got: ({rpos_sel.size(0)}, {rpos_sel.size(1)})"
        )

        assert real_unsel.size(0) == num_unsel and real_unsel.size(1) == numADCSample, (
            f"Size mismatch for real tensor: expected ({num_unsel}, {numADCSample}), "
            f"but got: ({real_unsel.size(0)}, {real_unsel.size(1)})"
        )

        assert image_unsel.size(0) == num_unsel and image_unsel.size(1) == numADCSample, (
            f"Size mismatch for image tensor: expected ({num_unsel}, {numADCSample}), "
            f"but got: ({image_unsel.size(0)}, {image_unsel.size(1)})"
        )

        assert tpos_unsel.size(0) == num_unsel and tpos_unsel.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({num_unsel}, 3), "
            f"but got: ({tpos_unsel.size(0)}, {tpos_unsel.size(1)})"
        )

        assert rpos_unsel.size(0) == num_unsel and rpos_unsel.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({num_unsel}, 3), "
            f"but got: ({rpos_unsel.size(0)}, {rpos_unsel.size(1)})"
        )

        assert voxel.size(0) == N and voxel.size(1) == 3, (
            f"Size mismatch for pos tensor: expected ({N}, 3), "
            f"but got: ({voxel.size(0)}, {voxel.size(1)})"
        )

        voxel = voxel.transpose(0, 1)

        # Ensure tensors are contiguous for efficiency
        if not real_sel.is_contiguous():
            real_sel = real_sel.contiguous()
        if not image_sel.is_contiguous():
            image_sel = image_sel.contiguous()
        if not rpos_sel.is_contiguous():
            rpos_sel = rpos_sel.contiguous()
        if not tpos_sel.is_contiguous():
            tpos_sel = tpos_sel.contiguous()
        if not real_unsel.is_contiguous():
            real_unsel = real_unsel.contiguous()
        if not image_unsel.is_contiguous():
            image_unsel = image_unsel.contiguous()
        if not rpos_unsel.is_contiguous():
            rpos_unsel = rpos_unsel.contiguous()
        if not tpos_unsel.is_contiguous():
            tpos_unsel = tpos_unsel.contiguous()
        if not voxel.is_contiguous():
            voxel = voxel.contiguous()

        real = torch.cat([real_sel, real_unsel], dim=0)
        image = torch.cat([image_sel, image_unsel], dim=0)
        tpos = torch.cat([tpos_sel, tpos_unsel], dim=0)
        rpos = torch.cat([rpos_sel, rpos_unsel], dim=0)

        # Initialize the output tensors
        output_real = torch.empty((N), dtype=real_sel.dtype, device=real_sel.device)
        output_image = torch.empty((N), dtype=real_sel.dtype, device=real_sel.device)

        # Call the CUDA kernel via C++/CUDA extension
        _C.MatchedFilter_forward(
            real, image, tpos, rpos, output_real, output_image,
            voxel,
            total_ant_size, numADCSample,
            adcSampleRate, As_sci,
            N
        )

        # Save tensors for the backward pass
        ctx.save_for_backward(real_sel, image_sel, 
                              real_unsel, image_unsel, 
                              tpos_sel, rpos_sel, 
                              tpos_unsel, rpos_unsel, voxel)
        
        # Save non-tensor parameters for backward pass
        ctx.total_ant_size = total_ant_size 
        ctx.num_sel = num_sel
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
        real_sel, image_sel, \
        real_unsel, image_unsel, \
        tpos_sel, rpos_sel, \
        tpos_unsel, rpos_unsel, voxel = ctx.saved_tensors

        # Initialize gradients for real, image, and pos
        grad_real = grad_image = None

        # Use saved non-tensor parameters
        total_ant_size = ctx.total_ant_size
        num_sel = ctx.num_sel
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
            grad_real = torch.empty_like(real_sel)
            grad_image = torch.empty_like(image_sel)
            _C.MatchedFilter_backward_realimag(
                grad_output_real, grad_output_image,
                real_sel, image_sel, tpos_sel, rpos_sel,
                grad_real, grad_image, voxel,
                num_sel, numADCSample,
                adcSampleRate, As_sci,
                N
            )
            grad_real = grad_real * num_sel / total_ant_size
            grad_image = grad_image * num_sel / total_ant_size

        # grad_real = torch.nan_to_num(grad_real, nan=0.0)  # Replaces NaNs with zero
        # grad_image = torch.nan_to_num(grad_image, nan=0.0)  # Replaces NaNs with zero

        # Return gradients for real, image, pos, and None for the non-tensor parameters
        return (
        grad_real,    # Gradient for real_sel
        grad_image,   # Gradient for image
        None, None,   # Gradient for unsel real, image
        None, None,   # Gradient for tpos, rpos
        None, None,   # Gradient for unsel tpos rpos
        None,  # Gradients for voxel
        None, None, None, None, None,  # Gradients for total_ant_size, numsel, numADCSample, adcSampleRate, As_sci
        None                   # Gradients for N
    )


# Wrapper function for the matched filter kernel
def faster_matched_filter_cuda(
    real_sel: torch.Tensor, 
    image_sel: torch.Tensor, 
    real_unsel: torch.Tensor, 
    image_unsel: torch.Tensor,
    tpos_sel: torch.Tensor,
    rpos_sel: torch.Tensor,
    tpos_unsel: torch.Tensor,
    rpos_unsel: torch.Tensor,
    voxel: torch.Tensor,
    total_ant_size: int, 
    num_sel: int,
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
    assert real_sel.is_cuda and image_sel.is_cuda and tpos_sel.is_cuda and rpos_sel.is_cuda and voxel.is_cuda, "All tensors must be on CUDA."
    assert real_unsel.is_cuda and image_unsel.is_cuda and tpos_unsel.is_cuda and rpos_unsel.is_cuda, "All tensors must be on CUDA."

    return _FasterMatchedFilterKernel.apply(
        real_sel, image_sel, 
        real_unsel, image_unsel, 
        tpos_sel, rpos_sel,
        tpos_unsel, rpos_unsel, 
        voxel,
        total_ant_size, num_sel, 
        numADCSample, 
        adcSampleRate, As_sci,
        N
    )
