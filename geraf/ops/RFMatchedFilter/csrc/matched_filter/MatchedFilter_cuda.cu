// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <torch/types.h>

#include "MatchedFilter.h"

namespace MatchedFilter {
namespace {

void CheckCommonTensorLayout(
    const at::Tensor& real,
    const at::Tensor& imag,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    const at::Tensor& voxel) {
  TORCH_CHECK(real.is_cuda(), "`real` must be a CUDA tensor.");
  TORCH_CHECK(imag.is_cuda(), "`imag` must be a CUDA tensor.");
  TORCH_CHECK(tx_pos.is_cuda(), "`tx_pos` must be a CUDA tensor.");
  TORCH_CHECK(rx_pos.is_cuda(), "`rx_pos` must be a CUDA tensor.");
  TORCH_CHECK(voxel.is_cuda(), "`voxel` must be a CUDA tensor.");

  TORCH_CHECK(real.is_contiguous(), "`real` must be contiguous.");
  TORCH_CHECK(imag.is_contiguous(), "`imag` must be contiguous.");
  TORCH_CHECK(tx_pos.is_contiguous(), "`tx_pos` must be contiguous.");
  TORCH_CHECK(rx_pos.is_contiguous(), "`rx_pos` must be contiguous.");
  TORCH_CHECK(voxel.is_contiguous(), "`voxel` must be contiguous.");

  TORCH_CHECK(real.dim() == 2, "`real` must have shape [A, S].");
  TORCH_CHECK(imag.dim() == 2, "`imag` must have shape [A, S].");
  TORCH_CHECK(tx_pos.dim() == 2, "`tx_pos` must have shape [A, 3].");
  TORCH_CHECK(rx_pos.dim() == 2, "`rx_pos` must have shape [A, 3].");
  TORCH_CHECK(voxel.dim() == 2, "`voxel` must have shape [3, N].");

  TORCH_CHECK(real.sizes() == imag.sizes(), "`real` and `imag` must match.");
  TORCH_CHECK(tx_pos.size(1) == 3, "`tx_pos` must store XYZ triplets.");
  TORCH_CHECK(rx_pos.size(1) == 3, "`rx_pos` must store XYZ triplets.");
}

void CheckForwardInputs(
    const at::Tensor& real,
    const at::Tensor& imag,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    const at::Tensor& output_real,
    const at::Tensor& output_imag,
    const at::Tensor& voxel,
    int total_ant_size,
    int num_samples,
    int num_voxels) {
  CheckCommonTensorLayout(real, imag, tx_pos, rx_pos, voxel);

  TORCH_CHECK(output_real.is_cuda(), "`output_real` must be a CUDA tensor.");
  TORCH_CHECK(output_imag.is_cuda(), "`output_imag` must be a CUDA tensor.");
  TORCH_CHECK(output_real.is_contiguous(), "`output_real` must be contiguous.");
  TORCH_CHECK(output_imag.is_contiguous(), "`output_imag` must be contiguous.");
  TORCH_CHECK(output_real.dim() == 1, "`output_real` must have shape [N].");
  TORCH_CHECK(output_imag.dim() == 1, "`output_imag` must have shape [N].");

  TORCH_CHECK(real.size(0) == total_ant_size, "`real` antenna dimension mismatch.");
  TORCH_CHECK(real.size(1) == num_samples, "`real` sample dimension mismatch.");
  TORCH_CHECK(imag.size(0) == total_ant_size, "`imag` antenna dimension mismatch.");
  TORCH_CHECK(imag.size(1) == num_samples, "`imag` sample dimension mismatch.");
  TORCH_CHECK(tx_pos.size(0) == total_ant_size, "`tx_pos` antenna dimension mismatch.");
  TORCH_CHECK(rx_pos.size(0) == total_ant_size, "`rx_pos` antenna dimension mismatch.");
  TORCH_CHECK(voxel.size(0) == 3, "`voxel` must be laid out as [3, N].");
  TORCH_CHECK(voxel.size(1) == num_voxels, "`voxel` count mismatch.");
  TORCH_CHECK(output_real.size(0) == num_voxels, "`output_real` voxel dimension mismatch.");
  TORCH_CHECK(output_imag.size(0) == num_voxels, "`output_imag` voxel dimension mismatch.");
}

void CheckBackwardInputs(
    const at::Tensor& grad_output_real,
    const at::Tensor& grad_output_imag,
    const at::Tensor& real,
    const at::Tensor& imag,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    const at::Tensor& grad_real,
    const at::Tensor& grad_imag,
    const at::Tensor& voxel,
    int total_ant_size,
    int num_samples,
    int num_voxels) {
  CheckCommonTensorLayout(real, imag, tx_pos, rx_pos, voxel);

  TORCH_CHECK(grad_output_real.is_cuda(), "`grad_output_real` must be a CUDA tensor.");
  TORCH_CHECK(grad_output_imag.is_cuda(), "`grad_output_imag` must be a CUDA tensor.");
  TORCH_CHECK(grad_real.is_cuda(), "`grad_real` must be a CUDA tensor.");
  TORCH_CHECK(grad_imag.is_cuda(), "`grad_imag` must be a CUDA tensor.");

  TORCH_CHECK(grad_output_real.is_contiguous(), "`grad_output_real` must be contiguous.");
  TORCH_CHECK(grad_output_imag.is_contiguous(), "`grad_output_imag` must be contiguous.");
  TORCH_CHECK(grad_real.is_contiguous(), "`grad_real` must be contiguous.");
  TORCH_CHECK(grad_imag.is_contiguous(), "`grad_imag` must be contiguous.");

  TORCH_CHECK(grad_output_real.dim() == 1, "`grad_output_real` must have shape [N].");
  TORCH_CHECK(grad_output_imag.dim() == 1, "`grad_output_imag` must have shape [N].");
  TORCH_CHECK(grad_real.dim() == 2, "`grad_real` must have shape [A, S].");
  TORCH_CHECK(grad_imag.dim() == 2, "`grad_imag` must have shape [A, S].");

  TORCH_CHECK(grad_output_real.size(0) == num_voxels, "`grad_output_real` voxel dimension mismatch.");
  TORCH_CHECK(grad_output_imag.size(0) == num_voxels, "`grad_output_imag` voxel dimension mismatch.");
  TORCH_CHECK(grad_real.size(0) == total_ant_size, "`grad_real` antenna dimension mismatch.");
  TORCH_CHECK(grad_real.size(1) == num_samples, "`grad_real` sample dimension mismatch.");
  TORCH_CHECK(grad_imag.size(0) == total_ant_size, "`grad_imag` antenna dimension mismatch.");
  TORCH_CHECK(grad_imag.size(1) == num_samples, "`grad_imag` sample dimension mismatch.");
}

}  // namespace

void LaunchMatchedFilterForwardKernel(
    const at::Tensor& real,
    const at::Tensor& imag,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    at::Tensor& output_real,
    at::Tensor& output_imag,
    const at::Tensor& voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels);

void LaunchMatchedFilterFftForwardKernel(
    const at::Tensor& real,
    const at::Tensor& imag,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    at::Tensor& output_real,
    at::Tensor& output_imag,
    const at::Tensor& voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels);

void LaunchMatchedFilterBackwardKernel(
    const at::Tensor& grad_output_real,
    const at::Tensor& grad_output_imag,
    const at::Tensor& real,
    const at::Tensor& imag,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    at::Tensor& grad_real,
    at::Tensor& grad_imag,
    const at::Tensor& voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels);

void LaunchMatchedFilterFftBackwardKernel(
    const at::Tensor& grad_output_real,
    const at::Tensor& grad_output_imag,
    const at::Tensor& real,
    const at::Tensor& imag,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    at::Tensor& grad_real,
    at::Tensor& grad_imag,
    const at::Tensor& voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels);

void MatchedFilter_forward(
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor output_real,
    at::Tensor output_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  CheckForwardInputs(
      real,
      imag,
      tx_pos,
      rx_pos,
      output_real,
      output_imag,
      voxel,
      total_ant_size,
      num_adc_samples,
      num_voxels);
  MatchedFilter_forward_cuda(
      real,
      imag,
      tx_pos,
      rx_pos,
      output_real,
      output_imag,
      voxel,
      total_ant_size,
      num_adc_samples,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

void MatchedFilter_fft_forward(
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor output_real,
    at::Tensor output_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  CheckForwardInputs(
      real,
      imag,
      tx_pos,
      rx_pos,
      output_real,
      output_imag,
      voxel,
      total_ant_size,
      num_range_bins,
      num_voxels);
  MatchedFilter_fft_forward_cuda(
      real,
      imag,
      tx_pos,
      rx_pos,
      output_real,
      output_imag,
      voxel,
      total_ant_size,
      num_range_bins,
      fft_spacing,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

void MatchedFilter_backward_realimag(
    const at::Tensor grad_output_real,
    const at::Tensor grad_output_imag,
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor grad_real,
    at::Tensor grad_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  CheckBackwardInputs(
      grad_output_real,
      grad_output_imag,
      real,
      imag,
      tx_pos,
      rx_pos,
      grad_real,
      grad_imag,
      voxel,
      total_ant_size,
      num_adc_samples,
      num_voxels);
  MatchedFilter_backward_realimag_cuda(
      grad_output_real,
      grad_output_imag,
      real,
      imag,
      tx_pos,
      rx_pos,
      grad_real,
      grad_imag,
      voxel,
      total_ant_size,
      num_adc_samples,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

void MatchedFilter_fft_backward_realimag(
    const at::Tensor grad_output_real,
    const at::Tensor grad_output_imag,
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor grad_real,
    at::Tensor grad_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  CheckBackwardInputs(
      grad_output_real,
      grad_output_imag,
      real,
      imag,
      tx_pos,
      rx_pos,
      grad_real,
      grad_imag,
      voxel,
      total_ant_size,
      num_range_bins,
      num_voxels);
  MatchedFilter_fft_backward_realimag_cuda(
      grad_output_real,
      grad_output_imag,
      real,
      imag,
      tx_pos,
      rx_pos,
      grad_real,
      grad_imag,
      voxel,
      total_ant_size,
      num_range_bins,
      fft_spacing,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

void MatchedFilter_forward_cuda(
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor output_real,
    at::Tensor output_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  LaunchMatchedFilterForwardKernel(
      real,
      imag,
      tx_pos,
      rx_pos,
      output_real,
      output_imag,
      voxel,
      total_ant_size,
      num_adc_samples,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

void MatchedFilter_fft_forward_cuda(
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor output_real,
    at::Tensor output_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  LaunchMatchedFilterFftForwardKernel(
      real,
      imag,
      tx_pos,
      rx_pos,
      output_real,
      output_imag,
      voxel,
      total_ant_size,
      num_range_bins,
      fft_spacing,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

void MatchedFilter_backward_realimag_cuda(
    const at::Tensor grad_output_real,
    const at::Tensor grad_output_imag,
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor grad_real,
    at::Tensor grad_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  LaunchMatchedFilterBackwardKernel(
      grad_output_real,
      grad_output_imag,
      real,
      imag,
      tx_pos,
      rx_pos,
      grad_real,
      grad_imag,
      voxel,
      total_ant_size,
      num_adc_samples,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

void MatchedFilter_fft_backward_realimag_cuda(
    const at::Tensor grad_output_real,
    const at::Tensor grad_output_imag,
    const at::Tensor real,
    const at::Tensor imag,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    at::Tensor grad_real,
    at::Tensor grad_imag,
    const at::Tensor voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  LaunchMatchedFilterFftBackwardKernel(
      grad_output_real,
      grad_output_imag,
      real,
      imag,
      tx_pos,
      rx_pos,
      grad_real,
      grad_imag,
      voxel,
      total_ant_size,
      num_range_bins,
      fft_spacing,
      adc_sample_rate,
      chirp_slope_scaled,
      num_voxels);
}

}  // namespace MatchedFilter
