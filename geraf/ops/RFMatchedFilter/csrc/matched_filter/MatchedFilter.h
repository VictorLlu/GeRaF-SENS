// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include <torch/types.h>

namespace MatchedFilter {

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
    int num_voxels);

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
    int num_voxels);

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
    int num_voxels);

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
    int num_voxels);

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
    int num_voxels);

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
    int num_voxels);

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
    int num_voxels);

}  // namespace MatchedFilter
