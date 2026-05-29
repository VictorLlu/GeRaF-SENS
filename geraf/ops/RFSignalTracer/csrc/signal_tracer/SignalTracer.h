// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include <torch/types.h>

namespace SignalTracer {

void BilinearRaysample_forward_cuda(
    const at::Tensor& normals,
    const at::Tensor& sigmas,
    const at::Tensor& z_vals,
    const at::Tensor& target_z_vals,
    at::Tensor& output_normals,
    at::Tensor& output_sigmas,
    at::Tensor& output_inbounds,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples);

void BilinearRaysample_backward_cuda(
    const at::Tensor& grad_output_normals,
    const at::Tensor& grad_output_sigmas,
    const at::Tensor& z_vals,
    const at::Tensor& target_z_vals,
    at::Tensor& grad_normals,
    at::Tensor& grad_sigmas,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples);

void SignalTracer_forward_cuda(
    const at::Tensor normals,
    const at::Tensor sigmas,
    const at::Tensor points,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    const at::Tensor inbounds,
    at::Tensor output_real,
    at::Tensor output_imag,
    int num_adc_samples,
    float adc_sample_rate,
    float start_frequency,
    float chirp_slope_scaled,
    int total_ant_size,
    int total_point_size);

void SignalTracer_backward_normal_sigma_cuda(
    const at::Tensor grad_output_real,
    const at::Tensor grad_output_imag,
    const at::Tensor normals,
    const at::Tensor sigmas,
    const at::Tensor points,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    const at::Tensor inbounds,
    at::Tensor grad_normals,
    at::Tensor grad_sigmas,
    int num_adc_samples,
    float adc_sample_rate,
    float start_frequency,
    float chirp_slope_scaled,
    int total_ant_size,
    int total_point_size);

void BilinearRaysample_forward(
    const at::Tensor& normals,
    const at::Tensor& sigmas,
    const at::Tensor& z_vals,
    const at::Tensor& target_z_vals,
    at::Tensor& output_normals,
    at::Tensor& output_sigmas,
    at::Tensor& output_inbounds,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples);

void BilinearRaysample_backward(
    const at::Tensor& grad_output_normals,
    const at::Tensor& grad_output_sigmas,
    const at::Tensor& z_vals,
    const at::Tensor& target_z_vals,
    at::Tensor& grad_normals,
    at::Tensor& grad_sigmas,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples);

void SignalTracer_forward(
    const at::Tensor normals,
    const at::Tensor sigmas,
    const at::Tensor points,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    const at::Tensor inbounds,
    at::Tensor output_real,
    at::Tensor output_imag,
    int num_adc_samples,
    float adc_sample_rate,
    float start_frequency,
    float chirp_slope_scaled,
    int total_ant_size,
    int total_point_size);

void SignalTracer_backward_normal_sigma(
    const at::Tensor grad_output_real,
    const at::Tensor grad_output_imag,
    const at::Tensor normals,
    const at::Tensor sigmas,
    const at::Tensor points,
    const at::Tensor tx_pos,
    const at::Tensor rx_pos,
    const at::Tensor inbounds,
    at::Tensor grad_normals,
    at::Tensor grad_sigmas,
    int num_adc_samples,
    float adc_sample_rate,
    float start_frequency,
    float chirp_slope_scaled,
    int total_ant_size,
    int total_point_size);

}  // namespace SignalTracer
