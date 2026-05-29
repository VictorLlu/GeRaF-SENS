// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <torch/types.h>

#include "SignalTracer.h"

namespace SignalTracer {
namespace {

void CheckCudaTensor(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), "`", name, "` must be a CUDA tensor.");
  TORCH_CHECK(tensor.is_contiguous(), "`", name, "` must be contiguous.");
}

void CheckBilinearForwardInputs(
    const at::Tensor& normals,
    const at::Tensor& sigmas,
    const at::Tensor& z_vals,
    const at::Tensor& target_z_vals,
    const at::Tensor& output_normals,
    const at::Tensor& output_sigmas,
    const at::Tensor& output_inbounds,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples) {
  CheckCudaTensor(normals, "normals");
  CheckCudaTensor(sigmas, "sigmas");
  CheckCudaTensor(z_vals, "z_vals");
  CheckCudaTensor(target_z_vals, "target_z_vals");
  CheckCudaTensor(output_normals, "output_normals");
  CheckCudaTensor(output_sigmas, "output_sigmas");
  CheckCudaTensor(output_inbounds, "output_inbounds");

  TORCH_CHECK(normals.dim() == 3, "`normals` must have shape [R, S, 3].");
  TORCH_CHECK(sigmas.dim() == 3, "`sigmas` must have shape [A, R, S].");
  TORCH_CHECK(z_vals.dim() == 2, "`z_vals` must have shape [R, S].");
  TORCH_CHECK(target_z_vals.dim() == 2, "`target_z_vals` must have shape [R, T].");
  TORCH_CHECK(output_normals.dim() == 3, "`output_normals` must have shape [R, T, 3].");
  TORCH_CHECK(output_sigmas.dim() == 3, "`output_sigmas` must have shape [A, R, T].");
  TORCH_CHECK(output_inbounds.dim() == 2, "`output_inbounds` must have shape [R, T].");
  TORCH_CHECK(output_inbounds.scalar_type() == at::kBool, "`output_inbounds` must be bool.");

  TORCH_CHECK(normals.size(0) == num_rays && normals.size(1) == num_samples && normals.size(2) == 3,
              "`normals` shape mismatch.");
  TORCH_CHECK(sigmas.size(0) == num_antennas && sigmas.size(1) == num_rays && sigmas.size(2) == num_samples,
              "`sigmas` shape mismatch.");
  TORCH_CHECK(z_vals.size(0) == num_rays && z_vals.size(1) == num_samples, "`z_vals` shape mismatch.");
  TORCH_CHECK(target_z_vals.size(0) == num_rays && target_z_vals.size(1) == num_target_samples,
              "`target_z_vals` shape mismatch.");
  TORCH_CHECK(output_normals.size(0) == num_rays && output_normals.size(1) == num_target_samples &&
                  output_normals.size(2) == 3,
              "`output_normals` shape mismatch.");
  TORCH_CHECK(output_sigmas.size(0) == num_antennas && output_sigmas.size(1) == num_rays &&
                  output_sigmas.size(2) == num_target_samples,
              "`output_sigmas` shape mismatch.");
  TORCH_CHECK(output_inbounds.size(0) == num_rays && output_inbounds.size(1) == num_target_samples,
              "`output_inbounds` shape mismatch.");
}

void CheckBilinearBackwardInputs(
    const at::Tensor& grad_output_normals,
    const at::Tensor& grad_output_sigmas,
    const at::Tensor& z_vals,
    const at::Tensor& target_z_vals,
    const at::Tensor& grad_normals,
    const at::Tensor& grad_sigmas,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples) {
  CheckCudaTensor(grad_output_normals, "grad_output_normals");
  CheckCudaTensor(grad_output_sigmas, "grad_output_sigmas");
  CheckCudaTensor(z_vals, "z_vals");
  CheckCudaTensor(target_z_vals, "target_z_vals");
  CheckCudaTensor(grad_normals, "grad_normals");
  CheckCudaTensor(grad_sigmas, "grad_sigmas");

  TORCH_CHECK(grad_output_normals.dim() == 3, "`grad_output_normals` must have shape [R, T, 3].");
  TORCH_CHECK(grad_output_sigmas.dim() == 3, "`grad_output_sigmas` must have shape [A, R, T].");
  TORCH_CHECK(grad_normals.dim() == 3, "`grad_normals` must have shape [R, S, 3].");
  TORCH_CHECK(grad_sigmas.dim() == 3, "`grad_sigmas` must have shape [A, R, S].");

  TORCH_CHECK(grad_output_normals.size(0) == num_rays && grad_output_normals.size(1) == num_target_samples &&
                  grad_output_normals.size(2) == 3,
              "`grad_output_normals` shape mismatch.");
  TORCH_CHECK(grad_output_sigmas.size(0) == num_antennas && grad_output_sigmas.size(1) == num_rays &&
                  grad_output_sigmas.size(2) == num_target_samples,
              "`grad_output_sigmas` shape mismatch.");
  TORCH_CHECK(z_vals.size(0) == num_rays && z_vals.size(1) == num_samples, "`z_vals` shape mismatch.");
  TORCH_CHECK(target_z_vals.size(0) == num_rays && target_z_vals.size(1) == num_target_samples,
              "`target_z_vals` shape mismatch.");
  TORCH_CHECK(grad_normals.size(0) == num_rays && grad_normals.size(1) == num_samples &&
                  grad_normals.size(2) == 3,
              "`grad_normals` shape mismatch.");
  TORCH_CHECK(grad_sigmas.size(0) == num_antennas && grad_sigmas.size(1) == num_rays &&
                  grad_sigmas.size(2) == num_samples,
              "`grad_sigmas` shape mismatch.");
}

void CheckSignalTracerForwardInputs(
    const at::Tensor& normals,
    const at::Tensor& sigmas,
    const at::Tensor& points,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    const at::Tensor& inbounds,
    const at::Tensor& output_real,
    const at::Tensor& output_imag,
    int num_adc_samples,
    int total_ant_size,
    int total_point_size) {
  CheckCudaTensor(normals, "normals");
  CheckCudaTensor(sigmas, "sigmas");
  CheckCudaTensor(points, "points");
  CheckCudaTensor(tx_pos, "tx_pos");
  CheckCudaTensor(rx_pos, "rx_pos");
  CheckCudaTensor(inbounds, "inbounds");
  CheckCudaTensor(output_real, "output_real");
  CheckCudaTensor(output_imag, "output_imag");

  TORCH_CHECK(inbounds.scalar_type() == at::kBool, "`inbounds` must be bool.");
  TORCH_CHECK(normals.dim() == 2 && normals.size(1) == 3, "`normals` must have shape [P, 3].");
  TORCH_CHECK(points.dim() == 2 && points.size(1) == 3, "`points` must have shape [P, 3].");
  TORCH_CHECK(sigmas.dim() == 2, "`sigmas` must have shape [A, P].");
  TORCH_CHECK(tx_pos.dim() == 2 && tx_pos.size(1) == 3, "`tx_pos` must have shape [A, 3].");
  TORCH_CHECK(rx_pos.dim() == 2 && rx_pos.size(1) == 3, "`rx_pos` must have shape [A, 3].");
  TORCH_CHECK(inbounds.dim() == 1, "`inbounds` must have shape [P].");
  TORCH_CHECK(output_real.dim() == 2, "`output_real` must have shape [A, S].");
  TORCH_CHECK(output_imag.dim() == 2, "`output_imag` must have shape [A, S].");

  TORCH_CHECK(normals.size(0) == total_point_size, "`normals` point dimension mismatch.");
  TORCH_CHECK(points.size(0) == total_point_size, "`points` point dimension mismatch.");
  TORCH_CHECK(sigmas.size(0) == total_ant_size && sigmas.size(1) == total_point_size, "`sigmas` shape mismatch.");
  TORCH_CHECK(tx_pos.size(0) == total_ant_size, "`tx_pos` antenna dimension mismatch.");
  TORCH_CHECK(rx_pos.size(0) == total_ant_size, "`rx_pos` antenna dimension mismatch.");
  TORCH_CHECK(inbounds.size(0) == total_point_size, "`inbounds` point dimension mismatch.");
  TORCH_CHECK(output_real.size(0) == total_ant_size && output_real.size(1) == num_adc_samples,
              "`output_real` shape mismatch.");
  TORCH_CHECK(output_imag.size(0) == total_ant_size && output_imag.size(1) == num_adc_samples,
              "`output_imag` shape mismatch.");
}

void CheckSignalTracerBackwardInputs(
    const at::Tensor& grad_output_real,
    const at::Tensor& grad_output_imag,
    const at::Tensor& normals,
    const at::Tensor& sigmas,
    const at::Tensor& points,
    const at::Tensor& tx_pos,
    const at::Tensor& rx_pos,
    const at::Tensor& inbounds,
    const at::Tensor& grad_normals,
    const at::Tensor& grad_sigmas,
    int num_adc_samples,
    int total_ant_size,
    int total_point_size) {
  CheckSignalTracerForwardInputs(
      normals,
      sigmas,
      points,
      tx_pos,
      rx_pos,
      inbounds,
      grad_output_real,
      grad_output_imag,
      num_adc_samples,
      total_ant_size,
      total_point_size);
  CheckCudaTensor(grad_normals, "grad_normals");
  CheckCudaTensor(grad_sigmas, "grad_sigmas");

  TORCH_CHECK(grad_normals.dim() == 2 && grad_normals.size(1) == 3, "`grad_normals` must have shape [P, 3].");
  TORCH_CHECK(grad_sigmas.dim() == 2, "`grad_sigmas` must have shape [A, P].");
  TORCH_CHECK(grad_normals.size(0) == total_point_size, "`grad_normals` point dimension mismatch.");
  TORCH_CHECK(grad_sigmas.size(0) == total_ant_size && grad_sigmas.size(1) == total_point_size,
              "`grad_sigmas` shape mismatch.");
}

}  // namespace

void bilinear_raysample_forward_cuda(
    const at::Tensor normals,
    const at::Tensor sigmas,
    const at::Tensor z_vals,
    const at::Tensor target_z_vals,
    at::Tensor output_normals,
    at::Tensor output_sigmas,
    at::Tensor output_inbounds,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples);

void bilinear_raysample_backward_cuda(
    const at::Tensor grad_output_normals,
    const at::Tensor grad_output_sigmas,
    const at::Tensor z_vals,
    const at::Tensor target_z_vals,
    at::Tensor grad_normals,
    at::Tensor grad_sigmas,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples);

void signal_tracer_forward_cuda(
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

void signal_tracer_backward_normal_sigma_cuda(
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
    int num_target_samples) {
  CheckBilinearForwardInputs(
      normals,
      sigmas,
      z_vals,
      target_z_vals,
      output_normals,
      output_sigmas,
      output_inbounds,
      num_antennas,
      num_rays,
      num_samples,
      num_target_samples);
  BilinearRaysample_forward_cuda(
      normals,
      sigmas,
      z_vals,
      target_z_vals,
      output_normals,
      output_sigmas,
      output_inbounds,
      num_antennas,
      num_rays,
      num_samples,
      num_target_samples);
}

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
    int num_target_samples) {
  CheckBilinearBackwardInputs(
      grad_output_normals,
      grad_output_sigmas,
      z_vals,
      target_z_vals,
      grad_normals,
      grad_sigmas,
      num_antennas,
      num_rays,
      num_samples,
      num_target_samples);
  BilinearRaysample_backward_cuda(
      grad_output_normals,
      grad_output_sigmas,
      z_vals,
      target_z_vals,
      grad_normals,
      grad_sigmas,
      num_antennas,
      num_rays,
      num_samples,
      num_target_samples);
}

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
    int total_point_size) {
  CheckSignalTracerForwardInputs(
      normals,
      sigmas,
      points,
      tx_pos,
      rx_pos,
      inbounds,
      output_real,
      output_imag,
      num_adc_samples,
      total_ant_size,
      total_point_size);
  SignalTracer_forward_cuda(
      normals,
      sigmas,
      points,
      tx_pos,
      rx_pos,
      inbounds,
      output_real,
      output_imag,
      num_adc_samples,
      adc_sample_rate,
      start_frequency,
      chirp_slope_scaled,
      total_ant_size,
      total_point_size);
}

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
    int total_point_size) {
  CheckSignalTracerBackwardInputs(
      grad_output_real,
      grad_output_imag,
      normals,
      sigmas,
      points,
      tx_pos,
      rx_pos,
      inbounds,
      grad_normals,
      grad_sigmas,
      num_adc_samples,
      total_ant_size,
      total_point_size);
  SignalTracer_backward_normal_sigma_cuda(
      grad_output_real,
      grad_output_imag,
      normals,
      sigmas,
      points,
      tx_pos,
      rx_pos,
      inbounds,
      grad_normals,
      grad_sigmas,
      num_adc_samples,
      adc_sample_rate,
      start_frequency,
      chirp_slope_scaled,
      total_ant_size,
      total_point_size);
}

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
    int num_target_samples) {
  bilinear_raysample_forward_cuda(
      normals,
      sigmas,
      z_vals,
      target_z_vals,
      output_normals,
      output_sigmas,
      output_inbounds,
      num_antennas,
      num_rays,
      num_samples,
      num_target_samples);
}

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
    int num_target_samples) {
  bilinear_raysample_backward_cuda(
      grad_output_normals,
      grad_output_sigmas,
      z_vals,
      target_z_vals,
      grad_normals,
      grad_sigmas,
      num_antennas,
      num_rays,
      num_samples,
      num_target_samples);
}

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
    int total_point_size) {
  signal_tracer_forward_cuda(
      normals,
      sigmas,
      points,
      tx_pos,
      rx_pos,
      inbounds,
      output_real,
      output_imag,
      num_adc_samples,
      adc_sample_rate,
      start_frequency,
      chirp_slope_scaled,
      total_ant_size,
      total_point_size);
}

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
    int total_point_size) {
  signal_tracer_backward_normal_sigma_cuda(
      grad_output_real,
      grad_output_imag,
      normals,
      sigmas,
      points,
      tx_pos,
      rx_pos,
      inbounds,
      grad_normals,
      grad_sigmas,
      num_adc_samples,
      adc_sample_rate,
      start_frequency,
      chirp_slope_scaled,
      total_ant_size,
      total_point_size);
}

}  // namespace SignalTracer
