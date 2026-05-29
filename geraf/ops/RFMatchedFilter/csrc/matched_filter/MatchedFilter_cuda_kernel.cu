// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

namespace MatchedFilter {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kSpeedOfLightScaled = 2.9979f;
constexpr float kStartFrequencyScaled = 773.704f;
constexpr float kPathLengthBias = 0.15f;
constexpr float kPi = 3.14159265358979323846f;

template <typename scalar_t>
__device__ inline float LoadAsFloat(const scalar_t* ptr, int index) {
  return static_cast<float>(ptr[index]);
}

__device__ inline float Squared(float value) {
  return value * value;
}

template <typename scalar_t>
__device__ inline float ComputeBistaticDistance(
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    const scalar_t* voxel,
    int antenna_index,
    int voxel_index,
    int num_voxels) {
  const float tx_x = LoadAsFloat(tx_pos, antenna_index * 3 + 0);
  const float tx_y = LoadAsFloat(tx_pos, antenna_index * 3 + 1);
  const float tx_z = LoadAsFloat(tx_pos, antenna_index * 3 + 2);

  const float rx_x = LoadAsFloat(rx_pos, antenna_index * 3 + 0);
  const float rx_y = LoadAsFloat(rx_pos, antenna_index * 3 + 1);
  const float rx_z = LoadAsFloat(rx_pos, antenna_index * 3 + 2);

  const float voxel_x = LoadAsFloat(voxel, voxel_index);
  const float voxel_y = LoadAsFloat(voxel, num_voxels + voxel_index);
  const float voxel_z = LoadAsFloat(voxel, 2 * num_voxels + voxel_index);

  const float tx_distance =
      sqrtf(Squared(tx_x - voxel_x) + Squared(tx_y - voxel_y) + Squared(tx_z - voxel_z));
  const float rx_distance =
      sqrtf(Squared(rx_x - voxel_x) + Squared(rx_y - voxel_y) + Squared(rx_z - voxel_z));

  return tx_distance + rx_distance + kPathLengthBias;
}

__device__ inline float ComputeCarrierPhase(float travel_time_seconds) {
  return 2.0f * kPi * kStartFrequencyScaled * travel_time_seconds;
}

__device__ inline void ComplexRotate(float phase, float* cos_phase, float* neg_sin_phase) {
  *cos_phase = cosf(phase);
  *neg_sin_phase = -sinf(phase);
}

template <typename scalar_t>
__global__ void MatchedFilterForwardKernel(
    const scalar_t* real,
    const scalar_t* imag,
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    scalar_t* output_real,
    scalar_t* output_imag,
    const scalar_t* voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  const int voxel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel_index >= num_voxels) {
    return;
  }

  float accum_real = 0.0f;
  float accum_imag = 0.0f;

  for (int antenna_index = 0; antenna_index < total_ant_size; ++antenna_index) {
    const float distance =
        ComputeBistaticDistance(tx_pos, rx_pos, voxel, antenna_index, voxel_index, num_voxels);
    const float tau = distance / kSpeedOfLightScaled;
    const float base_phase = ComputeCarrierPhase(tau);

    #pragma unroll 4
    for (int sample_index = 0; sample_index < num_adc_samples; ++sample_index) {
      const float sample_time =
          static_cast<float>(sample_index) / adc_sample_rate;
      const float phase =
          base_phase + 2.0f * kPi * chirp_slope_scaled * sample_time * tau;

      float cos_phase = 0.0f;
      float neg_sin_phase = 0.0f;
      ComplexRotate(phase, &cos_phase, &neg_sin_phase);

      const int sample_offset = antenna_index * num_adc_samples + sample_index;
      const float sample_real = LoadAsFloat(real, sample_offset);
      const float sample_imag = LoadAsFloat(imag, sample_offset);

      accum_real += cos_phase * sample_real - neg_sin_phase * sample_imag;
      accum_imag += cos_phase * sample_imag + neg_sin_phase * sample_real;
    }
  }

  output_real[voxel_index] = static_cast<scalar_t>(accum_real / total_ant_size);
  output_imag[voxel_index] = static_cast<scalar_t>(accum_imag / total_ant_size);
}

template <typename scalar_t>
__global__ void MatchedFilterFftForwardKernel(
    const scalar_t* real,
    const scalar_t* imag,
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    scalar_t* output_real,
    scalar_t* output_imag,
    const scalar_t* voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float /*adc_sample_rate*/,
    float /*chirp_slope_scaled*/,
    int num_voxels) {
  const int voxel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel_index >= num_voxels) {
    return;
  }

  float accum_real = 0.0f;
  float accum_imag = 0.0f;

  for (int antenna_index = 0; antenna_index < total_ant_size; ++antenna_index) {
    const float distance =
        ComputeBistaticDistance(tx_pos, rx_pos, voxel, antenna_index, voxel_index, num_voxels);
    if (distance < 0.0f || distance > fft_spacing * num_range_bins) {
      continue;
    }

    const int range_bin = static_cast<int>(floorf(distance / fft_spacing / 2.0f));
    if (range_bin < 0 || range_bin >= num_range_bins) {
      continue;
    }

    const float tau = distance / kSpeedOfLightScaled;
    const float phase = ComputeCarrierPhase(tau);

    float cos_phase = 0.0f;
    float neg_sin_phase = 0.0f;
    ComplexRotate(phase, &cos_phase, &neg_sin_phase);

    const int sample_offset = antenna_index * num_range_bins + range_bin;
    const float sample_real = LoadAsFloat(real, sample_offset);
    const float sample_imag = LoadAsFloat(imag, sample_offset);

    accum_real += cos_phase * sample_real - neg_sin_phase * sample_imag;
    accum_imag += cos_phase * sample_imag + neg_sin_phase * sample_real;
  }

  output_real[voxel_index] = static_cast<scalar_t>(accum_real / total_ant_size);
  output_imag[voxel_index] = static_cast<scalar_t>(accum_imag / total_ant_size);
}

template <typename scalar_t>
__global__ void MatchedFilterBackwardKernel(
    const scalar_t* grad_output_real,
    const scalar_t* grad_output_imag,
    const scalar_t* real,
    const scalar_t* imag,
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    scalar_t* grad_real,
    scalar_t* grad_imag,
    const scalar_t* voxel,
    int total_ant_size,
    int num_adc_samples,
    float adc_sample_rate,
    float chirp_slope_scaled,
    int num_voxels) {
  const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int antenna_index = linear_index / num_adc_samples;
  const int sample_index = linear_index % num_adc_samples;

  if (antenna_index >= total_ant_size || sample_index >= num_adc_samples) {
    return;
  }

  float accum_grad_real = 0.0f;
  float accum_grad_imag = 0.0f;
  const float sample_time = static_cast<float>(sample_index) / adc_sample_rate;

  for (int voxel_index = 0; voxel_index < num_voxels; ++voxel_index) {
    const float distance =
        ComputeBistaticDistance(tx_pos, rx_pos, voxel, antenna_index, voxel_index, num_voxels);
    const float tau = distance / kSpeedOfLightScaled;
    const float phase =
        ComputeCarrierPhase(tau) + 2.0f * kPi * chirp_slope_scaled * sample_time * tau;

    float cos_phase = 0.0f;
    float neg_sin_phase = 0.0f;
    ComplexRotate(phase, &cos_phase, &neg_sin_phase);

    const float grad_real_out = LoadAsFloat(grad_output_real, voxel_index);
    const float grad_imag_out = LoadAsFloat(grad_output_imag, voxel_index);

    accum_grad_real += cos_phase * grad_real_out + neg_sin_phase * grad_imag_out;
    accum_grad_imag += (-neg_sin_phase) * grad_real_out + cos_phase * grad_imag_out;
  }

  const int sample_offset = antenna_index * num_adc_samples + sample_index;
  grad_real[sample_offset] = static_cast<scalar_t>(accum_grad_real / total_ant_size);
  grad_imag[sample_offset] = static_cast<scalar_t>(accum_grad_imag / total_ant_size);
}

template <typename scalar_t>
__global__ void MatchedFilterFftBackwardKernel(
    const scalar_t* grad_output_real,
    const scalar_t* grad_output_imag,
    const scalar_t* real,
    const scalar_t* imag,
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    scalar_t* grad_real,
    scalar_t* grad_imag,
    const scalar_t* voxel,
    int total_ant_size,
    int num_range_bins,
    float fft_spacing,
    float /*adc_sample_rate*/,
    float /*chirp_slope_scaled*/,
    int num_voxels) {
  const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int antenna_index = linear_index / num_range_bins;
  const int range_bin = linear_index % num_range_bins;

  if (antenna_index >= total_ant_size || range_bin >= num_range_bins) {
    return;
  }

  float accum_grad_real = 0.0f;
  float accum_grad_imag = 0.0f;

  for (int voxel_index = 0; voxel_index < num_voxels; ++voxel_index) {
    const float distance =
        ComputeBistaticDistance(tx_pos, rx_pos, voxel, antenna_index, voxel_index, num_voxels);
    if (distance < 0.0f || distance > fft_spacing * num_range_bins) {
      continue;
    }

    const int voxel_range_bin = static_cast<int>(floorf(distance / fft_spacing / 2.0f));
    if (voxel_range_bin != range_bin) {
      continue;
    }

    const float tau = distance / kSpeedOfLightScaled;
    const float phase = ComputeCarrierPhase(tau);

    float cos_phase = 0.0f;
    float neg_sin_phase = 0.0f;
    ComplexRotate(phase, &cos_phase, &neg_sin_phase);

    const float grad_real_out = LoadAsFloat(grad_output_real, voxel_index);
    const float grad_imag_out = LoadAsFloat(grad_output_imag, voxel_index);

    accum_grad_real += cos_phase * grad_real_out + neg_sin_phase * grad_imag_out;
    accum_grad_imag += (-neg_sin_phase) * grad_real_out + cos_phase * grad_imag_out;
  }

  const int sample_offset = antenna_index * num_range_bins + range_bin;
  grad_real[sample_offset] = static_cast<scalar_t>(accum_grad_real / total_ant_size);
  grad_imag[sample_offset] = static_cast<scalar_t>(accum_grad_imag / total_ant_size);
}

struct ForwardLaunch {
  const at::Tensor& real;
  const at::Tensor& imag;
  const at::Tensor& tx_pos;
  const at::Tensor& rx_pos;
  at::Tensor& output_real;
  at::Tensor& output_imag;
  const at::Tensor& voxel;
  int total_ant_size;
  int num_adc_samples;
  float adc_sample_rate;
  float chirp_slope_scaled;
  int num_voxels;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int blocks = (num_voxels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    MatchedFilterForwardKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        real.data_ptr<scalar_t>(),
        imag.data_ptr<scalar_t>(),
        tx_pos.data_ptr<scalar_t>(),
        rx_pos.data_ptr<scalar_t>(),
        output_real.data_ptr<scalar_t>(),
        output_imag.data_ptr<scalar_t>(),
        voxel.data_ptr<scalar_t>(),
        total_ant_size,
        num_adc_samples,
        adc_sample_rate,
        chirp_slope_scaled,
        num_voxels);
  }
};

struct FftForwardLaunch {
  const at::Tensor& real;
  const at::Tensor& imag;
  const at::Tensor& tx_pos;
  const at::Tensor& rx_pos;
  at::Tensor& output_real;
  at::Tensor& output_imag;
  const at::Tensor& voxel;
  int total_ant_size;
  int num_range_bins;
  float fft_spacing;
  float adc_sample_rate;
  float chirp_slope_scaled;
  int num_voxels;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int blocks = (num_voxels + kThreadsPerBlock - 1) / kThreadsPerBlock;
    MatchedFilterFftForwardKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        real.data_ptr<scalar_t>(),
        imag.data_ptr<scalar_t>(),
        tx_pos.data_ptr<scalar_t>(),
        rx_pos.data_ptr<scalar_t>(),
        output_real.data_ptr<scalar_t>(),
        output_imag.data_ptr<scalar_t>(),
        voxel.data_ptr<scalar_t>(),
        total_ant_size,
        num_range_bins,
        fft_spacing,
        adc_sample_rate,
        chirp_slope_scaled,
        num_voxels);
  }
};

struct BackwardLaunch {
  const at::Tensor& grad_output_real;
  const at::Tensor& grad_output_imag;
  const at::Tensor& real;
  const at::Tensor& imag;
  const at::Tensor& tx_pos;
  const at::Tensor& rx_pos;
  at::Tensor& grad_real;
  at::Tensor& grad_imag;
  const at::Tensor& voxel;
  int total_ant_size;
  int num_adc_samples;
  float adc_sample_rate;
  float chirp_slope_scaled;
  int num_voxels;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int num_threads = total_ant_size * num_adc_samples;
    const int blocks = (num_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;
    MatchedFilterBackwardKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        grad_output_real.data_ptr<scalar_t>(),
        grad_output_imag.data_ptr<scalar_t>(),
        real.data_ptr<scalar_t>(),
        imag.data_ptr<scalar_t>(),
        tx_pos.data_ptr<scalar_t>(),
        rx_pos.data_ptr<scalar_t>(),
        grad_real.data_ptr<scalar_t>(),
        grad_imag.data_ptr<scalar_t>(),
        voxel.data_ptr<scalar_t>(),
        total_ant_size,
        num_adc_samples,
        adc_sample_rate,
        chirp_slope_scaled,
        num_voxels);
  }
};

struct FftBackwardLaunch {
  const at::Tensor& grad_output_real;
  const at::Tensor& grad_output_imag;
  const at::Tensor& real;
  const at::Tensor& imag;
  const at::Tensor& tx_pos;
  const at::Tensor& rx_pos;
  at::Tensor& grad_real;
  at::Tensor& grad_imag;
  const at::Tensor& voxel;
  int total_ant_size;
  int num_range_bins;
  float fft_spacing;
  float adc_sample_rate;
  float chirp_slope_scaled;
  int num_voxels;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int num_threads = total_ant_size * num_range_bins;
    const int blocks = (num_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;
    MatchedFilterFftBackwardKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        grad_output_real.data_ptr<scalar_t>(),
        grad_output_imag.data_ptr<scalar_t>(),
        real.data_ptr<scalar_t>(),
        imag.data_ptr<scalar_t>(),
        tx_pos.data_ptr<scalar_t>(),
        rx_pos.data_ptr<scalar_t>(),
        grad_real.data_ptr<scalar_t>(),
        grad_imag.data_ptr<scalar_t>(),
        voxel.data_ptr<scalar_t>(),
        total_ant_size,
        num_range_bins,
        fft_spacing,
        adc_sample_rate,
        chirp_slope_scaled,
        num_voxels);
  }
};

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
    int num_voxels) {
  at::cuda::CUDAGuard device_guard(real.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(real.scalar_type(), "matched_filter_forward_cuda", [&] {
    ForwardLaunch{
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
        num_voxels}
        .template operator()<scalar_t>(stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int num_voxels) {
  at::cuda::CUDAGuard device_guard(real.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(real.scalar_type(), "matched_filter_fft_forward_cuda", [&] {
    FftForwardLaunch{
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
        num_voxels}
        .template operator()<scalar_t>(stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int num_voxels) {
  at::cuda::CUDAGuard device_guard(real.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(real.scalar_type(), "matched_filter_backward_realimag_cuda", [&] {
    BackwardLaunch{
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
        num_voxels}
        .template operator()<scalar_t>(stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int num_voxels) {
  at::cuda::CUDAGuard device_guard(real.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      real.scalar_type(), "matched_filter_fft_backward_realimag_cuda", [&] {
        FftBackwardLaunch{
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
            num_voxels}
            .template operator()<scalar_t>(stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace MatchedFilter
