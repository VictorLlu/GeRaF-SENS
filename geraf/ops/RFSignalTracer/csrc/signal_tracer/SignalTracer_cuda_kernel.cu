// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

namespace SignalTracer {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr float kSpeedOfLightScaled = 2.9979f;
constexpr float kPathLengthBias = 0.15f;
constexpr float kEpsilon = 1.0e-7f;
constexpr float kPi = 3.14159265358979323846f;

template <typename scalar_t>
__device__ inline float LoadAsFloat(const scalar_t* ptr, int index) {
  return static_cast<float>(ptr[index]);
}

__device__ inline float Squared(float value) {
  return value * value;
}

template <typename scalar_t>
__device__ inline float Dot3(const scalar_t* lhs, const scalar_t* rhs) {
  return static_cast<float>(lhs[0]) * static_cast<float>(rhs[0]) +
         static_cast<float>(lhs[1]) * static_cast<float>(rhs[1]) +
         static_cast<float>(lhs[2]) * static_cast<float>(rhs[2]);
}

template <typename scalar_t>
__device__ inline float Norm3(const scalar_t* value) {
  return sqrtf(Dot3(value, value));
}

template <typename scalar_t>
__device__ inline void Normalize3(const scalar_t* input, scalar_t* output) {
  const float norm = Norm3(input);
  const float inv_norm = 1.0f / (norm + kEpsilon);
  output[0] = static_cast<scalar_t>(static_cast<float>(input[0]) * inv_norm);
  output[1] = static_cast<scalar_t>(static_cast<float>(input[1]) * inv_norm);
  output[2] = static_cast<scalar_t>(static_cast<float>(input[2]) * inv_norm);
}

template <typename scalar_t>
__device__ inline void Normalize3InPlace(scalar_t* value) {
  scalar_t normalized[3];
  Normalize3(value, normalized);
  value[0] = normalized[0];
  value[1] = normalized[1];
  value[2] = normalized[2];
}

template <typename scalar_t>
__device__ inline float ComputeTwoPointDistance(const scalar_t* lhs, const scalar_t* rhs) {
  return sqrtf(Squared(static_cast<float>(lhs[0]) - static_cast<float>(rhs[0])) +
               Squared(static_cast<float>(lhs[1]) - static_cast<float>(rhs[1])) +
               Squared(static_cast<float>(lhs[2]) - static_cast<float>(rhs[2])));
}

template <typename scalar_t>
__device__ inline float ComputePathDistance(
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    const scalar_t* point) {
  return ComputeTwoPointDistance(tx_pos, point) + ComputeTwoPointDistance(rx_pos, point);
}

template <typename scalar_t>
__device__ inline float ComputeTransmitPower(float propagation_distance, scalar_t sigma, float diff) {
  return static_cast<float>(sigma) * diff / Squared(propagation_distance);
}

template <typename scalar_t>
__device__ inline void ComputeReflectedDirection(
    const scalar_t* incident_dir,
    const scalar_t* normal,
    scalar_t* reflected_dir) {
  const float incident_dot_normal = Dot3(incident_dir, normal);
  for (int axis = 0; axis < 3; ++axis) {
    reflected_dir[axis] = static_cast<scalar_t>(
        static_cast<float>(incident_dir[axis]) -
        2.0f * incident_dot_normal * static_cast<float>(normal[axis]));
  }
}

template <typename scalar_t>
__device__ inline void ComputeReflectedDirectionNormalJacobian(
    const scalar_t* incident_dir,
    const scalar_t* normal,
    scalar_t* jacobian) {
  const float incident_dot_normal = Dot3(incident_dir, normal);
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      const float shared_term = -2.0f * static_cast<float>(normal[row]) *
                                static_cast<float>(incident_dir[col]);
      jacobian[row * 3 + col] =
          static_cast<scalar_t>(row == col ? shared_term - 2.0f * incident_dot_normal
                                           : shared_term);
    }
  }
}

template <typename scalar_t>
__device__ inline void ComputeNormalizedVectorJacobian(
    const scalar_t* input_jacobian,
    const scalar_t* normalized_value,
    float value_norm,
    scalar_t* output_jacobian) {
  const float x = static_cast<float>(normalized_value[0]);
  const float y = static_cast<float>(normalized_value[1]);
  const float z = static_cast<float>(normalized_value[2]);
  const float inv_norm = 1.0f / (value_norm + kEpsilon);

  output_jacobian[0] = static_cast<scalar_t>(((1.0f - x * x) * input_jacobian[0] +
                                              (-x * y) * input_jacobian[3] +
                                              (-x * z) * input_jacobian[6]) *
                                             inv_norm);
  output_jacobian[1] = static_cast<scalar_t>(((1.0f - x * x) * input_jacobian[1] +
                                              (-x * y) * input_jacobian[4] +
                                              (-x * z) * input_jacobian[7]) *
                                             inv_norm);
  output_jacobian[2] = static_cast<scalar_t>(((1.0f - x * x) * input_jacobian[2] +
                                              (-x * y) * input_jacobian[5] +
                                              (-x * z) * input_jacobian[8]) *
                                             inv_norm);
  output_jacobian[3] = static_cast<scalar_t>(((-y * x) * input_jacobian[0] +
                                              (1.0f - y * y) * input_jacobian[3] +
                                              (-y * z) * input_jacobian[6]) *
                                             inv_norm);
  output_jacobian[4] = static_cast<scalar_t>(((-y * x) * input_jacobian[1] +
                                              (1.0f - y * y) * input_jacobian[4] +
                                              (-y * z) * input_jacobian[7]) *
                                             inv_norm);
  output_jacobian[5] = static_cast<scalar_t>(((-y * x) * input_jacobian[2] +
                                              (1.0f - y * y) * input_jacobian[5] +
                                              (-y * z) * input_jacobian[8]) *
                                             inv_norm);
  output_jacobian[6] = static_cast<scalar_t>(((-z * x) * input_jacobian[0] +
                                              (-z * y) * input_jacobian[3] +
                                              (1.0f - z * z) * input_jacobian[6]) *
                                             inv_norm);
  output_jacobian[7] = static_cast<scalar_t>(((-z * x) * input_jacobian[1] +
                                              (-z * y) * input_jacobian[4] +
                                              (1.0f - z * z) * input_jacobian[7]) *
                                             inv_norm);
  output_jacobian[8] = static_cast<scalar_t>(((-z * x) * input_jacobian[2] +
                                              (-z * y) * input_jacobian[5] +
                                              (1.0f - z * z) * input_jacobian[8]) *
                                             inv_norm);
}

template <typename scalar_t>
__device__ inline void ProjectDirectionJacobianOntoReturnVector(
    const scalar_t* normalized_jacobian,
    const scalar_t* return_dir,
    scalar_t* grad_diff) {
  for (int axis = 0; axis < 3; ++axis) {
    float value = 0.0f;
    for (int component = 0; component < 3; ++component) {
      value += static_cast<float>(normalized_jacobian[component * 3 + axis]) *
               static_cast<float>(return_dir[component]);
    }
    grad_diff[axis] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
__device__ inline void ComputeSpecularGeometry(
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    const scalar_t* normal,
    const scalar_t* point,
    scalar_t* specular_alignment,
    scalar_t* incident_dot_normal,
    scalar_t* incident_dir,
    scalar_t* reflected_dir_raw,
    scalar_t* reflected_dir,
    scalar_t* return_dir) {
  for (int axis = 0; axis < 3; ++axis) {
    incident_dir[axis] = point[axis] - tx_pos[axis];
    return_dir[axis] = rx_pos[axis] - point[axis];
  }

  Normalize3InPlace(incident_dir);
  *incident_dot_normal = static_cast<scalar_t>(Dot3(incident_dir, normal));

  ComputeReflectedDirection(incident_dir, normal, reflected_dir_raw);
  Normalize3(reflected_dir_raw, reflected_dir);
  Normalize3InPlace(return_dir);

  *specular_alignment = static_cast<scalar_t>(Dot3(return_dir, reflected_dir));
}

template <typename scalar_t>
__global__ void BilinearRaySampleForwardKernel(
    const scalar_t* normals,
    const scalar_t* sigmas,
    const scalar_t* z_vals,
    const scalar_t* target_z_vals,
    scalar_t* output_normals,
    scalar_t* output_sigmas,
    bool* output_inbounds,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples) {
  const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int antenna_index = linear_index / num_rays;
  const int ray_index = linear_index % num_rays;

  if (antenna_index >= num_antennas || ray_index >= num_rays) {
    return;
  }

  const scalar_t* ray_z_vals = z_vals + ray_index * num_samples;
  const scalar_t* ray_target_z_vals = target_z_vals + ray_index * num_target_samples;
  const scalar_t* ray_normals = normals + ray_index * num_samples * 3;
  const scalar_t* ray_sigmas =
      sigmas + antenna_index * num_rays * num_samples + ray_index * num_samples;

  scalar_t* ray_output_normals = output_normals + ray_index * num_target_samples * 3;
  scalar_t* ray_output_sigmas =
      output_sigmas + antenna_index * num_rays * num_target_samples + ray_index * num_target_samples;
  bool* ray_output_inbounds = output_inbounds + ray_index * num_target_samples;

  for (int target_index = 0; target_index < num_target_samples; ++target_index) {
    const scalar_t target_z = ray_target_z_vals[target_index];

    int lower_index = 0;
    int upper_index = num_samples - 1;
    while (lower_index < upper_index - 1) {
      const int mid_index = (lower_index + upper_index) / 2;
      if (ray_z_vals[mid_index] <= target_z) {
        lower_index = mid_index;
      } else {
        upper_index = mid_index;
      }
    }

    const scalar_t z0 = ray_z_vals[lower_index];
    const scalar_t z1 = ray_z_vals[lower_index + 1];
    const bool out_of_bounds = (target_z < z0) || (target_z > z1);

    if (antenna_index == 0) {
      ray_output_inbounds[target_index] = !out_of_bounds;
    }

    if (out_of_bounds) {
      ray_output_sigmas[target_index] = static_cast<scalar_t>(0);
      if (antenna_index == 0) {
        ray_output_normals[target_index * 3 + 0] = static_cast<scalar_t>(0);
        ray_output_normals[target_index * 3 + 1] = static_cast<scalar_t>(0);
        ray_output_normals[target_index * 3 + 2] = static_cast<scalar_t>(0);
      }
      continue;
    }

    const scalar_t weight_upper =
        (target_z - z0) / max(z1 - z0, static_cast<scalar_t>(1.0e-6f));
    const scalar_t weight_lower = static_cast<scalar_t>(1) - weight_upper;

    ray_output_sigmas[target_index] = weight_lower * ray_sigmas[lower_index] +
                                      weight_upper * ray_sigmas[lower_index + 1];

    if (antenna_index == 0) {
      for (int axis = 0; axis < 3; ++axis) {
        const scalar_t normal_lower = ray_normals[lower_index * 3 + axis];
        const scalar_t normal_upper = ray_normals[(lower_index + 1) * 3 + axis];
        ray_output_normals[target_index * 3 + axis] =
            weight_lower * normal_lower + weight_upper * normal_upper;
      }
    }
  }
}

template <typename scalar_t>
__global__ void BilinearRaySampleBackwardKernel(
    const scalar_t* grad_output_normals,
    const scalar_t* grad_output_sigmas,
    const scalar_t* z_vals,
    const scalar_t* target_z_vals,
    scalar_t* grad_normals,
    scalar_t* grad_sigmas,
    int num_antennas,
    int num_rays,
    int num_samples,
    int num_target_samples) {
  const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int antenna_index = linear_index / num_rays;
  const int ray_index = linear_index % num_rays;

  if (antenna_index >= num_antennas || ray_index >= num_rays) {
    return;
  }

  const scalar_t* ray_z_vals = z_vals + ray_index * num_samples;
  const scalar_t* ray_target_z_vals = target_z_vals + ray_index * num_target_samples;
  const scalar_t* ray_grad_output_normals =
      grad_output_normals + ray_index * num_target_samples * 3;
  const scalar_t* ray_grad_output_sigmas =
      grad_output_sigmas + antenna_index * num_rays * num_target_samples +
      ray_index * num_target_samples;

  scalar_t* ray_grad_normals = grad_normals + ray_index * num_samples * 3;
  scalar_t* ray_grad_sigmas =
      grad_sigmas + antenna_index * num_rays * num_samples + ray_index * num_samples;

  for (int target_index = 0; target_index < num_target_samples; ++target_index) {
    const scalar_t target_z = ray_target_z_vals[target_index];

    int lower_index = 0;
    int upper_index = num_samples - 1;
    while (lower_index < upper_index - 1) {
      const int mid_index = (lower_index + upper_index) / 2;
      if (ray_z_vals[mid_index] <= target_z) {
        lower_index = mid_index;
      } else {
        upper_index = mid_index;
      }
    }

    const scalar_t z0 = ray_z_vals[lower_index];
    const scalar_t z1 = ray_z_vals[lower_index + 1];
    if (target_z < z0 || target_z > z1) {
      continue;
    }

    const scalar_t weight_upper =
        (target_z - z0) / max(z1 - z0, static_cast<scalar_t>(1.0e-6f));
    const scalar_t weight_lower = static_cast<scalar_t>(1) - weight_upper;

    const scalar_t sigma_grad = ray_grad_output_sigmas[target_index];
    atomicAdd(ray_grad_sigmas + lower_index, weight_lower * sigma_grad);
    atomicAdd(ray_grad_sigmas + lower_index + 1, weight_upper * sigma_grad);

    if (antenna_index == 0) {
      for (int axis = 0; axis < 3; ++axis) {
        const scalar_t normal_grad = ray_grad_output_normals[target_index * 3 + axis];
        atomicAdd(ray_grad_normals + lower_index * 3 + axis, weight_lower * normal_grad);
        atomicAdd(ray_grad_normals + (lower_index + 1) * 3 + axis, weight_upper * normal_grad);
      }
    }
  }
}

template <typename scalar_t>
__global__ void SignalTracerForwardKernel(
    const scalar_t* normals,
    const scalar_t* sigmas,
    const scalar_t* points,
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    const bool* inbounds,
    scalar_t* output_real,
    scalar_t* output_imag,
    int num_adc_samples,
    float adc_sample_rate,
    float start_frequency,
    float chirp_slope_scaled,
    int total_ant_size,
    int total_point_size) {
  const int linear_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int antenna_index = linear_index / num_adc_samples;
  const int sample_index = linear_index % num_adc_samples;

  if (antenna_index >= total_ant_size || sample_index >= num_adc_samples) {
    return;
  }

  const float sample_time = static_cast<float>(sample_index) / adc_sample_rate;
  const float phase_base = 2.0f * kPi * start_frequency;
  const float phase_slope = 2.0f * kPi * chirp_slope_scaled;

  float accum_real = 0.0f;
  float accum_imag = 0.0f;

  scalar_t tx_position[3] = {
      tx_pos[antenna_index * 3 + 0],
      tx_pos[antenna_index * 3 + 1],
      tx_pos[antenna_index * 3 + 2],
  };
  scalar_t rx_position[3] = {
      rx_pos[antenna_index * 3 + 0],
      rx_pos[antenna_index * 3 + 1],
      rx_pos[antenna_index * 3 + 2],
  };

  for (int point_index = 0; point_index < total_point_size; ++point_index) {
    if (!inbounds[point_index]) {
      continue;
    }

    scalar_t normal[3] = {
        normals[point_index * 3 + 0],
        normals[point_index * 3 + 1],
        normals[point_index * 3 + 2],
    };
    scalar_t point[3] = {
        points[point_index * 3 + 0],
        points[point_index * 3 + 1],
        points[point_index * 3 + 2],
    };

    scalar_t specular_alignment = static_cast<scalar_t>(0);
    scalar_t incident_dot_normal = static_cast<scalar_t>(0);
    scalar_t incident_dir[3];
    scalar_t reflected_dir_raw[3];
    scalar_t reflected_dir[3];
    scalar_t return_dir[3];

    ComputeSpecularGeometry(
        tx_position,
        rx_position,
        normal,
        point,
        &specular_alignment,
        &incident_dot_normal,
        incident_dir,
        reflected_dir_raw,
        reflected_dir,
        return_dir);

    if (static_cast<float>(specular_alignment) < 1.0e-6f ||
        static_cast<float>(incident_dot_normal) > 0.0f) {
      continue;
    }

    float propagation_distance = ComputePathDistance(tx_position, rx_position, point);
    const float amplitude = ComputeTransmitPower(
        propagation_distance, sigmas[antenna_index * total_point_size + point_index],
        static_cast<float>(specular_alignment));

    propagation_distance += kPathLengthBias;
    const float tau = propagation_distance / kSpeedOfLightScaled;
    const float phase = (phase_base + phase_slope * sample_time) * tau;

    accum_real += amplitude * cosf(phase);
    accum_imag += amplitude * sinf(phase);
  }

  output_real[antenna_index * num_adc_samples + sample_index] =
      static_cast<scalar_t>(accum_real / total_point_size);
  output_imag[antenna_index * num_adc_samples + sample_index] =
      static_cast<scalar_t>(accum_imag / total_point_size);
}

template <typename scalar_t>
__global__ void SignalTracerBackwardNormalSigmaKernel(
    const scalar_t* grad_output_real,
    const scalar_t* grad_output_imag,
    const scalar_t* normals,
    const scalar_t* sigmas,
    const scalar_t* points,
    const scalar_t* tx_pos,
    const scalar_t* rx_pos,
    const bool* inbounds,
    scalar_t* grad_normals,
    scalar_t* grad_sigmas,
    int num_adc_samples,
    float adc_sample_rate,
    float start_frequency,
    float chirp_slope_scaled,
    int total_ant_size,
    int total_point_size) {
  const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= total_point_size || !inbounds[point_index]) {
    return;
  }

  const float phase_base = 2.0f * kPi * start_frequency;
  const float phase_slope = 2.0f * kPi * chirp_slope_scaled;

  float accum_grad_normal[3] = {0.0f, 0.0f, 0.0f};

  scalar_t normal[3] = {
      normals[point_index * 3 + 0],
      normals[point_index * 3 + 1],
      normals[point_index * 3 + 2],
  };
  scalar_t point[3] = {
      points[point_index * 3 + 0],
      points[point_index * 3 + 1],
      points[point_index * 3 + 2],
  };

  for (int antenna_index = 0; antenna_index < total_ant_size; ++antenna_index) {
    float accum_grad_sigma = 0.0f;

    scalar_t tx_position[3] = {
        tx_pos[antenna_index * 3 + 0],
        tx_pos[antenna_index * 3 + 1],
        tx_pos[antenna_index * 3 + 2],
    };
    scalar_t rx_position[3] = {
        rx_pos[antenna_index * 3 + 0],
        rx_pos[antenna_index * 3 + 1],
        rx_pos[antenna_index * 3 + 2],
    };

    scalar_t specular_alignment = static_cast<scalar_t>(0);
    scalar_t incident_dot_normal = static_cast<scalar_t>(0);
    scalar_t incident_dir[3];
    scalar_t reflected_dir_raw[3];
    scalar_t reflected_dir[3];
    scalar_t return_dir[3];

    ComputeSpecularGeometry(
        tx_position,
        rx_position,
        normal,
        point,
        &specular_alignment,
        &incident_dot_normal,
        incident_dir,
        reflected_dir_raw,
        reflected_dir,
        return_dir);

    grad_sigmas[antenna_index * total_point_size + point_index] = static_cast<scalar_t>(0);

    if (static_cast<float>(specular_alignment) < 1.0e-6f ||
        static_cast<float>(incident_dot_normal) > 0.0f) {
      continue;
    }

    scalar_t reflected_dir_jacobian[9];
    ComputeReflectedDirectionNormalJacobian(incident_dir, normal, reflected_dir_jacobian);

    const float reflected_dir_norm = Norm3(reflected_dir_raw);
    scalar_t normalized_reflected_jacobian[9];
    ComputeNormalizedVectorJacobian(
        reflected_dir_jacobian, reflected_dir, reflected_dir_norm,
        normalized_reflected_jacobian);

    scalar_t grad_specular_alignment[3];
    ProjectDirectionJacobianOntoReturnVector(
        normalized_reflected_jacobian, return_dir, grad_specular_alignment);

    float propagation_distance = ComputePathDistance(tx_position, rx_position, point);
    const float propagation_attenuation = 1.0f / Squared(propagation_distance);
    const float sigma_value =
        static_cast<float>(sigmas[antenna_index * total_point_size + point_index]);

    propagation_distance += kPathLengthBias;
    const float tau = propagation_distance / kSpeedOfLightScaled;

    for (int sample_index = 0; sample_index < num_adc_samples; ++sample_index) {
      const float sample_time = static_cast<float>(sample_index) / adc_sample_rate;
      const float phase = (phase_base + phase_slope * sample_time) * tau;
      const float phase_real = cosf(phase);
      const float phase_imag = sinf(phase);

      const float grad_real =
          static_cast<float>(grad_output_real[antenna_index * num_adc_samples + sample_index]) *
          phase_real;
      const float grad_imag =
          static_cast<float>(grad_output_imag[antenna_index * num_adc_samples + sample_index]) *
          phase_imag;

      const float normal_scale = sigma_value * propagation_attenuation;
      accum_grad_normal[0] +=
          normal_scale * static_cast<float>(grad_specular_alignment[0]) * (grad_real + grad_imag);
      accum_grad_normal[1] +=
          normal_scale * static_cast<float>(grad_specular_alignment[1]) * (grad_real + grad_imag);
      accum_grad_normal[2] +=
          normal_scale * static_cast<float>(grad_specular_alignment[2]) * (grad_real + grad_imag);

      accum_grad_sigma +=
          propagation_attenuation * static_cast<float>(specular_alignment) * (grad_real + grad_imag);
    }

    grad_sigmas[antenna_index * total_point_size + point_index] =
        static_cast<scalar_t>(accum_grad_sigma / total_point_size);
  }

  grad_normals[point_index * 3 + 0] =
      static_cast<scalar_t>(accum_grad_normal[0] / total_point_size);
  grad_normals[point_index * 3 + 1] =
      static_cast<scalar_t>(accum_grad_normal[1] / total_point_size);
  grad_normals[point_index * 3 + 2] =
      static_cast<scalar_t>(accum_grad_normal[2] / total_point_size);
}

struct BilinearForwardLaunch {
  const at::Tensor& normals;
  const at::Tensor& sigmas;
  const at::Tensor& z_vals;
  const at::Tensor& target_z_vals;
  at::Tensor& output_normals;
  at::Tensor& output_sigmas;
  at::Tensor& output_inbounds;
  int num_antennas;
  int num_rays;
  int num_samples;
  int num_target_samples;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int num_threads = num_antennas * num_rays;
    const int blocks = (num_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;
    BilinearRaySampleForwardKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        normals.data_ptr<scalar_t>(),
        sigmas.data_ptr<scalar_t>(),
        z_vals.data_ptr<scalar_t>(),
        target_z_vals.data_ptr<scalar_t>(),
        output_normals.data_ptr<scalar_t>(),
        output_sigmas.data_ptr<scalar_t>(),
        output_inbounds.data_ptr<bool>(),
        num_antennas,
        num_rays,
        num_samples,
        num_target_samples);
  }
};

struct BilinearBackwardLaunch {
  const at::Tensor& grad_output_normals;
  const at::Tensor& grad_output_sigmas;
  const at::Tensor& z_vals;
  const at::Tensor& target_z_vals;
  at::Tensor& grad_normals;
  at::Tensor& grad_sigmas;
  int num_antennas;
  int num_rays;
  int num_samples;
  int num_target_samples;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int num_threads = num_antennas * num_rays;
    const int blocks = (num_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;
    BilinearRaySampleBackwardKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        grad_output_normals.data_ptr<scalar_t>(),
        grad_output_sigmas.data_ptr<scalar_t>(),
        z_vals.data_ptr<scalar_t>(),
        target_z_vals.data_ptr<scalar_t>(),
        grad_normals.data_ptr<scalar_t>(),
        grad_sigmas.data_ptr<scalar_t>(),
        num_antennas,
        num_rays,
        num_samples,
        num_target_samples);
  }
};

struct SignalTracerForwardLaunch {
  const at::Tensor& normals;
  const at::Tensor& sigmas;
  const at::Tensor& points;
  const at::Tensor& tx_pos;
  const at::Tensor& rx_pos;
  const at::Tensor& inbounds;
  at::Tensor& output_real;
  at::Tensor& output_imag;
  int num_adc_samples;
  float adc_sample_rate;
  float start_frequency;
  float chirp_slope_scaled;
  int total_ant_size;
  int total_point_size;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int num_threads = total_ant_size * num_adc_samples;
    const int blocks = (num_threads + kThreadsPerBlock - 1) / kThreadsPerBlock;
    SignalTracerForwardKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        normals.data_ptr<scalar_t>(),
        sigmas.data_ptr<scalar_t>(),
        points.data_ptr<scalar_t>(),
        tx_pos.data_ptr<scalar_t>(),
        rx_pos.data_ptr<scalar_t>(),
        inbounds.data_ptr<bool>(),
        output_real.data_ptr<scalar_t>(),
        output_imag.data_ptr<scalar_t>(),
        num_adc_samples,
        adc_sample_rate,
        start_frequency,
        chirp_slope_scaled,
        total_ant_size,
        total_point_size);
  }
};

struct SignalTracerBackwardLaunch {
  const at::Tensor& grad_output_real;
  const at::Tensor& grad_output_imag;
  const at::Tensor& normals;
  const at::Tensor& sigmas;
  const at::Tensor& points;
  const at::Tensor& tx_pos;
  const at::Tensor& rx_pos;
  const at::Tensor& inbounds;
  at::Tensor& grad_normals;
  at::Tensor& grad_sigmas;
  int num_adc_samples;
  float adc_sample_rate;
  float start_frequency;
  float chirp_slope_scaled;
  int total_ant_size;
  int total_point_size;

  template <typename scalar_t>
  void operator()(cudaStream_t stream) {
    const int blocks = (total_point_size + kThreadsPerBlock - 1) / kThreadsPerBlock;
    SignalTracerBackwardNormalSigmaKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
        grad_output_real.data_ptr<scalar_t>(),
        grad_output_imag.data_ptr<scalar_t>(),
        normals.data_ptr<scalar_t>(),
        sigmas.data_ptr<scalar_t>(),
        points.data_ptr<scalar_t>(),
        tx_pos.data_ptr<scalar_t>(),
        rx_pos.data_ptr<scalar_t>(),
        inbounds.data_ptr<bool>(),
        grad_normals.data_ptr<scalar_t>(),
        grad_sigmas.data_ptr<scalar_t>(),
        num_adc_samples,
        adc_sample_rate,
        start_frequency,
        chirp_slope_scaled,
        total_ant_size,
        total_point_size);
  }
};

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
    int num_target_samples) {
  at::cuda::CUDAGuard device_guard(normals.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(normals.scalar_type(), "bilinear_raysample_forward_cuda", [&] {
    BilinearForwardLaunch{
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
        num_target_samples}
        .template operator()<scalar_t>(stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int num_target_samples) {
  at::cuda::CUDAGuard device_guard(grad_output_normals.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_output_normals.scalar_type(), "bilinear_raysample_backward_cuda", [&] {
        BilinearBackwardLaunch{
            grad_output_normals,
            grad_output_sigmas,
            z_vals,
            target_z_vals,
            grad_normals,
            grad_sigmas,
            num_antennas,
            num_rays,
            num_samples,
            num_target_samples}
            .template operator()<scalar_t>(stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int total_point_size) {
  at::cuda::CUDAGuard device_guard(normals.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(normals.scalar_type(), "signal_tracer_forward_cuda", [&] {
    SignalTracerForwardLaunch{
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
        total_point_size}
        .template operator()<scalar_t>(stream);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    int total_point_size) {
  at::cuda::CUDAGuard device_guard(grad_output_real.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      grad_output_real.scalar_type(), "signal_tracer_backward_normal_sigma_cuda", [&] {
        SignalTracerBackwardLaunch{
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
            total_point_size}
            .template operator()<scalar_t>(stream);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace SignalTracer
