// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <sstream>
#include <string>

#include <torch/extension.h>

#include "matched_filter/MatchedFilter.h"

namespace MatchedFilter {

#if defined(WITH_CUDA) || defined(WITH_HIP)
extern int get_cudart_version();
#endif

std::string get_cuda_version() {
#if defined(WITH_CUDA) || defined(WITH_HIP)
  std::ostringstream stream;

#if defined(WITH_CUDA)
  stream << "CUDA ";
#else
  stream << "HIP ";
#endif

  const auto print_version = [&](int version) {
    stream << (version / 1000) << "." << (version / 10 % 100);
    if (version % 10 != 0) {
      stream << "." << (version % 10);
    }
  };

  print_version(get_cudart_version());
  return stream.str();
#else
  return std::string("not available");
#endif
}

bool has_cuda() {
#if defined(WITH_CUDA)
  return true;
#else
  return false;
#endif
}

std::string get_compiler_version() {
  std::ostringstream stream;

#if defined(__GNUC__) && !defined(__clang__)
#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif
  stream << "GCC " << __GNUC__ << "." << __GNUC_MINOR__;
#endif

#if defined(__clang_major__)
  stream << "clang " << __clang_major__ << "." << __clang_minor__ << "."
         << __clang_patchlevel__;
#endif

#if defined(_MSC_VER)
  stream << "MSVC " << _MSC_FULL_VER;
#endif

  return stream.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "get_compiler_version",
      &get_compiler_version,
      "Return the compiler used to build the extension.");
  module.def(
      "get_cuda_version",
      &get_cuda_version,
      "Return the linked CUDA runtime version.");
  module.def("has_cuda", &has_cuda, "Return whether the extension has CUDA support.");

  module.def(
      "MatchedFilter_forward",
      &MatchedFilter_forward,
      "Run matched-filter accumulation on raw ADC samples.");
  module.def(
      "MatchedFilter_fft_forward",
      &MatchedFilter_fft_forward,
      "Run matched-filter accumulation on range-binned FFT samples.");
  module.def(
      "MatchedFilter_backward_realimag",
      &MatchedFilter_backward_realimag,
      "Backpropagate matched-filter gradients into raw ADC samples.");
  module.def(
      "MatchedFilter_fft_backward_realimag",
      &MatchedFilter_fft_backward_realimag,
      "Backpropagate matched-filter gradients into FFT range bins.");
}

}  // namespace MatchedFilter
