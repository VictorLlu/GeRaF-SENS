// Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <cuda_runtime_api.h>

namespace SignalTracer {

int get_cudart_version() {
    // Check if we are using CUDA (i.e., not on a ROCm platform)
#if !defined(__HIP_PLATFORM_HCC__)
    // Return CUDA Runtime version
    return CUDART_VERSION;

#else
    // We are on a ROCm platform, get the HIP runtime version
    int version = 0;

#if HIP_VERSION_MAJOR != 0
    // Create a version format similar to CUDA's versioning scheme
    version = HIP_VERSION_MINOR;
    version += (HIP_VERSION_MAJOR * 100);
#else
    // Directly get the version using the HIP runtime API
    hipRuntimeGetVersion(&version);
#endif

    return version;
#endif
}

} // namespace SignalTracer
