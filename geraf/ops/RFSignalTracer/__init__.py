# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from importlib import import_module


_EXPORTS = {
    "signal_tracer_cuda": ".signal_tracer_cuda",
    "bilinear_raysample_cuda": ".bilinear_raysample_cuda",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
