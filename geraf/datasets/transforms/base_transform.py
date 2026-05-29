# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

class BaseTransform:
    """Minimal replacement for mmcv.transforms.BaseTransform."""
    def __call__(self, results):
        return self.transform(results)
    
    def transform(self, results):
        raise NotImplementedError
