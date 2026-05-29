# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
BaseModel = nn.Module
from torch import Tensor

from geraf.utils.registry import MODELS
from geraf.utils import OptConfigType, OptMultiConfig

ForwardResults = Union[Dict[str, torch.Tensor],
                       Tuple[torch.Tensor], torch.Tensor]

class BaseRendering(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__()
        self.init_cfg = init_cfg
        
    def forward(self, data_sample, mode: str = 'tensor') -> ForwardResults:
        if mode == 'loss':
            return self.loss(data_sample)
        elif mode == 'predict':
            return self.predict(data_sample)
        elif mode == 'tensor':
            return self._forward(data_sample)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    @abstractmethod
    def loss(self, data_sample) -> Union[dict, tuple]:
        pass

    @abstractmethod
    def predict(self, data_sample):
        pass

    @abstractmethod
    def _forward(self, data_sample):
        pass