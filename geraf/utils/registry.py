# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
class Registry:
    def __init__(self, name):
        self.name = name
        self._module_dict = dict()

    def register_module(self, name=None, module=None):
        def _register(cls):
            module_name = name if name is not None else cls.__name__
            self._module_dict[module_name] = cls
            return cls

        if module is not None:
            return _register(module)
        return _register

    def build(self, cfg, **kwargs):
        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            raise KeyError(f'cfg must contain the key "type", but got {cfg}')
        
        cfg_ = cfg.copy()
        obj_type = cfg_.pop('type')
        if obj_type not in self._module_dict:
            raise KeyError(
                f'{obj_type} is not in the {self.name} registry')
        
        cls = self._module_dict[obj_type]
        kwargs.update(cfg_)
        return cls(**kwargs)

MODELS = Registry('models')
DATASETS = Registry('datasets')
TRANSFORMS = Registry('transforms')
