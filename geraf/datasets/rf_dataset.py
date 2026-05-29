# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
import copy
import logging
import os
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from geraf.utils.registry import DATASETS, TRANSFORMS


def print_log(*args, **kwargs):
    print(*args)


def _object_paths(data_root: str, object_name: str):
    """Resolve the accumulated-MF file and the per-frame directory.

        accmf  -> {data_root}/process/{object_name}/{object_name}.npy
        frames -> {data_root}/process/{object_name}/mf/{frame}
    """
    object_dir = os.path.join(data_root, "process", object_name)
    accmf_path = os.path.join(object_dir, f"{object_name}.npy")
    frames_dir = os.path.join(object_dir, "mf")
    return accmf_path, frames_dir


def _list_frames(frames_dir: str):
    """Return the per-frame folder names, ordered by (index, sub-index)."""
    frames = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
    return sorted(frames, key=lambda x: (int(x.split("_")[0]), int(x.split("_")[-1])))


class Compose:
    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []
        for transform in transforms or []:
            if isinstance(transform, dict):
                transform = TRANSFORMS.build(transform)
            if not callable(transform):
                raise TypeError(f"transform must be callable, but got {type(transform)}")
            self.transforms.append(transform)

    def __call__(self, data: dict) -> Optional[dict]:
        for transform in self.transforms:
            data = transform(data)
            if data is None:
                return None
        return data


@DATASETS.register_module()
class RFDataset(Dataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        object_name: str = "",
        sub_frame=None,
        radar_cfg: Optional[dict] = None,
        sample_cfg: Optional[dict] = None,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        max_refetch: int = 1000,
        **kwargs,
    ) -> None:
        self.data_root = data_root
        self.object_name = object_name
        self.radar_cfg = radar_cfg
        self.sample_cfg = sample_cfg
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.sub_frame = sub_frame
        self.pipeline = Compose(pipeline)
        self._fully_initialized = False
        self.full_init()

    def full_init(self) -> None:
        if self._fully_initialized:
            return
        self.data_list = self.load_data_list()
        self._fully_initialized = True

    def load_data_list(self) -> List[dict]:
        accmf_path, frames_dir = _object_paths(self.data_root, self.object_name)
        object_files = self.sub_frame or _list_frames(frames_dir)
        return [
            dict(
                plane_name=object_file,
                data_root=self.data_root,
                accmf_path=accmf_path,
                data_path=os.path.join(frames_dir, object_file),
                data_name=object_file,
                radar_cfg=self.radar_cfg,
                sample_cfg=self.sample_cfg,
            )
            for object_file in object_files
        ]

    def get_data_info(self, idx: int) -> dict:
        data_info = copy.deepcopy(self.data_list[idx])
        data_info["sample_idx"] = idx if idx >= 0 else len(self) + idx
        return data_info

    def __len__(self) -> int:
        return len(self.data_list)

    def prepare_data(self, idx: int) -> Any:
        return self.pipeline(self.get_data_info(idx))

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self))

    def __getitem__(self, idx: int) -> dict:
        if not self._fully_initialized:
            print_log(
                "Please call `full_init()` method manually to accelerate the speed.",
                logger="current",
                level=logging.WARNING,
            )
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise RuntimeError("Test pipeline should not return None.")
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            if data is not None:
                return data
            idx = self._rand_another()

        raise RuntimeError(f"Cannot find valid sample after {self.max_refetch} retries.")


@DATASETS.register_module()
class RFEvalDataset(Dataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        object_name: str = "",
        radar_cfg: Optional[dict] = None,
        sample_cfg: Optional[dict] = None,
        pipeline: List[Union[dict, Callable]] = [],
        eval_index: int = 0,
        **kwargs,
    ) -> None:
        self.data_root = data_root
        self.object_name = object_name
        self.radar_cfg = radar_cfg
        self.sample_cfg = sample_cfg
        self.eval_index = eval_index
        self.pipeline = Compose(pipeline)
        self.data_info = self.load_single_data_info()

    def load_single_data_info(self) -> dict:
        accmf_path, frames_dir = _object_paths(self.data_root, self.object_name)
        object_files = _list_frames(frames_dir)
        if not object_files:
            raise RuntimeError(f"No frames found for object {self.object_name!r} in {frames_dir}.")

        frame_name = object_files[self.eval_index]
        return dict(
            plane_name=frame_name,
            data_root=self.data_root,
            accmf_path=accmf_path,
            data_path=os.path.join(frames_dir, frame_name),
            data_name=frame_name,
            radar_cfg=self.radar_cfg,
            sample_cfg=self.sample_cfg,
            sample_idx=0,
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        if idx != 0:
            raise IndexError(idx)
        return self.pipeline(copy.deepcopy(self.data_info))
