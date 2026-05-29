# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from torch.utils.data import Dataset

from geraf.datasets.rf_dataset import Compose
from geraf.utils.registry import DATASETS


@DATASETS.register_module()
class RGBDataset(Dataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        object_name: str = "",
        radar_cfg: Optional[dict] = None,
        sample_cfg: Optional[dict] = None,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        max_refetch: int = 1000,
        **kwargs,
    ) -> None:
        self.data_root = data_root
        self.object_name = object_name
        self.radar_cfg = radar_cfg or {}
        self.sample_cfg = sample_cfg or {}
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.max_refetch = max_refetch
        self.data_list = self.load_data_list()

    def load_data_list(self) -> List[dict]:
        data_path = os.path.join(self.data_root, self.object_name)
        meta_path = os.path.join(data_path, "transforms.json")
        with open(meta_path, "r", encoding="utf-8") as handle:
            meta = json.load(handle)

        intrinsic_dict = dict(
            w=meta["w"],
            h=meta["h"],
            fl_x=meta["fl_x"],
            fl_y=meta["fl_y"],
            cx=meta["cx"],
            cy=meta["cy"],
        )
        scan2robot = np.asarray(self.radar_cfg["scan2robot"], dtype=np.float32)

        data_list = []
        for frame in meta["frames"]:
            image_path = frame["file_path"]
            match = re.search(r"frame_(\d+)\.(png|jpg|jpeg)$", image_path)
            frame_id = int(match.group(1)) if match else len(data_list)

            extrinsic = np.asarray(frame["transform_matrix"], dtype=np.float32)
            extrinsic = scan2robot @ extrinsic
            extrinsic = extrinsic @ np.diag([1, -1, -1, 1]).astype(np.float32)

            data_list.append(
                dict(
                    id=frame_id,
                    data_root=self.data_root,
                    data_path=data_path,
                    img_path=os.path.join(data_path, image_path),
                    sample_cfg=copy.deepcopy(self.sample_cfg),
                    intrinsic_dict=intrinsic_dict,
                    extrinsic=extrinsic,
                )
            )
        return data_list

    def __len__(self) -> int:
        return len(self.data_list)

    def get_data_info(self, idx: int) -> dict:
        data_info = copy.deepcopy(self.data_list[idx])
        data_info["sample_idx"] = idx if idx >= 0 else len(self) + idx
        return data_info

    def prepare_data(self, idx: int) -> Any:
        return self.pipeline(self.get_data_info(idx))

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self))

    def __getitem__(self, idx: int) -> dict:
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
class MeshDataset(Dataset):
    def __init__(
        self,
        data_root: Optional[str] = None,
        object_name: str = "",
        radar_cfg: Optional[dict] = None,
        sample_cfg: Optional[dict] = None,
        pipeline: List[Union[dict, Callable]] = [],
        **kwargs,
    ) -> None:
        self.data_root = data_root
        self.object_name = object_name
        self.radar_cfg = radar_cfg or {}
        self.sample_cfg = sample_cfg or {}
        self.pipeline = Compose(pipeline)
        self.data_info = dict(
            data_root=data_root,
            data_path=os.path.join(data_root, object_name),
            sample_cfg=copy.deepcopy(self.sample_cfg),
            extrinsic=np.eye(4, dtype=np.float32),
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict:
        if idx != 0:
            raise IndexError(idx)
        return self.pipeline(copy.deepcopy(self.data_info))
