# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from geraf.utils.matched_filter.cuda_tools import grid_num


def _save_ply(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(str(path))


class _MeshVisualizationHook:
    def __init__(
        self,
        draw: bool = False,
        sample_cfg=None,
        test_out_dir: Optional[str] = None,
    ):
        self.sample_cfg = sample_cfg or {}
        self.draw = draw
        self.test_out_dir = test_out_dir

        bound = self.sample_cfg["bound"]
        radius = self.sample_cfg["radius"]
        x_c = (bound[0] + bound[1]) / 2.0
        y_c = (bound[2] + bound[3]) / 2.0
        z_c = (bound[4] + bound[5]) / 2.0
        scale = 1.0 / radius

        self.glb2normglb = np.array(
            [[scale, 0, 0, -scale * x_c], [0, scale, 0, -scale * y_c], [0, 0, scale, -scale * z_c], [0, 0, 0, 1]]
        )
        self.normglb2glb = np.array(
            [[radius, 0, 0, x_c], [0, radius, 0, y_c], [0, 0, radius, z_c], [0, 0, 0, 1]]
        )
        self.normglb_scale = scale

        half_side = 1.0 / np.sqrt(3.0)
        corners = np.array(
            [[x, y, z, 1] for x in (-half_side, half_side) for y in (-half_side, half_side) for z in (-half_side, half_side)]
        )
        transformed = (self.normglb2glb @ corners.T).T[:, :3]
        xmin, ymin, zmin = np.min(transformed, axis=0)
        xmax, ymax, zmax = np.max(transformed, axis=0)
        self.sample_bound = [xmin, xmax, ymin, ymax, zmin, zmax]

        sample_res = self.sample_cfg["sample_res"]
        self.sample_shape = (
            grid_num(self.sample_bound[1], self.sample_bound[0], sample_res[0]),
            grid_num(self.sample_bound[3], self.sample_bound[2], sample_res[1]),
            grid_num(self.sample_bound[5], self.sample_bound[4], sample_res[2]),
        )

    def _ensure_out_dir(self, out_dir: str | Path | None = None) -> None:
        if out_dir is not None:
            self.test_out_dir = str(out_dir)
        if self.test_out_dir:
            Path(self.test_out_dir).mkdir(parents=True, exist_ok=True)

    def after_test_iter(self, *args) -> None:
        if not self.draw:
            return

        if len(args) == 3:
            _, outputs, out_dir = args
            self._ensure_out_dir(out_dir)
        elif len(args) == 4:
            _, _, _, outputs = args
            self._ensure_out_dir(self.test_out_dir)
        else:
            raise TypeError(
                "after_test_iter expects either (batch_idx, outputs, out_dir) or (runner, batch_idx, data_batch, outputs)."
            )

        nx, ny, nz = self.sample_shape
        sdfs = outputs["sdfs"].reshape(nx, ny, nz)
        sdfs_ = sdfs - self.sample_cfg.get("sdf_offset", 0.0)
        self.plot_surface_go(sdfs_)

    def plot_surface_go(self, sdf):
        import plotly.graph_objects as go
        from skimage.measure import marching_cubes

        nx, ny, nz = self.sample_shape
        x_range = np.linspace(self.sample_bound[0], self.sample_bound[1], nx)
        y_range = np.linspace(self.sample_bound[2], self.sample_bound[3], ny)
        z_range = np.linspace(self.sample_bound[4], self.sample_bound[5], nz)
        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing="ij")
        spacing = (x[1, 0, 0] - x[0, 0, 0], y[0, 1, 0] - y[0, 0, 0], z[0, 0, 1] - z[0, 0, 0])

        verts, faces, _, _ = marching_cubes(sdf, level=0, spacing=spacing)

        verts[:, 0] += self.sample_bound[0]
        verts[:, 1] += self.sample_bound[2]
        verts[:, 2] += self.sample_bound[4]

        if self.test_out_dir:
            _save_ply(Path(self.test_out_dir) / "mesh.ply", verts, faces)

        fig = go.Figure()
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.5,
                color="lightblue",
                name="SDF Surface",
            )
        )
        fig.update_layout(scene=dict(aspectmode="data"), title="SDF Isosurface Visualization")
        fig.show()


class RFMeshVisualizationHook(_MeshVisualizationHook):
    pass


class RGBMeshVisualizationHook(_MeshVisualizationHook):
    pass
