# Preparing Data

GeRaF uses two data sources:

- **RGB** — multi-view photos of the object, used to pretrain the vision SDF prior (the NeuS stage).
- **RF** — millimeter-wave radar matched-filter volumes, the target for radar reconstruction (stages 1–2).

> **Download.** The processed RGB and RF datasets are hosted on Dropbox: **`<Dropbox link — to be added>`**.
> Download and extract them so the folders below live under `data/rgb/custom/` and
> `data/rf/process/`. `data/rgb` and `data/rf` may be symlinks to wherever you keep large files.

## RGB

RGB captures follow the NeRF / nerfstudio `transforms.json` convention — one folder per object
under `data/rgb/custom/<object>/`:

```text
data/rgb/custom/<object>/
├── transforms.json     # camera intrinsics + per-frame camera-to-world poses
├── images/             # full-resolution frames: frame_00001.jpg, frame_00002.jpg, ...
├── images_2/           # optional downsampled copies (1/2, 1/4, 1/8 resolution)
├── images_4/
└── images_8/
```

`transforms.json` holds the shared pinhole intrinsics and the per-frame poses:

```json
{
  "w": 1440, "h": 1920,
  "fl_x": 1596.53, "fl_y": 1596.53,
  "cx": 720.0, "cy": 960.0,
  "camera_model": "OPENCV",
  "frames": [
    { "file_path": "images/frame_00001.jpg", "transform_matrix": [[ /* 4x4 */ ]] }
  ]
}
```

`RGBDataset` reads `transforms.json`, loads each `file_path` image, and maps the camera poses into
the scan/robot frame using the config's `radar.scan2robot`. The NeuS config selects an object via
`obj_name` (e.g. `boxv1`).

## RF

RF data is the matched-filter output — one folder per object under `data/rf/process/`. An object
captured as `N` sub-acquisitions carries the `_sub<N>` suffix:

```text
data/rf/process/<obj>_sub<N>/
├── <obj>_sub<N>.npy            # aggregated matched-filter volume (accmf), float32 (Nx x Ny x Nz)
└── mf/                         # one folder per radar frame, named <idx>_<angle>_<sub>
    └── <idx>_<angle>_<sub>/
        ├── sarimage.bin         # per-frame matched-filter SAR cube, float32 (Nx x Ny x Nz)
        ├── rpos.npy             # receive-antenna positions, (M, 3)
        ├── tpos.npy             # transmit-antenna positions, (M, 3)
        ├── adcr.npy             # raw ADC samples, real part
        ├── adci.npy             # raw ADC samples, imaginary part
        └── rotation.npy         # frame rotation / pose
```

- Frame folders are named `<idx>_<angle>_<sub>`: a running index, the turntable angle in degrees,
  and the sub-acquisition id `0..N-1`.
- `<obj>_sub<N>.npy` is the sum of all per-frame `sarimage.bin` volumes.
- The grid size `(Nx, Ny, Nz)` is set by the `bound` / `mf_res` (`mathced_filter`) entries in the
  config; `RFDataset` discovers frames directly from `mf/` (no meta file).
