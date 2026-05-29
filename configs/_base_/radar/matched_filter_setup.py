# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
mathced_filter = dict(
    # Scan steps and positions
    num_x_stp = 142,     # horizontal (azimuth) scan steps = aperture columns per capture
    num_z_stp = 512,     # vertical scan steps = radar frames per capture [read: sets numFrames in vis tool]

    num_rx_pos = 1,      # number of receive-antenna positions
    num_tx_pos = 1,      # number of transmit-antenna positions

    # Reconstruction voxel size in meters  [read by mfaggregate / vis]
    resx = 0.001,   # depth (x) resolution
    resy = 0.001,   # azimuth (y) resolution
    resz = 0.001,   # elevation (z) resolution

    # Reconstruction bounding box in meters  [read by mfaggregate / vis]
    # Depth range (x-axis)
    xmin = 0.55,    # minimum depth
    xmax = 0.85,    # maximum depth

    # Azimuth range (y-axis)
    ymin = -0.15,   # minimum azimuth
    ymax = 0.15,    # maximum azimuth

    # Elevation range (z-axis)
    zmin = 0.0,     # minimum elevation
    zmax = 0.35,    # maximum elevation

    plt_threshold = 0.12,  # intensity threshold for the radar point-cloud plot
)

