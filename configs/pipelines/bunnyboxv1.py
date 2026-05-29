# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
pipeline = dict(
    name="bunnyboxv1",
    stages=[
        dict(
            id="neus",
            config="configs/neus/boxv1/neus_boxv1.py",
        ),
        dict(
            id="stage1",
            config="configs/geraf/bunnybv2/geraf2_bunnyboxv1_stage1.py",
        ),
        dict(
            id="stage2",
            config="configs/geraf/bunnybv2/geraf2_bunnybv3c_stage2clean_r6_newsched.py",
            checkpoint_sources=dict(
                tgt_sdf_ckpt_path="neus",
                pretrained="stage1",
            ),
        ),
    ],
)
