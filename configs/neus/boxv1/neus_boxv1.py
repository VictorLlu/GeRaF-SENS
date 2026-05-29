# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
_base_ = [
    '../../_base_/schedules/schedule-2x.py', 
    '../../_base_/default_runtime.py'
]

obj_name = 'boxv1'                            # dataset folder: data/rgb/custom/<obj_name>/   [used]

radar = dict(
    # [USED] camera extrinsics are mapped into the scan/robot frame via this matrix (RGBDataset)
    scan2robot = [[ 0.8152638,  -0.0167912,  -0.57884626,  0.37517157-0.03],
                [-0.57889605,  0.00222062, -0.81539833, -0.39258672+0.01],
                [ 0.01497691,  0.99985655, -0.00790997,  0.42854119],
                [ 0.,          0.,          0.,          1.,        ]],
)


sample_cfg = dict(
    sample_res = [0.002, 0.002, 0.002],   # eval/vis field-extraction grid resolution (x,y,z) in m
    bound =  [0.500, 1.100, -0.37, 0.23, -0.15, 0.45], # scene bounds [xmin,xmax,ymin,ymax,zmin,zmax] in m
    radius = 0.3,                         # unit-sphere normalization radius in m
    n_ray = 512,                          # rays per training image (LoadRandomRays); mesh-extraction chunk size
    N_samples = 64,                       # coarse samples per ray
    n_outside = 32,                       # background samples outside the unit sphere (NeRF++)
    n_importance=64,                      # fine/importance samples per ray
    up_sample_steps=4,                    # hierarchical importance up-sampling iterations
    perturb = 1.0,                        # stratified-sampling jitter magnitude

    sdf_offset = 0.0,                     # SDF iso-level offset for mesh extraction (vis hook)

    anneal_end = 50000,                   # step at which the cosine s-annealing reaches 1.0

)

model=dict(
    type = "NeusRendering",
    with_grad_regression=0.1,
    loss_mode = "charbonier",
    loss_scale = 1e0,
    sample_cfg = sample_cfg,
    radar_cfg = radar,
    nerf = dict(
        type = 'NeRF',
        D = 8,
        d_in = 4,
        d_in_view = 3,
        W = 256,
        multires = 10,
        multires_view = 4,
        output_ch = 4,
        skips=[4],
        use_viewdirs=True
    ),
    sdf_network = dict(
        type='SDFNetwork',
        d_out = 257,
        d_in = 3,
        d_hidden = 256,
        n_layers = 8,
        skip_in = [4],
        multires = 6,
        bias = 0.5,
        scale = 1.0,
        geometric_init = True,
        weight_norm = True,
        ), 
    deviation_network = dict(
        type="SingleVarianceNetwork",
        init_val=0.3,
        activation='exp',
        ),
    color_network = dict(
        type="RenderingNetwork",
        d_feature = 256,
        mode = 'idr',
        d_in = 9,
        d_out = 3,
        d_hidden = 256,
        n_layers = 4,
        weight_norm = True,
        multires_view = 4,
        squeeze_out = True,
    )
)

dataset_type = 'RGBDataset'
meshset_type = 'MeshDataset'
data_root = 'data/rgb/custom/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RGBTo01'),
    dict(type='LoadIntrinsic'),
    dict(type='RGBSceneToUnitSphere'),
    dict(type='CameraToUnitSphere'),
    dict(type='LoadRandomRays'),
    dict(type='GetNearFarFromSphere'),
    dict(type='PackRGBInputs', meta_keys=('color', 'ray_o', 'ray_d', 'near', 'far', 'hit_mask'))
]

test_pipeline = [
    dict(type='RGBSceneToUnitSphere'),
    dict(type='CameraToUnitSphere'),
    dict(type='ExtractFields'),
    dict(type='PackMeshInputs', meta_keys=('sampled_poses_norm', 'sample_cfg', 'normglb_scale'))
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        object_name=obj_name,
        radar_cfg = radar,
        sample_cfg = sample_cfg,
        pipeline=train_pipeline,))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=meshset_type,
        data_root=data_root,
        object_name=obj_name,
        radar_cfg = radar,
        sample_cfg = sample_cfg,
        pipeline=test_pipeline,))

val_evaluator = dict(
    type='SarMetric',
    sample_cfg=sample_cfg,
    metric=['3dsar'],)

test_dataloader = val_dataloader
test_evaluator = val_evaluator


max_iters = 300000
train_cfg = dict(max_iters=max_iters, val_interval=max_iters+1)

base_lr = 5e-4

param_scheduler = [
    dict(
        type='ExponentialLR',
        eta_min=base_lr * 0.05,
        by_epoch=False,
        begin=0,
        end=max_iters,)
]


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0),
    # max_norm=10 is better for SECOND
    paramwise_cfg=dict(
        custom_keys={
            'sdf_network': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10000, max_keep_ckpts=300000),
    visualization=dict(type='RGBMeshVisualizationHook', draw=True, sample_cfg = sample_cfg))

custom_hooks = [dict(type='StepHook'), ]
find_unused_parameters = True
