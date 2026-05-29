# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
_base_ = [
    '../../_base_/schedules/schedule-2x.py',
    '../../_base_/default_runtime.py',
]

sub_adc = 8                                   # ADC decimation factor: keep 1 of every `sub_adc` raw samples

numADCSample = 512 // sub_adc                 # ADC samples per chirp after decimation (= 64)
adcSampleRate = 10e6 / sub_adc                # effective ADC sample rate in Hz (= 1.25 MHz)
chirpSlope = 70.15e12                         # FMCW chirp frequency slope in Hz/s
chirpRampTime = numADCSample / adcSampleRate  # active sampling window per chirp in s
c_sci = 2.9979                                # speed of light, scaled units (x1e8 m/s)
fc_sci = 785.0045                             # carrier centre frequency, scaled units (x1e8 Hz ~ 78.5 GHz)
wave_len = c_sci / fc_sci                     # carrier wavelength in m (~3.8 mm)

obj_name = 'bunnyboxv1_sub4'                  # dataset folder under data/rf/process/<obj_name>/

radar = dict(
    radar_path = None,                            # legacy raw-data path (unused)
    # Radar parameters
    numADCSample = numADCSample,                  # ADC samples per chirp            [read by ray tracer / MF]
    adcSampleRate = adcSampleRate,                # ADC sample rate in Hz            [read by ray tracer / MF]
    chirpSlope = chirpSlope,                      # chirp slope in Hz/s (feeds As_sci)
    chirpRampTime = chirpRampTime,                # chirp ramp time in s (feeds chirpBandwidth)
    chirpBandwidth = chirpSlope * chirpRampTime,  # swept bandwidth in Hz (unused)
    As_sci = chirpSlope / 1e8,                    # chirp slope in scaled units for the CUDA kernels [read]
    sub_adc = sub_adc,                            # ADC decimation factor; MF ground-truth divided by it [read]

    sci_fac = 1e8,                                # global unit-scaling factor (unused)
    c_sci = 2.9979,                               # speed of light, x1e8 m/s (unused in dict)
    fc_sci = 785.0045,                            # centre frequency, x1e8 Hz (unused in dict)
    fc_start = 773.704,                           # chirp start frequency, x1e8 Hz (~77.37 GHz) [read]
    wave_len = wave_len,                          # carrier wavelength in m (unused in dict)
    period = 24e-3,                               # frame period in s (unused)
    periodicity = 0.02,                           # time between frames in s (unused)

    # Radar robot calibration
    receiver_trans_end = [0.0681, 0.01325, 0.01535], # RX antenna offset in end-effector frame, m (unused)
    rt_dist = 0.005 + 3 * wave_len / 2,           # TX<->RX path offset distance in m (unused)

    trans_power = 0.0158489319,                   # transmit power; MF images divided by it to normalize [read]
)

sample_cfg = dict(
    sample_res = [0.002, 0.002, 0.002],           # eval/vis grid resolution (x,y,z) in m for field extraction
    bound = [0.500, 1.100, -0.37, 0.23, -0.15, 0.45], # scene bounds [xmin,xmax,ymin,ymax,zmin,zmax] in m
    mf_res = [0.001, 0.001, 0.001],               # matched-filter image voxel size (x,y,z) in m
    radius = 0.3,                                 # unit-sphere normalization radius in m
    N_aperture = 32,                              # aperture sampling grid per side (rays jittered in N x N cells)
    N_samples = 32,                               # samples per ray for the coarse/uniform pass
    N_samples_tgt =64,                            # samples per ray for the target/fine pass
    sdf_offset = 0.04,                            # SDF iso-level offset for mesh/vis extraction
    bank_size = 2,                                # number of antenna chunks processed round-robin per step
    anneal_end = 50000,                           # step at which the cosine s-annealing reaches 1.0
    freeze_inv_s_step = 10000,                    # keep deviation-network inv_s frozen before this step
)

model=dict(
    type = "GeRaFStage1",
    with_grad_regression=0.1,
    rt_backend="cuda",
    mf_backend="cuda",
    loss_mode = "charbonier",
    loss_scale = 1e0,
    sample_cfg = sample_cfg,
    radar_cfg = radar,
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
    signal_network = dict(
        type="FixedPowerNetwork",
        cfg=dict(
            refelective_act = 'sigmoid',
            refelective_exp_max = None,
            light_power = 7.5644,
        ),
    )
)

dataset_type = 'RFDataset'
data_root = 'data/rf/'
train_pipeline = [
    dict(type='LoadRF', normalization=True),
    dict(type='LoadAntennaPositions'),
    dict(type='SceneToUnitSphere'),
    dict(type='AntennasToUnitSphere'),
    dict(type='GetPrimeRay'),
    dict(type='UniformRaySampler'),
    dict(type='TargetRaySampler'),
    dict(type='InterpolateMFAtTargets', mode='bilinear', align_corners=True),
    dict(type='DynamicLossMask', thres_rate=0.04, tot_thres_rate=0.15),
    dict(type='PackRFInputs')
]

test_pipeline = [
    dict(type='LoadAntennaPositions'),
    dict(type='SceneToUnitSphere'),
    dict(type='AntennasToUnitSphere'),
    dict(type='ExtractFields'),
    dict(type='PackRFInputs')
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        object_name=obj_name,
        radar_cfg = radar,
        sample_cfg = sample_cfg,
        pipeline=train_pipeline,))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RFEvalDataset',
        data_root=data_root,
        object_name=obj_name,
        radar_cfg = radar,
        sample_cfg = sample_cfg,
        eval_index=0,
        pipeline=test_pipeline,))

val_evaluator = dict(
    type='SarMetric',
    sample_cfg=sample_cfg,
    metric=['3dsar'],)

test_dataloader = val_dataloader
test_evaluator = val_evaluator


max_iters = 50000
train_cfg = dict(max_iters=max_iters, val_interval=max_iters+1)

base_lr = 1e-3

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.5,
        by_epoch=False,
        begin=0,)
]


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.0),
    paramwise_cfg=dict(
        custom_keys={
            'sdf_network': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=5000, max_keep_ckpts=10000),
    visualization=dict(type='RFMeshVisualizationHook', draw=True, sample_cfg = sample_cfg))

custom_hooks = [dict(type='StepHook')]
find_unused_parameters = True
