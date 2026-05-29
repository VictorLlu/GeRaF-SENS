# Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
numADCSample = 512                            # ADC samples per chirp (full rate, no decimation)
adcSampleRate = 10.0e6                         # ADC sample rate in Hz (10 MHz)
chirpSlope = 70.15e12                          # FMCW chirp frequency slope in Hz/s
chirpRampTime = numADCSample / adcSampleRate   # active sampling window per chirp in s
c_sci = 2.9979                                 # speed of light, scaled units (x1e8 m/s)
fc_sci = 785.0045                              # carrier centre frequency, scaled units (x1e8 Hz ~ 78.5 GHz)
wave_len = c_sci / fc_sci                      # carrier wavelength in m (~3.8 mm)

radar = dict(
    radar_path = None,                            # legacy raw-data path
    # Radar parameters
    numADCSample = numADCSample,                  # ADC samples per chirp
    adcSampleRate = adcSampleRate,                # ADC sample rate in Hz
    chirpSlope = chirpSlope,                      # chirp slope in Hz/s (feeds As_sci)
    chirpRampTime = chirpRampTime,                # chirp ramp time in s (feeds chirpBandwidth)
    chirpBandwidth = chirpSlope * chirpRampTime,  # swept bandwidth in Hz
    As_sci = chirpSlope / 1e8,                    # chirp slope in scaled units for the CUDA kernels

    sci_fac = 1e8,                                # global unit-scaling factor (x1e8)
    c_sci = 2.9979,                               # speed of light, x1e8 m/s
    fc_sci = 785.0045,                            # centre frequency, x1e8 Hz
    fc_start = 773.704,                           # chirp start frequency, x1e8 Hz (~77.37 GHz)
    wave_len = wave_len,                          # carrier wavelength in m
    period = 35e-3,                               # frame period in s
    # period = 50e-3,                             # (alternate frame period)
    periodicity = 0.005,                          # time between frames in s
    # periodicity = 0.01,                         # (alternate inter-frame time)

    # Radar robot calibration
    receiver_trans_end = [0.0681, 0.01325, 0.01535], # RX antenna offset in end-effector frame, m
    rt_dist = 0.005 + 3 * wave_len / 2,           # TX<->RX path offset distance in m
    # end-effector "plate" -> robot "arm" calibration transform (4x4); the +0.013/-0.032
    # terms are a manual fine-calibration offset on the translation column
    plate2arm = [[ 0.99925354,  0.03856638, -0.0022367 ,  0.77854254+0.013],
       [-0.03856026,  0.99925258,  0.00271737, -0.0265301-0.032],
       [ 0.00233982, -0.00262909,  0.99999381,  0.02742596],
       [ 0.        ,  0.        ,  0.        ,  1.        ]]
)
