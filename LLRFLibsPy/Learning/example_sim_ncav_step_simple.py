"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_sim_ncav_step_simple
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/21 下午2:44

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/21 下午2:44: Initial Create.

"""

import numpy as np

def sim_ncav_step_simple(half_bw, detuning, vf_step, vb_step, vc_step0, Ts, beta = 1e4):
    '''
    Simulate the cavity response for a time step using the simple discrete
    cavtiy equation (Euler method for discretization).

    Parameters:
        half_bw:  float, half bandwidth of the cavity (constant), rad/s
        detuning: float, detuning of the cavity (constant), rad/s
        vf_step:  complex, cavity forward voltage of this step
        vb_step:  complex, beam drive voltage of this step
        vc_step0: complex, cavity voltage of the last step
        Ts:       float, sampling time, s
        beta:     float, input coupling factor (needed for NC cavities;
                   for SC cavities, can use the default value, or you can
                   specify it if more accurate result is needed)
    Returns:
        status:   boolean, success (True) or fail (False)
        vc_step:  complex, cavity voltage of this step
        vr_step:  complex, cavity reflected voltage of this step
    '''
    # check the input
    if (half_bw <= 0.0) or (Ts <= 0.0) or (beta <= 0.0):
        return False, None, None

    # make a step of calculation
    vc_step = (1 - Ts * (half_bw - 1j*detuning)) * vc_step0 + \
              2 * half_bw * Ts * (beta * vf_step / (beta + 1) + vb_step)
    vr_step = vc_step - vf_step

    # return the results of the step
    return True, vc_step, vr_step

half_bw = 2 * np.pi * 100  # 100 Hz 半带宽
detuning = 2 * np.pi * 10  # 10 Hz 失谐
vf_step = 1 + 0j  # 前向电压
vb_step = 0 + 0j  # 束流驱动电压
vc_step0 = 0 + 0j  # 上一个时间步的腔体电压
Ts = 1e-3  # 1 毫秒采样时间
beta = 1e4  # 输入耦合因子

status, vc_step, vr_step = sim_ncav_step_simple(half_bw, detuning, vf_step, vb_step, vc_step0, Ts, beta)

if status:
    print(f"Cavity voltage: {vc_step}")
    print(f"Reflected voltage: {vr_step}")
else:
    print("Simulation failed.")