"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_cav_impulse
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/24 下午3:49

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/24 下午3:49: Initial Create.

"""

import numpy as np

# 设置参数
half_bw = 2 * np.pi * 10e3    # 半带宽，弧度每秒
detuning = 2 * np.pi * 500    # 失谐，弧度每秒
Ts = 1e-6                     # 采样时间，秒
order = 20                    # 脉冲响应阶数


def cav_impulse(half_bw, detuning, Ts, order=20):
    '''
    Derive the impulse response from the cavity equation. We assume that the
    system gain has been normalized to 1 and the system phase corrected to 0.
    Therefore, the referred cavity equation is ``dvc/dt + (half_bw - 1j * detuning)*vc = half_bw * vd``,
    where ``vc`` is the vector of cavity voltage and ``vd`` is the vector of cavity
    drive (in principle ``vd = 2 * beta * vf / (beta + 1)``).

    Refer to LLRF Book section 4.5.2.

    Parameters:
        half_bw:  float, constant half bandwidth of the cavity, rad/s
        detuning: float, constant detuning of the cavity, rad/s
        Ts:       float, sampling time, s
        order:    int, order of the impulse response

    Returns:
        status:   boolean, success (True) or fail (False)
        h:        numpy array (complex), impulse response
    '''
    # check the input
    if (half_bw <= 0.0) or (detuning <= 0.0) or (Ts <= 0.0) or \
            (order < 2):
        return False, None

    # calculate the impulse response
    k = np.arange(order)
    h = Ts * half_bw * (1.0 - Ts * (half_bw - 1j * detuning)) ** k

    # return the results
    return True, h

# 计算脉冲响应
status, h = cav_impulse(half_bw, detuning, Ts, order)

if status:
    print("脉冲响应计算成功")
    print(h)
else:
    print("脉冲响应计算失败")