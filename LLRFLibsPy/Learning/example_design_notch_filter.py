"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_design_notch_filter
Author: mliu
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/7/15 10:59

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/7/15 10:59: Initial Create.

"""

import numpy as np
import scipy.signal as signal


def design_notch_filter(fnotch, Q, fs):
    '''
    Design the notch filter (return a discrete filter).

    Parameters:
        fnotch:  float, frequency to be notched, Hz
        Q:       float, quality factor of the notch filter
        fs:      float, sampling frequency, Hz

    Returns:
        status:         boolean, success (True) or fail (False)
        Ad, Bd, Cd, Dd: numpy matrix, discrete notch filter
    '''
    # check the input
    if (fnotch <= 0.0) or (Q <= 0.0) or (fs <= 0.0):
        return (False,) + (None,) * 4

        # create the notch filter
    b, a = signal.iirnotch(fnotch, Q, fs)
    dsys = signal.dlti(b, a, dt=1.0 / fs)
    dsys = signal.StateSpace(dsys)
    Ad, Bd, Cd, Dd = dsys.A, dsys.B, dsys.C, dsys.D

    # filter the data
    return True, Ad, Bd, Cd, Dd


# 示例参数
fnotch = 60.0  # 陷波频率 (Hz)
Q = 30.0  # 质量因数
fs = 1000.0  # 采样频率 (Hz)

# 设计陷波滤波器
status, Ad, Bd, Cd, Dd = design_notch_filter(fnotch, Q, fs)

# 打印结果
if status:
    print("设计成功!")
    print("Ad 矩阵:")
    print(Ad)
    print("Bd 矩阵:")
    print(Bd)
    print("Cd 矩阵:")
    print(Cd)
    print("Dd 矩阵:")
    print(Dd)
else:
    print("设计失败.")
