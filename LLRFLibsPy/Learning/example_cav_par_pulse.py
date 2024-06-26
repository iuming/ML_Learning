"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_cav_par_pulse
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/25 下午4:27

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/25 下午4:27: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_calib import *
from rf_sysid import *
from rf_noise import *
from rf_misc import *

# 示例参数设置
sampling_time = 1e-6  # 采样时间，单位为秒
half_bandwidth = 2 * np.pi * 100  # 半带宽，单位为rad/s
beta = 1e4  # 输入耦合因子

# 生成示例波形
time = np.arange(0, 0.01, sampling_time)  # 时间数组
vc = np.exp(1j * 2 * np.pi * 50 * time) * np.exp(-time / 0.002)  # 腔体探测信号
vf = np.exp(1j * 2 * np.pi * 50 * time)  # 腔体前向信号

# 调用函数计算
success, wh_pul, dw_pul = cav_par_pulse(vc, vf, half_bandwidth, sampling_time, beta)

# 打印结果
if success:
    print("Calculation successful!")
    print("Half-bandwidth in pulse:", wh_pul)
    print("Detuning in pulse:", dw_pul)

    # 绘制结果
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time, wh_pul, label='Half-bandwidth (rad/s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Half-bandwidth (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(time, dw_pul, label='Detuning (rad/s)', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Detuning (rad/s)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
else:
    print("Calculation failed.")
