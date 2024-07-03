
"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_iden_impulse
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/7/3 下午2:52

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/7/3 下午2:52: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt

from set_path import *
from rf_sim import *
from rf_calib import *
from rf_sysid import *
from rf_control import *

# 参数设置
order = 20  # 要识别的脉冲响应的阶数
n_wf = 100  # 波形数量
N = 1000  # 每个波形的采样点数

# 生成脉冲响应 h[n] = (0.5)^n
true_h = (0.5) ** np.arange(order)

# 生成输入和输出数据
U = np.random.randn(n_wf, N) + 1j * np.random.randn(n_wf, N)
Y = np.zeros((n_wf, N), dtype=complex)

# 通过滤波器生成输出数据
for i in range(n_wf):
    Y[i] = np.convolve(U[i], true_h, mode='same')

# 使用 iden_impulse 函数识别脉冲响应
status, identified_h = iden_impulse(U, Y, order)

# 检查结果并绘制对比图
if status:
    plt.figure()
    plt.stem(np.abs(true_h), label='True Impulse Response', use_line_collection=True)
    plt.stem(np.abs(identified_h), linefmt='r--', markerfmt='ro', label='Identified Impulse Response',
             use_line_collection=True)
    plt.xlabel('Sample')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid()
    plt.show()
else:
    print("Impulse response identification failed.")
