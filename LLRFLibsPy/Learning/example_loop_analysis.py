"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_loop_analysis
Author: mliu
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/7/15 14:42

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/7/15 14:42: Initial Create.

"""

import numpy as np

from set_path import *
from rf_sim import *
from rf_control import *
from rf_calib import *
from rf_sysid import *
from rf_noise import *

# Plant (被控对象) 的状态空间矩阵
AG = np.array([[0, 1], [-2, -3]], dtype=complex)
BG = np.array([[0], [1]], dtype=complex)
CG = np.array([[1, 0]], dtype=complex)
DG = np.array([[0]], dtype=complex)

# 控制器的状态空间矩阵
AK = np.array([[0]], dtype=complex)
BK = np.array([[1]], dtype=complex)
CK = np.array([[1]], dtype=complex)
DK = np.array([[1]], dtype=complex)

# 采样时间（连续系统），设置为None代表连续系统
Ts = None

# 回路延迟时间，单位为秒
delay_s = 0.1

# 是否绘图
plot = True

# 绘图的点数
plot_pno = 500

# 绘制的最大频率范围（Hz）
plot_maxf = 100.0

# 绘图的标签
label = "Example of control system"

# 调用 loop_analysis 函数
status, S_max, T_max = loop_analysis(AG, BG, CG, DG, AK, BK, CK, DK, Ts, delay_s, plot, plot_pno, plot_maxf, label)

# 输出结果
if status:
    print(f"最大灵敏度 S_max: {S_max} dB")
    print(f"最大互补灵敏度 T_max: {T_max} dB")
else:
    print("分析失败")
