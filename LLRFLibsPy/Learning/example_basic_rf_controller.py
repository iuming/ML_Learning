"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_basic_rf_controller
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/25 下午3:23

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/25 下午3:23: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from set_path import *
from rf_sim import *
from rf_calib import *
from rf_sysid import *
from rf_control import *

# 设置控制器参数
Kp = 1.0
Ki = 0.1

# 设置陷波器配置
notch_conf = {
    'freq_offs': [50.0, 100.0],       # 频率偏移列表，单位Hz
    'gain': [0.5, 0.5],               # 陷波器增益列表（抑制比的倒数）
    'half_bw': [10.0*np.pi, 10.0*np.pi]  # 陷波器带宽一半列表，单位rad/s
}

# 调用函数生成控制器模型
success, Akc, Bkc, Ckc, Dkc = basic_rf_controller(Kp, Ki, notch_conf, plot=True, plot_pno=1000, plot_maxf=200.0)

# 如果绘图成功，显示频率响应图
if success:
    plt.title('Frequency Response of RF Controller')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.show()