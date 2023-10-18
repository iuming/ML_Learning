"""
Program Name: VirtualCavityVoltageEnvelope
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/18 下午3:12

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: Plot Virtual Cavity Voltage Envelope.

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above
- Numpy library(Version 1.23.5 or above)

Modifications:
- 2023/10/18 下午3:12: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt

def v(t, delta_w):
    i0 = 32e-3
    RL = 1560e6
    w_half = 216 * 2 * np.pi
    numerator = i0 * w_half * RL * (np.exp((-w_half + 1j * delta_w) * t) - 1)
    denominator = -w_half + 1j * delta_w
    return numerator / denominator

t = np.linspace(0, 0.01, 1000)  # 定义时间范围和点数
delta_w_values = [-400 * 2 * np.pi, -150 * 2 * np.pi, -50 * 2 * np.pi, 0 * 2 * np.pi, 50 * 2 * np.pi, 150 * 2 * np.pi, 400 * 2 * np.pi]  # 不同频率差的值

plt.figure()  # 创建新的图形窗口

for delta_w in delta_w_values:
    v_t = v(t, delta_w)  # 计算函数值

    I = np.real(v_t)  # 提取实部
    Q = np.imag(v_t)  # 提取虚部

    plt.plot(t, I, label=f'delta_w={delta_w/(2*np.pi)}, I Part')  # 绘制实部曲线
    plt.plot(t, Q, label=f'delta_w={delta_w/(2*np.pi)}, Q Part')  # 绘制虚部曲线

plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Cavity Envelope I, Q Components')
plt.legend()
plt.grid(True)
plt.show()
