"""
Program Name: VirtualCavityEnvelopeGraph
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/18 下午3:35

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/10/18 下午3:35: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def v(t, w_half, delta_w):
    i0 = 32e-3
    RL = 1560e6
    numerator = i0 * w_half * RL * (np.exp((-w_half + 1j * delta_w) * t) - 1)
    denominator = -w_half + 1j * delta_w
    return numerator / denominator

t = np.linspace(0, 0.01, 1000)  # 定义时间范围和点数
delta_w_values = [-400 * 2 * np.pi, -150 * 2 * np.pi, -50 * 2 * np.pi, 0 * 2 * np.pi, 50 * 2 * np.pi, 150 * 2 * np.pi, 400 * 2 * np.pi]  # 不同频率差的值

plt.figure()  # 创建新的图形窗口

for delta_w in delta_w_values:
    v_t = v(t, 216 * 2 * np.pi, delta_w)  # 计算函数值

    I = np.real(v_t)  # 提取实部
    Q = np.imag(v_t)  # 提取虚部

    plt.plot(I, Q, label=f'delta_w={delta_w/(2*np.pi)}')  # 绘制实部为横轴，虚部为纵轴的曲线

# 绘制虚线的圆
circle = Circle((2.5e7, 0), radius=2.5e7, linestyle='dashed', edgecolor='black', facecolor='none')
plt.gca().add_patch(circle)

plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Cavity Envelope')
plt.grid(True)
plt.legend()
plt.show()
