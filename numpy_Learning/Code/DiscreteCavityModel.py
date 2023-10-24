"""
Program Name: DiscreteCavityModel
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/24 上午10:58

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/10/24 上午10:58: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt

# 参数定义
T = 1e-6  # 离散化时间间隔
i_b = np.zeros((2, 1))  # 输入电流
i_g = np.squeeze(np.array([[16e-3], [16e-3]]))  # 输入电流
R_L = 1560e6  # 电阻
omega_half = 2 * np.pi * 216  # 半角频率
omega_0 = 2 * np.pi * 390  # 总失谐量
f_m = np.array([235, 290, 450])  # 机械模式频率
Q_m = np.array([100, 100, 100])  # 机械模式品质因数
K_m = np.array([0.4, 0.3, 0.2])  # 机械模式耦合系数

# 离散化电路模型
A_e = np.array([[-omega_half, -omega_0], [omega_0, -omega_half]])
E = np.array([[1 - T * omega_half, -T * omega_0], [T * omega_0, 1 - T * omega_half]])

# 离散化机电模型
A_m = np.zeros((6, 6))
B_m = np.zeros((6, 1))
omega_m = np.zeros((6, 1))

for m in range(3):
    A_m[2 * m:2 * m + 2, 2 * m:2 * m + 2] = np.eye(2) + T * np.array([[0, 1], [-2 * np.pi * f_m[m] ** 2, -2 * np.pi * f_m[m] / Q_m[m]]])
    B_m[2 * m:2 * m + 2] = T * np.array([[0], [-(2 * np.pi * f_m[m]) ** 2 * K_m[m]]])

# 初始化变量
v = np.zeros((2, int(3 / T)))
omega = np.zeros((6, int(3 / T)))
delta_omega = np.zeros(int(3 / T))

# 进行离散化仿真计算
for n in range(int(3 / T)):
    v[:, n] = np.squeeze(np.dot(E, v[:, n - 1]) + np.dot(np.dot(T * omega_half * R_L, i_g) - np.dot(T * omega_half * R_L, i_b), np.ones(2)))
    omega[:, n] = np.squeeze(np.dot(np.eye(6) + T * A_m, np.expand_dims(omega[:, n - 1], axis=1)) + np.dot(T * B_m, v[0, n] ** 2))
    delta_omega[n] = omega_0 + np.sum(omega[0:6:2, n])

# 绘制图像
time = np.arange(0, 3, T)
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, v[0, :], label='I Component')
plt.plot(time, v[1, :], label='Q Component')
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time, delta_omega)
plt.xlabel('Time (s)')
plt.ylabel('Total Detuning (rad/s)')
plt.tight_layout()
plt.show()
