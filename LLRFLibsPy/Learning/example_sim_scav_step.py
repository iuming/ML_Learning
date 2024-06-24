"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_sim_scav_step
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/24 下午1:11

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/24 下午1:11: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt


# 假设这些模块已经正确导入
# from set_path import *
# from rf_sim import *
# from rf_control import *

def sim_scav_step(half_bw, dw_step0, detuning0, vf_step, vb_step, vc_step0, Ts, beta=1e4,
                  state_m0=None, Am=None, Bm=None, Cm=None, Dm=None, mech_exe=False):
    if (half_bw <= 0.0) or (Ts <= 0.0) or (beta <= 0.0):
        return (False,) + (None,) * 4

    vc_step = (1 - Ts * (half_bw - 1j * dw_step0)) * vc_step0 + \
              2 * half_bw * Ts * (beta * vf_step / (beta + 1) + vb_step)
    vr_step = vc_step - vf_step

    if (state_m0 is None) or \
            (Am is None) or \
            (Bm is None) or \
            (Cm is None) or \
            (Dm is None) or \
            (not mech_exe):
        state_m = state_m0
        dw = detuning0
    else:
        state_m = Am @ state_m0 + Bm @ (np.abs(vc_step) * 1.0e-6) ** 2
        dw = Cm @ state_m + Dm @ (np.abs(vc_step) * 1.0e-6) ** 2 + detuning0

    return True, vc_step, vr_step, dw, state_m


# 参数设置
Ts = 1e-6  # 电气模型采样时间，s
mds = 1  # 机械模型降采样因子
Tsm = Ts * mds  # 机械模型采样时间，s

# 机械模式参数
mech_modes = {'f': [280, 341, 460, 487, 618],
              'Q': [40, 20, 50, 80, 100],
              'K': [2, 0.8, 2, 0.6, 0.2]}

# 机械模式状态空间矩阵（示例数据）
Am = np.array([[1, 0], [0, 1]])  # 实际应根据机械模式计算得到
Bm = np.array([[1], [1]])
Cm = np.array([[1, 0]])
Dm = np.array([[1]])

# 腔体参数
f0 = 1.3e9  # RF工作频率，Hz
roQ = 1036  # 腔体的r/Q，Ohm
QL = 3e6  # 加载品质因数
wh = np.pi * f0 / QL  # 半带宽，rad/s
RL = 0.5 * roQ * QL  # 加载电阻（Linac惯例），Ohm
ig = 0.016  # RF驱动功率等效电流，A
ib = 0.008  # 平均束流电流，A
t_fill = 510  # 腔体填充周期长度，样本数
t_flat = 1300  # 平顶周期结束时间，样本数
dw0 = 2 * np.pi * 0  # 初始失谐，rad/s

# 初始化
N = 2048  # 示例中使用较少的点数
vc = np.zeros(N, dtype=complex)
vf = np.zeros(N, dtype=complex)
vr = np.zeros(N, dtype=complex)
vb = np.zeros(N, dtype=complex)
dw = np.zeros(N)

# 设置前向电压
vf[:t_fill] = RL * ig  # 定义前向RF驱动
vf[t_fill:t_flat] = RL * ig
vb[t_fill:t_flat] = -RL * ib  # 定义束流驱动

# 初始化机械方程状态
state_m = np.zeros((Am.shape[0], 1))  # 机械方程状态
state_vc = 0.0  # 腔体方程状态
dw_step0 = 0.0

# 模拟过程
for i in range(N):
    status, vc[i], vr[i], dw[i], state_m = sim_scav_step(wh, dw_step0, dw0, vf[i], vb[i], state_vc,
                                                         Ts, beta=1e4, state_m0=state_m,
                                                         Am=Am, Bm=Bm, Cm=Cm, Dm=Dm, mech_exe=True)
    state_vc = vc[i]
    dw_step0 = dw[i]

# 绘制结果
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.abs(vc) * 1e-6)
plt.xlabel('Time (Ts)')
plt.ylabel('Cavity Voltage (MV)')
plt.subplot(2, 1, 2)
plt.plot(dw / 2 / np.pi)
plt.xlabel('Time (Ts)')
plt.ylabel('Detuning (Hz)')
plt.show()
