"""
Program Name: ML_Learning
IDE: PyCharm
File Name: eample_sim_ncav_pulse
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/20 下午8:07

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/20 下午8:07: Initial Create.

"""

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def sim_ncav_pulse(Arfc, Brfc, Crfc, Drfc, vf, Ts,
                   Abmc=None,
                   Bbmc=None,
                   Cbmc=None,
                   Dbmc=None,
                   vb=None):
    '''
    Simulate the cavity response to a pulsed RF drive and beam current. This
    function is for normal conducting cavties with constant QL and detuning.

    Parameters:
        Arfc, Brfc, Crfc, Drfc: numpy matrix (complex), continous cavity model for RF drive
        vf:                     numpy array (complex), cavity forward voltage (calibrated to
                                 the cavity probe signal reference plane)
        Ts:                     float, sampling frequency, Hz
        Abmc, Bbmc, Cbmc, Dbmc: numpy matrix (complex), continous cavity model for beam drive
        vb:                     numpy array (complex), beam drive voltage (calibrated to
                                 the cavity probe signal reference plane)

    Returns:
        status: boolean, success (True) or fail (False)
        T:      numpy array, time waveform, s
        vc:     numpy array (complex), cavity voltage waveform
        vr:     numpy array (complex), cavity reflected voltage waveform
    '''
    # check the input
    if (Ts <= 0.0):
        return False, None, None, None

    if vb is not None:
        if (not vb.shape == vf.shape):
            return False, None, None, None

    # simulate the response of the continous system
    T = np.arange(vf.shape[0]) * Ts
    _, vc_rf, _ = signal.lsim((Arfc, Brfc, Crfc, Drfc), vf, T)  # Returns: T, Yout, Xout
    vc = vc_rf
    if not any([x is None for x in (Abmc, Bbmc, Cbmc, Dbmc, vb)]):
        _, vc_bm, _ = signal.lsim((Abmc, Bbmc, Cbmc, Dbmc), vb, T)
        vc += vc_bm

    # get the cavity reflected
    vr = vc - vf

    return True, T, vc, vr

# 定义连续腔体模型的状态空间矩阵（假设的值）
Arfc = np.array([[0.0, 1.0], [-1.0, -1.0]])
Brfc = np.array([[0.0], [1.0]])
Crfc = np.array([[1.0, 0.0]])
Drfc = np.array([[0.0]])

# 定义采样时间
Ts = 1e-3  # 1毫秒

# 定义前向电压驱动信号（示例脉冲信号）
time_steps = 1000
vf = np.zeros(time_steps)
vf[100:200] = 1.0  # 在100到200时间步之间有一个幅度为1的脉冲

# 定义束流驱动模型的状态空间矩阵（假设的值，可选）
Abmc = np.array([[0.0, 1.0], [-1.0, -1.0]])
Bbmc = np.array([[0.0], [1.0]])
Cbmc = np.array([[1.0, 0.0]])
Dbmc = np.array([[0.0]])

# 定义束流驱动信号（与前向电压驱动信号形状相同，可选）
vb = np.zeros(time_steps)
vb[150:250] = 0.5  # 在150到250时间步之间有一个幅度为0.5的脉冲

# 调用 sim_ncav_pulse 函数
status, T, vc, vr = sim_ncav_pulse(Arfc, Brfc, Crfc, Drfc, vf, Ts, Abmc, Bbmc, Cbmc, Dbmc, vb)

# 检查状态并打印结果
if status:
    plt.figure(figsize=(10, 6))

    plt.plot(T, np.real(vf), label='Forward Voltage (Real Part)')
    plt.plot(T, np.imag(vf), '--', label='Forward Voltage (Imaginary Part)')
    plt.plot(T, np.real(vc), label='Cavity Voltage (Real Part)')
    plt.plot(T, np.imag(vc), '--', label='Cavity Voltage (Imaginary Part)')
    plt.plot(T, np.real(vr), label='Reflected Voltage (Real Part)')
    plt.plot(T, np.imag(vr), '--', label='Reflected Voltage (Imaginary Part)')

    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]')
    plt.title('Cavity Response to Pulsed RF Drive and Beam Current')
    plt.grid(True)
    plt.show()
    plt.savefig('example_sim_ncav_pulse.')
else:
    print("Simulation failed.")
