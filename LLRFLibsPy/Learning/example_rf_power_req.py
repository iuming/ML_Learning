"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_rf_power_req
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/25 上午10:57

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/25 上午10:57: Initial Create.

"""

import numpy as np

from set_path import *
from rf_sim import *
from rf_calib import *
from rf_sysid import *
from rf_control import *
def rf_power_req(f0, vc0, ib0, phib, Q0, roQ_or_RoQ,
                 QL_vec=None,
                 detuning_vec=None,
                 machine='linac',
                 plot=False):
    '''
    Plot the steady-state forward and reflected power for given cavity voltage,
    beam current and beam phase, as function of loaded Q and detuning. The beam
    phase is defined to be zero for on-crest acceleration.

    Refer to LLRF Book section 3.3.9.

    Parameters:
        f0:           float, RF operating frequency, Hz
        vc0:          float, desired cavity voltage, V
        ib0:          float, desired average beam current, A
        phib:         float, desired beam phase, degree
        Q0:           float, unloaded quality factor (for SC cavity, give it a
                       very high value like 1e10)
        roQ_or_RoQ:   float, cavity r/Q of Linac or R/Q of circular accelerator, Ohm
                       (see the note below)
        QL_vec:       numpy array, QL vector for power calculation
        detuning_vec: numpy array, detuning at which to evaluated the powers
        machine:      string, ``linac`` or ``circular``, used to select r/Q or R/Q
        plot:         boolean, enable/disable the plotting

    Returns:
        status:       boolean, success (True) or fail (False)
        Pfor:         dictionary, keyed by detuning, forward power at different QL
        Pref:         dictionary, keyed by detuning, reflected power at different QL

    Note:
          Linacs define the ``r/Q = Vacc**2 / (w0 * U)`` while circular machines
          define ``R/Q = Vacc**2 / (2 * w0 * U)``, where ``Vacc`` is the accelerating
          voltage, ``w0`` is the angular cavity resonance frequency and ``U`` is the
          cavity energy storage. Therefore, to use this function, one needs to
          specify the ``machine`` to be ``linac`` or ``circular``. Generally, we have
          ``R/Q = 1/2 * r/Q``.
    '''
    # check the input
    if (f0 <= 0.0) or (vc0 < 0.0) or (ib0 < 0.0) or (Q0 <= 0.0) or \
            (roQ_or_RoQ <= 0.0) or (QL_vec.shape[0] < 1):
        return False, None, None

    if detuning_vec is None:
        detuning_vec = [0.0]

    # calcualte the necessary parameters
    if machine == 'circular':
        RL_vec = roQ_or_RoQ * QL_vec  # calculate the loaded resistance RL, Ohm
    else:
        RL_vec = 0.5 * roQ_or_RoQ * QL_vec
    beta_vec = Q0 / QL_vec - 1.0  # input coupling factor
    wh_vec = np.pi * f0 / QL_vec  # hald bandwidth, rad/s
    phib_rad = phib * np.pi / 180.0  # beam phase in radian

    # calculate for each detuning
    Pfor = {}
    Pref = {}
    for dw in detuning_vec:
        Pfor[dw] = (beta_vec + 1) / beta_vec * vc0 ** 2 / 8 / RL_vec * ( \
                    (1 + 2 * RL_vec * ib0 * np.cos(phib_rad) / vc0) ** 2 + \
                    (dw / wh_vec + 2 * RL_vec * ib0 * np.sin(phib_rad) / vc0) ** 2)
        Pref[dw] = (beta_vec + 1) / beta_vec * vc0 ** 2 / 8 / RL_vec * ( \
                    ((beta_vec - 1) / (beta_vec + 1) - 2 * RL_vec * ib0 * np.cos(phib_rad) / vc0) ** 2 + \
                    (dw / wh_vec + 2 * RL_vec * ib0 * np.sin(phib_rad) / vc0) ** 2)

    # plot the result
    if plot:
        from rf_plot import plot_rf_power_req
        plot_rf_power_req(Pfor, Pref, QL_vec)

    return True, Pfor, Pref

# 设置参数
f0 = 1.3e9  # RF 工作频率，Hz
vc0 = 1.5e6  # 期望的腔体电压，V
ib0 = 0.01  # 期望的平均束流电流，A
phib = 30  # 期望的束流相位，度
Q0 = 1e10  # 未加载品质因数
roQ_or_RoQ = 1036  # r/Q 或 R/Q，Ohm
QL_vec = np.array([1e6, 2e6, 3e6])  # QL 向量
detuning_vec = np.array([-100, 0, 100])  # 失谐向量
machine = 'linac'  # 机器类型

# 计算前向和反射功率
status, Pfor, Pref = rf_power_req(f0, vc0, ib0, phib, Q0, roQ_or_RoQ,
                                  QL_vec=QL_vec, detuning_vec=detuning_vec,
                                  machine=machine, plot=True)

if status:
    print("前向功率:", Pfor)
    print("反射功率:", Pref)
else:
    print("计算失败")