"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_for_ref_volt2power
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/21 下午2:52

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/21 下午2:52: Initial Create.

"""

import numpy as np


def for_ref_volt2power(roQ_or_RoQ, QL,
                       vf_pcal=None,
                       vr_pcal=None,
                       beta=1e4,
                       machine='linac'):
    '''
    Convert the calibrated forward and reflected signals (in physical unit, V)
    to forward and reflected power (in W).

    Refer to LLRF Book section 3.3.9.

    Parameters:
        roQ_or_RoQ:  float, cavity r/Q of Linac or R/Q of circular accelerator, Ohm
                      (see the note below)
        QL:          float, loaded quality factor of the cavity
        vf_pcal:     numpy array (complex), forward waveform (calibrated to physical unit)
        vf_pcal:     numpy array (complex), reflected waveform (calibrated to physical unit)
        beta:        float, input coupling factor (needed for NC cavities;
                      for SC cavities, can use the default value, or you can
                      specify it if more accurate result is needed)
        machine:     string, 'linac' or 'circular', used to select r/Q or R/Q

    Returns:
        status:      boolean, success (True) or fail (False)
        for_power:   numpy array, waveform of forward power (if input is not None), W
        ref_power:   numpy array, waveform of reflected power (if input is not None), W
        C:           float (complex), calibration coefficient: power_W = C * Volt_V^2

    Note:
          Linacs define the ``r/Q = Vacc**2 / (w0 * U)`` while circular machines
          define ``R/Q = Vacc**2 / (2 * w0 * U)``, where ``Vacc`` is the accelerating
          voltage, ``w0`` is the angular cavity resonance frequency and ``U`` is the
          cavity energy storage. Therefore, to use this function, one needs to
          specify the ``machine`` to be ``linac`` or ``circular``. Generally, we have
          ``R/Q = 1/2 * r/Q``.
    '''
    # check the input
    if (roQ_or_RoQ <= 0.0) or (QL <= 0.0) or (beta <= 0.0):
        return (False,) + (None,) * 3

    # calculate the loaded resistance
    if machine == 'circular':
        RL = roQ_or_RoQ * QL
    else:
        RL = 0.5 * roQ_or_RoQ * QL

    # calculate the coefficient to convert voltage to power
    C = beta / (beta + 1) / (2 * RL)

    # convert the voltage to power if they are valid
    for_power = ref_power = None
    if vf_pcal is not None: for_power = C * np.abs(vf_pcal) ** 2
    if vr_pcal is not None: ref_power = C * np.abs(vr_pcal) ** 2

    return True, for_power, ref_power, C

roQ_or_RoQ = 100  # Ohm
QL = 10000
vf_pcal = np.array([1+1j, 2+2j, 3+3j])  # 校准后的前向电压波形
vr_pcal = np.array([0.5+0.5j, 1+1j, 1.5+1.5j])  # 校准后的反射电压波形
beta = 1e4
machine = 'linac'

status, for_power, ref_power, C = for_ref_volt2power(roQ_or_RoQ, QL, vf_pcal, vr_pcal, beta, machine)

if status:
    print(f"Forward power: {for_power}")
    print(f"Reflected power: {ref_power}")
    print(f"Calibration coefficient: {C}")
else:
    print("Conversion failed.")