"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_ss_freqresp
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/21 下午4:47

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/21 下午4:47: Initial Create.

"""

import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
def plot_ss_freqresp(f_wf, A_wf_dB, P_wf_deg, h, fs, Ts, title):
    '''
    Plot the frequency response of a state-space system, used in function
    ``ss_freqresp`` of the ``rf_control`` module.
    '''
    plt.figure()
    plt.subplot(2,2,1)                          # bode plot
    plt.plot(f_wf, A_wf_dB)
    if Ts is not None:
        plt.axvline( fs / 2, ls = '--')
        plt.axvline(-fs / 2, ls = '--')
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.subplot(2,2,3)
    plt.plot(f_wf, P_wf_deg)
    if Ts is not None:
        plt.axvline( fs / 2, ls = '--')
        plt.axvline(-fs / 2, ls = '--')
    plt.grid()
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (deg)')
    plt.subplot(1,2,2)                          # Nyquist plot (need to plot +/- frequencies differently)
    plt.plot(np.real(h[f_wf >= 0.0]), np.imag(h[f_wf >= 0.0]), label = 'Positive Frequency')
    plt.plot(np.real(h[f_wf <  0.0]), np.imag(h[f_wf <  0.0]), label = 'Negative Frequency')
    plt.plot([-1], [0], '*')
    plt.legend()
    plt.grid()
    plt.xlabel('Real')
    plt.ylabel('Imag')
    plt.suptitle(title)
    plt.show(block = False)

def ss_freqresp(A, B, C, D, Ts=None, plot=False, plot_pno=1000, plot_maxf=0.0,
                title='Frequency Response'):
    '''
    Plot the frequency response of a state-space system. This function works
    for both continous system (``Ts`` is None) and discrete systems (``Ts`` has a
    nonzero floating value).

    Parameters:
        A, B, C, D: numpy matrix (complex), state-space model of system
        Ts:         float, sampling time, s
        plot:       boolean, enable the plot of frequency response
        plot_pno:   int, number of point in the plot
        plot_maxf:  float, frequency range (+-) to be plotted, Hz
        title:      string, title showed on the plot

    Returns:
        status:     boolean, success (True) or fail (False)
        f_wf:       numpy array, frequency waveform, Hz
        A_wf_dB:    numpy array, amplitude response waveform, dB
        P_wf_deg:   numpy array, phase response waveform, deg
        h:          numpy array (complex), complex response
    '''
    # check the input
    plot_pno = 1000 if (plot_pno <= 0) else plot_pno
    plot_maxf = 1e6 if (plot_maxf <= 0) else plot_maxf

    if Ts is not None:
        if Ts <= 0.0:
            return (False,) + (None,) * 4

    # calculate the frequency response
    if Ts is None:  # continous
        maxw = 2 * np.pi * plot_maxf
        fs = 1.0
        w, h = signal.freqresp((A, B, C, D), w=np.linspace(-maxw, maxw, plot_pno))
    else:  # discrete
        maxw = np.pi
        fs = 1.0 / Ts
        w, h = signal.dfreqresp((A, B, C, D, Ts), w=np.linspace(-maxw, maxw, plot_pno))

    # calculate the results for display
    f_wf = w / 2 / np.pi * fs
    A_wf_dB = 20 * np.log10(np.abs(h))
    P_wf_deg = np.angle(h, deg=True)

    # plot
    if plot:
        # from rf_plot import plot_ss_freqresp
        plot_ss_freqresp(f_wf, A_wf_dB, P_wf_deg, h, fs, Ts, title)

    # return the frequency response (frequency in absolute Hz)
    return True, f_wf, A_wf_dB, P_wf_deg, h

A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])
Ts = 0.1  # 采样时间

status, f_wf, A_wf_dB, P_wf_deg, h = ss_freqresp(A, B, C, D, Ts, plot=True)
if status:
    print("Frequency response calculated successfully.")
    print("Frequency (Hz):", f_wf)
    print("Amplitude (dB):", A_wf_dB)
    print("Phase (deg):", P_wf_deg)
else:
    print("Failed to calculate frequency response.")