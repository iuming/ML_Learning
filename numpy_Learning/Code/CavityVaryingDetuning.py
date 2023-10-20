"""
Program Name: CavityVaryingDetuning
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/19 下午6:20

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/10/19 下午6:20: Initial Create.

"""

import numpy as np
import matplotlib.pyplot as plt

f_m = np.array([335, 490, 750])
Q_m = np.array([100, 70, 50])
K_m = np.array([0.4, 0.3, 0.2])
omega_0 = 0
t = np.linspace(0, 0.12, 1000000)

def voltage_function(t):
    t = t % 0.04
    return 14 * ((t < 0.0004) * 1.47 * (1 - np.exp(-t / 0.00032)) + ((t >= 0.0004) & (t <= 0.001)) * 1 + (t > 0.001) * np.exp(-(t - 0.001) / 0.000497))

def calculate_discrepancy(t, V):
    omega_m = np.zeros((2, len(t)))
    for i in range(3):
        A_m = np.array([[0, 1], [-2 * np.pi * f_m[i]**2, -2 * np.pi * f_m[i] / Q_m[i]]])
        B_m = np.array([0, -(2 * np.pi * f_m[i])**2 * K_m[i]])
        omega_m[:, 0] = np.array([0, 0])
        for j in range(len(t) - 1):
            dt = t[j + 1] - t[j]
            domega_m = A_m.dot(omega_m[:, j]) + B_m * V[j]**2
            omega_m[:, j + 1] = omega_m[:, j] + domega_m * dt
    return omega_m[0, :]

V = voltage_function(t)
omega_m = calculate_discrepancy(t, V)

plt.plot(t, omega_m)
plt.xlabel('Time(s)')
plt.ylabel('Detuning(Hz)')
plt.title('Cavity detuning due to dynamjc Lorentz forces')
plt.grid(True)
plt.show()
