
"""
Project Name: Voltage Acquisition and Plotting
File Name: voltage_acq_int_clk_plot_data.py
Author: Liu Ming
Created Time: Feb 27th, 2025
File Description: This script acquires voltage data from a NI DAQ device and plots it in real-time using matplotlib. 
                  The data is read from the specified channel and plotted as a waveform.
Preparation:
1. Ensure that the NI DAQ device is properly connected and configured.
2. Install the required Python packages: nidaqmx, matplotlib.
Run Instructions:
1. Execute the script using a Python interpreter.
2. The script will open a matplotlib window displaying the real-time waveform.
Modification History:
- Modified Time: 
- Modified Notes: 
"""

import matplotlib.pyplot as plot

import nidaqmx
from nidaqmx.constants import READ_ALL_AVAILABLE, AcquisitionType

import matplotlib.animation as animation

fig, ax = plot.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, 5000000)
ax.set_ylim(-10, 10)
ax.set_xlabel('Sample Points')
ax.set_ylabel('Amplitude')
ax.set_title('Waveform')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("cDAQ9189-1D712C2Mod1/ai0")
        task.timing.cfg_samp_clk_timing(500000.0, sample_mode=AcquisitionType.FINITE, samps_per_chan=5000000)

        data = task.read(READ_ALL_AVAILABLE)
        x = range(len(data))
        line.set_data(x, data)
    return line,

ani = animation.FuncAnimation(fig, update, init_func=init, blit=True, interval=1000)
plot.show()