"""
Project Name: Data Acquisition and Plotting
File Name: plot_data.py
Author: Liu Ming
Created Time: Feb 27th, 2025
Description:
This script acquires voltage data from a specified NI DAQ device channel and plots the data in real-time using matplotlib. 
The script continuously reads data from the DAQ device and updates the plot with new data points.
Preparation:
- Ensure that the NI DAQ device is properly connected and configured.
- Install the required Python packages: matplotlib, nidaqmx.
Run Instructions:
- Execute the script using a Python interpreter.
- The plot will update in real-time with the acquired data.
Modification History:
- Modified Time: 
- Modified Notes: 
"""

import matplotlib.pyplot as plt

import nidaqmx
from nidaqmx.constants import TerminalConfiguration

def main():
    plt.ion()  # prevents graph from closing between new data sets
    plt.ylim(-1, 1)  # setting graph axis

    i = 0
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("cDAQ9189-1D712C2Mod1/ai0", terminal_config=TerminalConfiguration.DIFF)  # initialize data acquisition task
        task.timing.cfg_samp_clk_timing(rate=1000.0)  # set sampling rate to 1000 samples per second

        while True:
            data = task.read(number_of_samples_per_channel=1)  # reads from channel every loop

            plt.scatter(i, data[0], c='r')  # plots data
            plt.pause(0.02)

            i += 1

if __name__ == "__main__":
    main()