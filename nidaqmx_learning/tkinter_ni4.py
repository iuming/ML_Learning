"""
Project Name: NI-DAQmx Continuous Voltage Input with Tkinter
File Name: tkinter_ni4.py
Author: Liu Ming
Created Time: February 27th, 2025
Description: This script creates a Tkinter-based GUI application for continuously acquiring voltage input using NI-DAQmx. The application allows users to configure channel settings, input settings, and visualize the acquired data in real-time using Matplotlib. Supports multiple channels.
Preparation:
1. Ensure NI-DAQmx drivers and Python nidaqmx package are installed.
2. Install required Python packages: tkinter, matplotlib, and nidaqmx.
Run Instructions:
1. Execute the script using Python: `python tkinter_ni4.py`
2. Configure the channel and input settings in the GUI (enter multiple channels separated by commas).
3. Click "Start Task" to begin data acquisition and visualization.
4. Click "Stop Task" to stop data acquisition.
Modification Notes:
- Modified Time: February 27th, 2025
- Modified Notes: Added support for multiple channel acquisition and visualization.
"""

import nidaqmx

import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class voltageContinuousInput(tk.Frame):

    def __init__(self, master):
        tk.Frame.__init__(self, master)

        # Configure root tk class
        self.master = master
        self.master.title("Voltage - Continuous Input")
        self.master.geometry("1100x600")

        self.create_widgets()
        self.pack()
        self.run = False

    def create_widgets(self):
        # The main frame is made up of three subframes
        self.channelSettingsFrame = channelSettings(self, title="Channel Settings")
        self.channelSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(20, 0), padx=(20, 20), ipady=10)

        self.inputSettingsFrame = inputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=1, column=1, pady=(20, 0), padx=(20, 20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0, rowspan=2, column=2, pady=(20, 0), ipady=10)

    def startTask(self):
        # Prevent user from starting task a second time
        self.inputSettingsFrame.startButton['state'] = 'disabled'

        # Shared flag to alert task if it should stop
        self.continueRunning = True

        # Get task settings from the user
        physicalChannels = self.channelSettingsFrame.physicalChannelsEntry.get().split(',')
        maxVoltage = float(self.channelSettingsFrame.maxVoltageEntry.get())
        minVoltage = float(self.channelSettingsFrame.minVoltageEntry.get())
        sampleRate = int(self.inputSettingsFrame.sampleRateEntry.get())
        self.numberOfSamples = int(self.inputSettingsFrame.numberOfSamplesEntry.get())

        # Create and start task
        self.task = nidaqmx.Task()
        for channel in physicalChannels:
            self.task.ai_channels.add_ai_voltage_chan(channel.strip(), min_val=minVoltage, max_val=maxVoltage)
        self.task.timing.cfg_samp_clk_timing(sampleRate, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS, samps_per_chan=self.numberOfSamples * 3)
        self.task.start()

        # Spin off call to check
        self.master.after(10, self.runTask)

    def runTask(self):
        # Check if task needs to update the graph
        samplesAvailable = self.task._in_stream.avail_samp_per_chan
        if samplesAvailable >= self.numberOfSamples:
            vals = self.task.read(number_of_samples_per_channel=self.numberOfSamples)
            self.graphDataFrame.ax.cla()
            self.graphDataFrame.ax.set_title("Acquired Data")
            for i, channel_data in enumerate(vals):
                self.graphDataFrame.ax.plot(channel_data, label=f'Channel {i}')
            self.graphDataFrame.ax.legend()
            self.graphDataFrame.graph.draw()

        # Check if the task should sleep or stop
        if self.continueRunning:
            self.master.after(10, self.runTask)
        else:
            self.task.stop()
            self.task.close()
            self.inputSettingsFrame.startButton['state'] = 'enabled'

    def stopTask(self):
        # Call back for the "stop task" button
        self.continueRunning = False


class channelSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.grid_columnconfigure(0, weight=1)
        self.xPadding = (30, 30)
        self.create_widgets()

    def create_widgets(self):
        self.physicalChannelsLabel = ttk.Label(self, text="Physical Channels (comma separated)")
        self.physicalChannelsLabel.grid(row=0, sticky='w', padx=self.xPadding, pady=(10, 0))

        self.physicalChannelsEntry = ttk.Entry(self)
        self.physicalChannelsEntry.insert(0, "cDAQ9189-1D712C2Mod1/ai0,cDAQ9189-1D712C2Mod1/ai1")
        self.physicalChannelsEntry.grid(row=1, sticky="ew", padx=self.xPadding)

        self.maxVoltageLabel = ttk.Label(self, text="Max Voltage")
        self.maxVoltageLabel.grid(row=2, sticky='w', padx=self.xPadding, pady=(10, 0))
        
        self.maxVoltageEntry = ttk.Entry(self)
        self.maxVoltageEntry.insert(0, "10")
        self.maxVoltageEntry.grid(row=3, sticky="ew", padx=self.xPadding)

        self.minVoltageLabel = ttk.Label(self, text="Min Voltage")
        self.minVoltageLabel.grid(row=4, sticky='w', padx=self.xPadding, pady=(10, 0))

        self.minVoltageEntry = ttk.Entry(self)
        self.minVoltageEntry.insert(0, "-10")
        self.minVoltageEntry.grid(row=5, sticky="ew", padx=self.xPadding, pady=(0, 10))


class inputSettings(tk.LabelFrame):

    def __init__(self, parent, title):
        tk.LabelFrame.__init__(self, parent, text=title, labelanchor='n')
        self.parent = parent
        self.xPadding = (30, 30)
        self.create_widgets()

    def create_widgets(self):
        self.sampleRateLabel = ttk.Label(self, text="Sample Rate")
        self.sampleRateLabel.grid(row=0, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10, 0))

        self.sampleRateEntry = ttk.Entry(self)
        self.sampleRateEntry.insert(0, "1000")
        self.sampleRateEntry.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.numberOfSamplesLabel = ttk.Label(self, text="Number of Samples")
        self.numberOfSamplesLabel.grid(row=2, column=0, columnspan=2, sticky='w', padx=self.xPadding, pady=(10, 0))

        self.numberOfSamplesEntry = ttk.Entry(self)
        self.numberOfSamplesEntry.insert(0, "100")
        self.numberOfSamplesEntry.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.xPadding)

        self.startButton = ttk.Button(self, text="Start Task", command=self.parent.startTask)
        self.startButton.grid(row=4, column=0, sticky='w', padx=self.xPadding, pady=(10, 0))

        self.stopButton = ttk.Button(self, text="Stop Task", command=self.parent.stopTask)
        self.stopButton.grid(row=4, column=1, sticky='e', padx=self.xPadding, pady=(10, 0))


class graphData(tk.Frame):

    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.create_widgets()

    def create_widgets(self):
        self.graphTitle = ttk.Label(self, text="Voltage Input")
        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_title("Acquired Data")
        self.graph = FigureCanvasTkAgg(self.fig, self)
        self.graph.draw()
        self.graph.get_tk_widget().pack()


# Creates the tk class and primary application "voltageContinuousInput"
root = tk.Tk()
app = voltageContinuousInput(root)

# Start the application
app.mainloop()