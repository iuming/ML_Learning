"""
Project Name: Voltage Continuous Input Application
File Name: tkinter_savefile.py
Author: Liu Ming
Created Time: February 27th, 2025
Description:
This script creates a Tkinter-based GUI application for continuously acquiring voltage data using NI-DAQmx.
The application allows users to configure channel settings, input settings, start and stop data acquisition,
and save the acquired data to a CSV file. The acquired data is displayed in real-time using Matplotlib.
Preparation Before Running:
1. Ensure NI-DAQmx is installed and properly configured on your system.
2. Connect the necessary hardware (e.g., NI DAQ device) and verify its connection.
3. Install required Python packages: nidaqmx, tkinter, matplotlib.
Running Method:
1. Run the script using Python: `python tkinter_savefile.py`
2. Configure the channel and input settings in the GUI.
3. Click "Start Task" to begin data acquisition.
4. Click "Stop Task" to stop data acquisition.
5. Click "Save Data" to save the acquired data to a CSV file.
Modification Notes:
- Modified Time: February 27th, 2025
- Modified By: Liu Ming
- Modified Notes: Description of changes made.
"""

import nidaqmx
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
        self.currentData = []  # Store current acquired data

    def create_widgets(self):
        # The main frame is made up of three subframes
        self.channelSettingsFrame = channelSettings(self, title="Channel Settings")
        self.channelSettingsFrame.grid(row=0, column=1, sticky="ew", pady=(20, 0), padx=(20, 20), ipady=10)

        self.inputSettingsFrame = inputSettings(self, title="Input Settings")
        self.inputSettingsFrame.grid(row=1, column=1, pady=(20, 0), padx=(20, 20), ipady=10)

        self.graphDataFrame = graphData(self)
        self.graphDataFrame.grid(row=0, rowspan=2, column=2, pady=(20, 0), ipady=10)

    def startTask(self):
        # Prevent user from starting task a second time and disable save button
        self.inputSettingsFrame.startButton['state'] = 'disabled'
        self.inputSettingsFrame.saveButton['state'] = 'disabled'

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
            self.currentData = self.task.read(number_of_samples_per_channel=self.numberOfSamples)
            self.graphDataFrame.ax.cla()
            self.graphDataFrame.ax.set_title("Acquired Data")
            for i, channel_data in enumerate(self.currentData):
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
            self.inputSettingsFrame.saveButton['state'] = 'normal'  # Enable save button after stopping

    def stopTask(self):
        # Call back for the "stop task" button
        self.continueRunning = False

    def saveData(self):
        # Check if there is data to save
        if self.currentData:
            # Open save dialog for user to choose file path and name
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Data"
            )
            if file_path:  # If user selected a file path
                try:
                    with open(file_path, 'w') as f:
                        for i, channel_data in enumerate(self.currentData):
                            f.write(f"Channel {i}\n")
                            for sample in channel_data:
                                f.write(f"{sample}\n")
                            f.write("\n")
                    messagebox.showinfo("Save Data", "Data saved successfully!")
                except Exception as e:
                    messagebox.showerror("Save Error", f"Error saving data: {str(e)}")
        else:
            messagebox.showwarning("Save Data", "No data available to save!")

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

        # Add save button, initially disabled
        self.saveButton = ttk.Button(self, text="Save Data", command=self.parent.saveData, state='disabled')
        self.saveButton.grid(row=5, column=0, columnspan=2, sticky='ew', padx=self.xPadding, pady=(10, 0))

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