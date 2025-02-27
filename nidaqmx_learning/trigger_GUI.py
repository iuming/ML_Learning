import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import nidaqmx
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Signal Acquisition Program")
        self.geometry("1200x800")
        self.create_widgets()
        self.data_ready = False
        self.acquired_data = None
        self.stop_monitoring = False
        self.monitoring = False

    def create_widgets(self):
        # Input fields
        tk.Label(self, text="Trigger Signal Address").grid(row=0, column=0, padx=5, pady=5)
        self.trigger_address_entry = tk.Entry(self)
        self.trigger_address_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="Trigger Threshold").grid(row=1, column=0, padx=5, pady=5)
        self.threshold_entry = tk.Entry(self)
        self.threshold_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="Trigger Type").grid(row=2, column=0, padx=5, pady=5)
        self.trigger_type_var = tk.StringVar()
        self.trigger_type_dropdown = tk.OptionMenu(self, self.trigger_type_var, "Rising Edge", "Falling Edge", "Both Edges")
        self.trigger_type_dropdown.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self, text="Signal Addresses (comma separated)").grid(row=3, column=0, padx=5, pady=5)
        self.signal_addresses_entry = tk.Entry(self)
        self.signal_addresses_entry.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self, text="Sampling Frequency (Hz)").grid(row=4, column=0, padx=5, pady=5)
        self.sampling_frequency_entry = tk.Entry(self)
        self.sampling_frequency_entry.grid(row=4, column=1, padx=5, pady=5)

        tk.Label(self, text="Number of Samples").grid(row=5, column=0, padx=5, pady=5)
        self.number_of_samples_entry = tk.Entry(self)
        self.number_of_samples_entry.grid(row=5, column=1, padx=5, pady=5)

        # Buttons
        self.monitor_button = tk.Button(self, text="Start Monitoring", command=self.start_monitoring)
        self.monitor_button.grid(row=6, column=0, padx=5, pady=5)

        self.save_button = tk.Button(self, text="Save Data", command=self.save_data, state="disabled")
        self.save_button.grid(row=6, column=1, padx=5, pady=5)

        # Plotting area
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=7, padx=5, pady=5)

    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.stop_monitoring = False
            self.monitor_button.config(text="Stop Monitoring", command=self.stop_monitoring_func)
            threading.Thread(target=self.acquire_data).start()
            self.check_data_ready()

    def stop_monitoring_func(self):
        self.stop_monitoring = True
        self.monitoring = False
        self.monitor_button.config(text="Start Monitoring", command=self.start_monitoring)

    def acquire_data(self):
        # Get input parameters
        trigger_address = self.trigger_address_entry.get()
        threshold = float(self.threshold_entry.get())
        trigger_type = self.trigger_type_var.get()
        signal_addresses = self.signal_addresses_entry.get().split(',')
        all_channels = [trigger_address] + signal_addresses
        sampling_frequency = float(self.sampling_frequency_entry.get())
        num_samples = int(self.number_of_samples_entry.get())

        # Create DAQ task
        task = nidaqmx.Task()
        for ch in all_channels:
            task.ai_channels.add_ai_voltage_chan(ch)

        if trigger_type in ["Rising Edge", "Falling Edge"]:
            # Use hardware trigger
            slope = nidaqmx.constants.Slope.RISING if trigger_type == "Rising Edge" else nidaqmx.constants.Slope.FALLING
            task.timing.cfg_samp_clk_timing(sampling_frequency, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=num_samples)
            task.triggers.start_trigger.cfg_anlg_edge_start_trig(trigger_address, trigger_level=threshold, slope=slope)
            task.start()
            data = task.read(number_of_samples_per_channel=num_samples, timeout=10.0)
            task.stop()
            task.close()
            self.acquired_data = data
            self.data_ready = True
        else:  # "Both Edges"
            # Use software trigger
            task.timing.cfg_samp_clk_timing(sampling_frequency, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            task.start()
            chunk_size = 100
            # Read the first sample (trigger channel), ensure it is a float
            last_sample = task.read(number_of_samples_per_channel=1)[0][0]  # First sample of the trigger channel
            while not self.stop_monitoring:
                chunk = task.read(number_of_samples_per_channel=chunk_size)
                for i in range(len(chunk[0])):
                    current = chunk[0][i]  # Current sample of the trigger channel (float)
                    if (last_sample < threshold and current >= threshold) or (last_sample > threshold and current <= threshold):
                        # Trigger condition detected
                        data = task.read(number_of_samples_per_channel=num_samples)
                        task.stop()
                        task.close()
                        self.acquired_data = data
                        self.data_ready = True
                        return
                    last_sample = current  # Update last_sample to current sample
            task.stop()
            task.close()

    def check_data_ready(self):
        if self.data_ready:
            self.plot_data()
            self.save_button.config(state="normal")
            self.monitoring = False
            self.monitor_button.config(text="Start Monitoring", command=self.start_monitoring)
        else:
            self.after(100, self.check_data_ready)

    def plot_data(self):
        self.ax.clear()
        for i, channel_data in enumerate(self.acquired_data):
            self.ax.plot(channel_data, label=f'Channel {i}')
        self.ax.set_xlabel("Sample")
        self.ax.set_ylabel("Voltage")
        self.ax.legend()
        self.canvas.draw()

    def save_data(self):
        if self.acquired_data:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(','.join([f'Channel {i}' for i in range(len(self.acquired_data))]) + '\n')
                    for row in zip(*self.acquired_data):
                        f.write(','.join(map(str, row)) + '\n')
                messagebox.showinfo("Save Data", "Data has been successfully saved!")
        else:
            messagebox.showwarning("Save Data", "No data to save!")

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
