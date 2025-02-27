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
        self.title("信号采集程序")
        self.geometry("1200x800")
        self.create_widgets()
        self.data_ready = False
        self.acquired_data = None
        self.stop_monitoring = False
        self.monitoring = False

    def create_widgets(self):
        # 输入框部分
        tk.Label(self, text="触发信号地址").grid(row=0, column=0, padx=5, pady=5)
        self.trigger_address_entry = tk.Entry(self)
        self.trigger_address_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="触发阈值").grid(row=1, column=0, padx=5, pady=5)
        self.threshold_entry = tk.Entry(self)
        self.threshold_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="触发门限类型").grid(row=2, column=0, padx=5, pady=5)
        self.trigger_type_var = tk.StringVar()
        self.trigger_type_dropdown = tk.OptionMenu(self, self.trigger_type_var, "上触发", "下触发", "上下触发")
        self.trigger_type_dropdown.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self, text="信号地址 (逗号分隔)").grid(row=3, column=0, padx=5, pady=5)
        self.signal_addresses_entry = tk.Entry(self)
        self.signal_addresses_entry.grid(row=3, column=1, padx=5, pady=5)

        tk.Label(self, text="采集频率 (Hz)").grid(row=4, column=0, padx=5, pady=5)
        self.sampling_frequency_entry = tk.Entry(self)
        self.sampling_frequency_entry.grid(row=4, column=1, padx=5, pady=5)

        tk.Label(self, text="采集数量").grid(row=5, column=0, padx=5, pady=5)
        self.number_of_samples_entry = tk.Entry(self)
        self.number_of_samples_entry.grid(row=5, column=1, padx=5, pady=5)

        # 按钮
        self.monitor_button = tk.Button(self, text="开始监测", command=self.start_monitoring)
        self.monitor_button.grid(row=6, column=0, padx=5, pady=5)

        self.save_button = tk.Button(self, text="保存数据", command=self.save_data, state="disabled")
        self.save_button.grid(row=6, column=1, padx=5, pady=5)

        # 绘图区域
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=2, rowspan=7, padx=5, pady=5)

    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.stop_monitoring = False
            self.monitor_button.config(text="停止监测", command=self.stop_monitoring_func)
            threading.Thread(target=self.acquire_data).start()
            self.check_data_ready()

    def stop_monitoring_func(self):
        self.stop_monitoring = True
        self.monitoring = False
        self.monitor_button.config(text="开始监测", command=self.start_monitoring)

    def acquire_data(self):
        # 获取输入参数
        trigger_address = self.trigger_address_entry.get()
        threshold = float(self.threshold_entry.get())
        trigger_type = self.trigger_type_var.get()
        signal_addresses = self.signal_addresses_entry.get().split(',')
        all_channels = [trigger_address] + signal_addresses
        sampling_frequency = float(self.sampling_frequency_entry.get())
        num_samples = int(self.number_of_samples_entry.get())

        # 创建DAQ任务
        task = nidaqmx.Task()
        for ch in all_channels:
            task.ai_channels.add_ai_voltage_chan(ch)

        if trigger_type in ["上触发", "下触发"]:
            # 使用硬件触发
            slope = nidaqmx.constants.Slope.RISING if trigger_type == "上触发" else nidaqmx.constants.Slope.FALLING
            task.timing.cfg_samp_clk_timing(sampling_frequency, sample_mode=nidaqmx.constants.AcquisitionType.FINITE, samps_per_chan=num_samples)
            task.triggers.start_trigger.cfg_anlg_edge_start_trig(trigger_address, level=threshold, slope=slope)
            task.start()
            data = task.read(number_of_samples_per_channel=num_samples, timeout=10.0)
            task.stop()
            task.close()
            self.acquired_data = data
            self.data_ready = True
        else:  # "上下触发"
            # 使用软件触发
            task.timing.cfg_samp_clk_timing(sampling_frequency, sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            task.start()
            chunk_size = 100
            # 读取第一个样本（触发通道），确保是浮点数
            last_sample = task.read(number_of_samples_per_channel=1)[0][0]  # 触发通道的第一个样本
            while not self.stop_monitoring:
                chunk = task.read(number_of_samples_per_channel=chunk_size)
                for i in range(len(chunk[0])):
                    current = chunk[0][i]  # 触发通道的当前样本（浮点数）
                    if (last_sample < threshold and current >= threshold) or (last_sample > threshold and current <= threshold):
                        # 检测到触发条件
                        data = task.read(number_of_samples_per_channel=num_samples)
                        task.stop()
                        task.close()
                        self.acquired_data = data
                        self.data_ready = True
                        return
                    last_sample = current  # 更新 last_sample 为当前样本
            task.stop()
            task.close()

    def check_data_ready(self):
        if self.data_ready:
            self.plot_data()
            self.save_button.config(state="normal")
            self.monitoring = False
            self.monitor_button.config(text="开始监测", command=self.start_monitoring)
        else:
            self.after(100, self.check_data_ready)

    def plot_data(self):
        self.ax.clear()
        for i, channel_data in enumerate(self.acquired_data):
            self.ax.plot(channel_data, label=f'通道 {i}')
        self.ax.set_xlabel("样本")
        self.ax.set_ylabel("电压")
        self.ax.legend()
        self.canvas.draw()

    def save_data(self):
        if self.acquired_data:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV 文件", "*.csv")])
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(','.join([f'通道 {i}' for i in range(len(self.acquired_data))]) + '\n')
                    for row in zip(*self.acquired_data):
                        f.write(','.join(map(str, row)) + '\n')
                messagebox.showinfo("保存数据", "数据已成功保存！")
        else:
            messagebox.showwarning("保存数据", "没有可保存的数据！")

if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()