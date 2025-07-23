#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resistance_measurement.py
Automated Four-Wire Resistance Measurement System
Tektronix 2182A + LPS-305
Author: Ming Liu
Created: July 15, 2025
Last Modified: July 15, 2025
This script provides a GUI for automated resistance measurement using a
Tektronix 2182A voltmeter and an LPS-305 current source.
It supports real-time plotting, data saving, and device configuration.
Dependencies:
- Python 3.x
- PySerial
- NumPy
- Matplotlib
- Tkinter
"""

import os
import sys
import csv
import time
import threading
import queue
from datetime import datetime

import serial
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


# -------------------- 仪器通信封装 --------------------
class Tek2182A:
    """2182A 电压表（SCPI）"""
    def __init__(self, port, baud=9600):
        self.ser = serial.Serial(port, baud, timeout=1)

    def idn(self):
        self.ser.write(b"*IDN?\n")
        return self.ser.readline().decode().strip()

    def read_voltage(self):
        self.ser.write(b":READ?\n")
        val = self.ser.readline().decode().strip()
        return float(val)

    def close(self):
        if self.ser.is_open:
            self.ser.close()


class LPS305:
    """LPS-305 电流源"""
    def __init__(self, port, baud=2400):
        self.ser = serial.Serial(port, baud, timeout=1)

    def idn(self):
        self.ser.write(b"MODEL\r\n")
        return self.ser.readline().decode().strip()

    def set_current(self, i_amps: float):
        cmd = f"ISET1 {i_amps:.4f}\r\n".encode()
        self.ser.write(cmd)
        self.ser.readline()  # 读掉 OK

    def output_on(self):
        self.ser.write(b"OUT 1\r\n")
        self.ser.readline()

    def output_off(self):
        self.ser.write(b"OUT 0\r\n")
        self.ser.readline()

    def close(self):
        if self.ser.is_open:
            self.ser.close()


# -------------------- 主 GUI --------------------
class ResistanceGUI:
    def __init__(self, root):
        self.root = root
        root.title("自动电阻测量系统")
        root.geometry("900x650")

        # 数据队列（线程安全）
        self.data_q = queue.Queue()
        self.meas_thread = None
        self.running = False
        self.data = []

        # 默认串口
        ports = [p.device for p in serial.tools.list_ports.comports()]
        if not ports:
            ports = ["COM1", "COM2", "/dev/ttyUSB0", "/dev/ttyUSB1"]

        # -------------------- 配置区 --------------------
        frm_cfg = ttk.LabelFrame(root, text="设备与参数")
        frm_cfg.pack(fill="x", padx=5, pady=5)

        ttk.Label(frm_cfg, text="电压表端口").grid(row=0, column=0, sticky="w", padx=4)
        self.cmb_v_port = ttk.Combobox(frm_cfg, values=ports, width=12)
        self.cmb_v_port.set(ports[0] if ports else "COM3")
        self.cmb_v_port.grid(row=0, column=1, padx=4)

        ttk.Label(frm_cfg, text="波特率").grid(row=0, column=2, padx=4)
        self.ent_v_baud = ttk.Entry(frm_cfg, width=8)
        self.ent_v_baud.insert(0, "9600")
        self.ent_v_baud.grid(row=0, column=3, padx=4)

        ttk.Label(frm_cfg, text="电流源端口").grid(row=1, column=0, padx=4)
        self.cmb_i_port = ttk.Combobox(frm_cfg, values=ports, width=12)
        self.cmb_i_port.set(ports[1] if len(ports) > 1 else "COM4")
        self.cmb_i_port.grid(row=1, column=1, padx=4)

        ttk.Label(frm_cfg, text="波特率").grid(row=1, column=2, padx=4)
        self.ent_i_baud = ttk.Entry(frm_cfg, width=8)
        self.ent_i_baud.insert(0, "2400")
        self.ent_i_baud.grid(row=1, column=3, padx=4)

        ttk.Label(frm_cfg, text="电流值 A").grid(row=0, column=4, padx=4)
        self.ent_current = ttk.Entry(frm_cfg, width=8)
        self.ent_current.insert(0, "0.01")
        self.ent_current.grid(row=0, column=5, padx=4)

        ttk.Label(frm_cfg, text="采样间隔 s").grid(row=1, column=4, padx=4)
        self.ent_interval = ttk.Entry(frm_cfg, width=8)
        self.ent_interval.insert(0, "1.0")
        self.ent_interval.grid(row=1, column=5, padx=4)

        ttk.Label(frm_cfg, text="测量次数").grid(row=0, column=6, padx=4)
        self.ent_count = ttk.Entry(frm_cfg, width=8)
        self.ent_count.insert(0, "10")
        self.ent_count.grid(row=0, column=7, padx=4)

        ttk.Label(frm_cfg, text="保存文件").grid(row=1, column=6, padx=4)
        self.ent_save = ttk.Entry(frm_cfg, width=18)
        self.ent_save.insert(0, "measurement.csv")
        self.ent_save.grid(row=1, column=7, padx=4)

        # 连接按钮
        self.btn_connect = ttk.Button(frm_cfg, text="连接", command=self.connect)
        self.btn_connect.grid(row=0, column=8, rowspan=2, padx=6, ipadx=10)

        # -------------------- 图形区 --------------------
        frm_plot = ttk.Frame(root)
        frm_plot.pack(fill="both", expand=True, padx=5, pady=5)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212, sharex=self.ax1)
        self.ax1.set_ylabel("Voltage (V)")
        self.ax2.set_ylabel("Resistance (Ω)")
        self.ax2.set_xlabel("Sample index")

        self.canvas = FigureCanvasTkAgg(self.fig, master=frm_plot)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # -------------------- 控制按钮 --------------------
        frm_ctrl = ttk.Frame(root)
        frm_ctrl.pack(fill="x", padx=5, pady=5)

        self.btn_start = ttk.Button(frm_ctrl, text="开始测量", command=self.start_measurement)
        self.btn_start.pack(side="left", padx=5)

        self.btn_save = ttk.Button(frm_ctrl, text="保存数据", command=self.save_data)
        self.btn_save.pack(side="left", padx=5)

        self.btn_exit = ttk.Button(frm_ctrl, text="退出", command=self.on_exit)
        self.btn_exit.pack(side="right", padx=5)

        self.status = ttk.Label(root, text="就绪", relief="sunken", anchor="w")
        self.status.pack(side="bottom", fill="x")

    # -------------------- 功能函数 --------------------
    def connect(self):
        try:
            v_port = self.cmb_v_port.get()
            v_baud = int(self.ent_v_baud.get())
            i_port = self.cmb_i_port.get()
            i_baud = int(self.ent_i_baud.get())

            if v_port == i_port:
                raise ValueError("电压表与电流源不能共用同一串口")

            self.meter = Tek2182A(v_port, v_baud)
            self.source = LPS305(i_port, i_baud)

            v_id = self.meter.idn()
            i_id = self.source.idn()
            self.status.config(text=f"已连接：{v_id} | {i_id}")
            self.btn_connect.config(text="已连接", state="disabled")
        except Exception as e:
            messagebox.showerror("连接失败", str(e))
            self.status.config(text="连接失败")

    def start_measurement(self):
        if self.running:
            self.running = False
            return
        try:
            current = float(self.ent_current.get())
            interval = float(self.ent_interval.get())
            count = int(self.ent_count.get())
            if current < 0 or interval <= 0 or count <= 0:
                raise ValueError("参数必须为正数")
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self.data.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

        self.running = True
        self.btn_start.config(text="停止")
        self.status.config(text="测量中...")

        self.meas_thread = threading.Thread(
            target=self.measure_loop,
            args=(current, interval, count),
            daemon=True
        )
        self.meas_thread.start()

    def measure_loop(self, current, interval, count):
        try:
            self.source.set_current(current)
            self.source.output_on()
            time.sleep(1)  # 稳定

            for idx in range(count):
                if not self.running:
                    break
                voltage = self.meter.read_voltage()
                resistance = voltage / current
                t = datetime.now().isoformat(timespec="seconds")
                self.data.append((t, voltage, resistance, current))
                self.data_q.put((idx, voltage, resistance))
                time.sleep(interval)

        except Exception as e:
            self.data_q.put(("ERROR", str(e)))
        finally:
            self.source.output_off()
            self.data_q.put(("DONE",))

    def update_gui(self):
        try:
            while True:
                item = self.data_q.get_nowait()
                if item[0] == "ERROR":
                    messagebox.showerror("测量错误", item[1])
                    self.running = False
                elif item[0] == "DONE":
                    self.running = False
                    self.btn_start.config(text="开始测量")
                    self.status.config(text="测量完成")
                    return
                else:
                    idx, v, r = item
                    self.ax1.plot(idx, v, 'bo')
                    self.ax2.plot(idx, r, 'ro')
                    self.canvas.draw()
        except queue.Empty:
            pass
        if self.running:
            self.root.after(200, self.update_gui)

    def save_data(self):
        if not self.data:
            messagebox.showinfo("提示", "没有数据可保存")
            return
        fname = self.ent_save.get()
        try:
            write_header = not os.path.isfile(fname)
            with open(fname, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["Timestamp", "Voltage(V)", "Resistance(Ω)", "SetCurrent(A)"])
                for row in self.data:
                    writer.writerow(row)
            self.status.config(text=f"已保存：{fname}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def on_exit(self):
        self.running = False
        if hasattr(self, "meter"):
            self.meter.close()
        if hasattr(self, "source"):
            self.source.close()
        self.root.quit()
        self.root.destroy()


# -------------------- 主入口 --------------------
if __name__ == "__main__":
    import serial.tools.list_ports
    root = tk.Tk()
    app = ResistanceGUI(root)
    root.after(500, app.update_gui)  # 启动 GUI 更新循环
    root.protocol("WM_DELETE_WINDOW", app.on_exit)
    root.mainloop()