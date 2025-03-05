import tkinter as tk
from tkinter import Canvas, Entry, Button, Label, OptionMenu, StringVar
import pyvisa
import time

# 连接示波器
rm = pyvisa.ResourceManager()
try:
    scope = rm.open_resource('TCPIP::10.4.193.26::INSTR')  # 请根据实际设备地址修改
    scope.read_termination = '\n'
    scope.write_termination = '\n'
    scope.timeout = 10000  # 设置10秒超时
except pyvisa.VisaIOError as e:
    print(f"无法连接示波器: {e}")
    exit()

# 创建主窗口
root = tk.Tk()
root.title("示波器控制程序")

# 创建顶部按钮框架
top_frame = tk.Frame(root)
top_frame.pack(side=tk.TOP, fill=tk.X)

# 创建左侧控制框架
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# 创建右侧画布
canvas = Canvas(root, width=600, height=400, bg='white')
canvas.pack(side=tk.RIGHT)

# 顶部按钮：Start 和 Stop
start_button = Button(top_frame, text="Start")
start_button.pack(side=tk.LEFT, padx=5)
stop_button = Button(top_frame, text="Stop")
stop_button.pack(side=tk.LEFT, padx=5)

# 模式选择
mode_label = Label(left_frame, text="模式:")
mode_label.pack()
mode_var = StringVar(root)
mode_var.set("Auto")  # 默认模式为Auto
mode_menu = OptionMenu(left_frame, mode_var, "Auto", "Single")
mode_menu.pack()

# 触发设置（仅在Single模式下生效）
trigger_channel_label = Label(left_frame, text="触发通道:")
trigger_channel_label.pack()
trigger_channel_var = StringVar(root)
trigger_channel_var.set("CH1")  # 默认通道
trigger_channel_menu = OptionMenu(left_frame, trigger_channel_var, "CH1", "CH2", "CH3", "CH4")
trigger_channel_menu.pack()

trigger_threshold_label = Label(left_frame, text="触发阈值 (V):")
trigger_threshold_label.pack()
trigger_threshold_entry = Entry(left_frame)
trigger_threshold_entry.pack()
trigger_threshold_entry.insert(0, "1.0")  # 默认阈值

trigger_mode_label = Label(left_frame, text="触发模式:")
trigger_mode_label.pack()
trigger_mode_var = StringVar(root)
trigger_mode_var.set("Rising")  # 默认上升沿触发
trigger_mode_menu = OptionMenu(left_frame, trigger_mode_var, "Rising", "Falling", "Both")
trigger_mode_menu.pack()

# 全局变量
update_flag = False
triggered = False

# 设置触发条件
def set_trigger():
    channel = trigger_channel_var.get()
    threshold = trigger_threshold_entry.get()
    mode = trigger_mode_var.get()
    try:
        scope.write(f":TRIGger:SOURce {channel}")
        scope.write(f":TRIGger:LEVel {threshold}")
        if mode == "Rising":
            scope.write(":TRIGger:EDGE:SLOPe POSitive")
        elif mode == "Falling":
            scope.write(":TRIGger:EDGE:SLOPe NEGative")
        elif mode == "Both":
            scope.write(":TRIGger:EDGE:SLOPe EITher")
    except pyvisa.VisaIOError as e:
        print(f"设置触发失败: {e}")

# 获取示波器数据
def acquire_data():
    try:
        scope.write(":RUN")
        scope.write(":SINGle")
        while scope.query(":TRIGger:STATUS?").strip() != "STOP":
            time.sleep(0.1)
        data = scope.query_binary_values(":WAVeform:DATA?", datatype='h', is_big_endian=True)
        return data
    except pyvisa.VisaIOError as e:
        print(f"获取数据失败: {e}")
        return []

# 在画布上绘制波形
def plot_waveform(data):
    canvas.delete("all")
    if not data:
        return
    height = 400
    width = 600
    scale_y = height / 10  # 假设电压范围-5V到5V
    scale_x = width / len(data)
    points = []
    for i, val in enumerate(data):
        x = i * scale_x
        y = height / 2 - val * scale_y / 5
        points.append((x, y))
    canvas.create_line(points, fill='blue')

# 周期性更新波形
def update_waveform():
    global update_flag, triggered
    if update_flag:
        if mode_var.get() == "Auto":
            data = acquire_data()
            plot_waveform(data)
            root.after(1000, update_waveform)  # 每1秒更新一次
        elif mode_var.get() == "Single" and not triggered:
            set_trigger()
            data = acquire_data()
            plot_waveform(data)
            triggered = True
            update_flag = False  # 触发后停止更新

# 开始更新
def start_update():
    global update_flag, triggered
    update_flag = True
    triggered = False
    update_waveform()

# 停止更新
def stop_update():
    global update_flag
    update_flag = False

# 绑定按钮功能
start_button.config(command=start_update)
stop_button.config(command=stop_update)

# 运行主循环
root.mainloop()

# 关闭示波器连接
scope.close()
rm.close()