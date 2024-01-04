import pyvisa
import csv
import time
import datetime

# 连接到频谱仪
rm = pyvisa.ResourceManager()
spec_analyzer = rm.open_resource('TCPIP0::10.4.194.129::inst0::INSTR')  # 请将地址替换为你的频谱仪地址

# 打开CSV文件
csv_file = open('microphonics_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# 写入CSV文件的表头
csv_writer.writerow(['Time', 'Frequency', 'Amplitude'])

try:
    while True:
        # 获取当前时间
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # 发送命令给频谱仪并读取频率和幅度
        spec_analyzer.write("INIT:CONT ON")  # 开启连续测量模式
        time.sleep(0.005)  # 等待0.005秒
        spec_analyzer.write("CALC:MARK1:X?")  # 设置频率
        frequency = spec_analyzer.read()  # 读取频率数据
        spec_analyzer.write("CALC:MARK1:Y?")  # 设置幅度
        amplitude = spec_analyzer.read()  # 读取幅度数据

        # 将数据写入CSV文件
        csv_writer.writerow([current_time, frequency, amplitude])
        csv_file.flush()  # 刷新缓冲区，确保数据被写入文件

except KeyboardInterrupt:
    csv_file.close()
    spec_analyzer.close()
