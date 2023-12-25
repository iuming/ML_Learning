"""
Program Name: FrequencyData
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 12/24/23 11:24 PM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 12/24/23 11:24 PM: Initial Create.

"""

import pyvisa
import csv
import time
import datetime

# 连接到仪器
rm = pyvisa.ResourceManager()
vna = rm.open_resource('TCPIP0::192.168.8.106::inst0::INSTR')  # 请替换为您的VNA的实际地址

# 打开CSV文件
csv_file = open('frequency_data.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)

# 写入CSV文件的表头
csv_writer.writerow(['Time', 'Marker1Freq'])

try:
    while True:
        # 获取当前时间
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

        # 发送SCPI命令获取Marker1频率
        vna.write("CALC:MARK1:X?")
        marker_frequency = vna.read()

        # 将时间和频率写入CSV文件
        csv_writer.writerow([current_time, marker_frequency])
        csv_file.flush()  # 立即将数据写入文件

        # 等待一段时间，例如1秒
        time.sleep(0.5)

except KeyboardInterrupt:
    # 如果用户按下Ctrl+C，关闭文件和VNA连接
    csv_file.close()
    vna.close()
