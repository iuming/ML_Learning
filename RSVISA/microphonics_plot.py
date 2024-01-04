"""
Program Name: microphonics_plot
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 1/4/24 19:24 PM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 1/4/24 19:24 PM: Initial Create.

"""
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件并跳过首行
data = pd.read_csv('microphonics_data.csv', header=0)

# 绘制直方图
plt.hist(data['Frequency'], bins=20, color='blue', alpha=0.7)
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.title('Histogram of Frequency')
plt.show()