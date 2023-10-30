"""
Program Name: TustinTransform
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/10/30 下午2:37

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/10/30 下午2:37: Initial Create.

"""

import numpy as np
from scipy import signal

# 定义连续时间系统的传输函数或状态空间模型
num = [1]  # 连续时间系统的分子多项式系数
den = [1, 2, 1]  # 连续时间系统的分母多项式系数
sys_c = signal.TransferFunction(num, den)

# 定义采样时间
Ts = 0.1

# 使用Tustin变换将连续时间系统转换为离散时间系统
sys_d = signal.cont2discrete((sys_c.num, sys_c.den), Ts, method='tustin')

# 打印离散时间系统的系数
print("离散时间系统的分子多项式系数：", sys_d[0][0])
print("离散时间系统的分母多项式系数：", [sys_d[1][0], sys_d[1][1], sys_d[1][2]])
print("离散时间系统的采样时间：", sys_d[2])
