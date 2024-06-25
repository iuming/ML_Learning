"""
Program Name: ML_Learning
IDE: PyCharm
File Name: example_opt_QL_detuning
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2024/6/25 下午12:48

Copyright (c) 2024 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2024/6/25 下午12:48: Initial Create.

"""

from set_path import *
from rf_sim import *
from rf_calib import *
from rf_sysid import *
from rf_control import *

# 设置参数
f0 = 1.3e9  # RF 工作频率，Hz
vc0 = 1.5e6  # 期望的腔体电压，V
ib0 = 0.01  # 期望的平均束流电流，A
phib = 30  # 期望的束流相位，度
Q0 = 1e10  # 未加载品质因数
roQ_or_RoQ = 1036  # r/Q 或 R/Q，Ohm
machine = 'linac'  # 机器类型
cav_type = 'sc'  # 腔体类型

# 计算优化的加载 Q 因子、失谐和输入耦合系数
status, QL_opt, dw_opt, beta_opt = opt_QL_detuning(f0, vc0, ib0, phib, Q0, roQ_or_RoQ,
                                                   machine=machine, cav_type=cav_type)

if status:
    print("优化的加载品质因数 QL_opt:", QL_opt)
    print("优化的失谐 dw_opt:", dw_opt)
    print("优化的输入耦合系数 beta_opt:", beta_opt)
else:
    print("计算失败")