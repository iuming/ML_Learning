"""
Program Name: Day24_tushare
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 1/24/24 4:58 PM

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 1/24/24 4:58 PM: Initial Create.

"""
import tushare as ts
import matplotlib.pyplot as plt

df1 = ts.get_k_data('600519', ktype='D', start='2010-04-26', end='2020-04-26')

datapath1 = "../datasets/SH600519.csv"
df1.to_csv(datapath1)