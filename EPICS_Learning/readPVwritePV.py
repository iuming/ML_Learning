"""
Program Name: readPVwritePV
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/11/17 上午9:37

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/11/17 上午9:37: Initial Create.

"""

import epics
import time

# 实时读取PV的值
def read_pv(pv_name):
    value = epics.caget(pv_name)
    print(f"当前 {pv_name} 的值为: {value}")

# 实时写入PV的值
def write_pv(pv_name, new_value):
    print(f"设置 {pv_name} 的值为: {new_value}")
    epics.caput(pv_name, new_value)

if __name__ == "__main__":
    # PV的名称
    pv_name = 'CavityIOC::example'

    # 每隔一秒读取一次PV的值
    while True:
        read_pv(pv_name)
        time.sleep(1)

        # 每隔两秒写入一次PV的值
        write_pv(pv_name, 10)
        time.sleep(2)
