import numpy as np
from scipy import signal

# 传递函数的分子和分母系数
num = [1, 3, 5]
den = [1, 2, 1]

# 将传递函数转换为状态空间表示
A, B, C, D = signal.tf2ss(num, den)

print("A matrix:")
print(A)
print("B matrix:")
print(B)
print("C matrix:")
print(C)
print("D matrix:")
print(D)
