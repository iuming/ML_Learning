import numpy as np
import matplotlib.pyplot as plt

# 参数定义
N = 256  # 信号总样本数
up_duration = 60  # 上升阶段的样本数
flat_duration = 90  # 平顶阶段的样本数
down_duration = 106  # 衰减阶段的样本数
peak_value = 14  # 峰值
K_L = -1.48  # 失谐量计算的常数

# 生成脉冲信号 E_acc
t = np.arange(N)
E_acc = np.zeros(N)

# 上升阶段 (0-500)
E_acc[0:up_duration] = peak_value * (1 - np.exp(-t[0:up_duration] / 10))

# 平顶阶段 (500-1300)
E_acc[up_duration:up_duration + flat_duration] = peak_value

# 衰减阶段 (1300-2048)
E_acc[up_duration + flat_duration:N] = peak_value * np.exp(- (t[up_duration + flat_duration:N] - (up_duration + flat_duration)) / 20)

# 计算失谐量信号 Δf
Delta_f = K_L * (E_acc ** 2)

# 将失谐量信号输出到Delta_f.txt文件
np.savetxt('Delta_f.txt', Delta_f)

# 绘制失谐量信号，供可视化参考
plt.plot(t, Delta_f)
plt.title('Generated Tuning Signal (Delta f)')
plt.xlabel('Sample Index')
plt.ylabel('Tuning Value (Delta f)')
plt.grid()
plt.show()