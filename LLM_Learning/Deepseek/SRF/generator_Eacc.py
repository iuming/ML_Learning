import numpy as np
import matplotlib.pyplot as plt

# 参数定义
N = 2048  # 信号总样本数
up_duration = 500  # 上升阶段的样本数
flat_duration = 800  # 平顶阶段的样本数
down_duration = 748  # 衰减阶段的样本数
peak_value = 14  # 峰值

# 时间数组
t = np.arange(N)

# 生成脉冲信号
signal = np.zeros(N)

# 上升阶段 (0-500)
# 使用指数上升函数
signal[0:up_duration] = peak_value * (1 - np.exp(-t[0:up_duration] / 80))  # 80是控制上升速度的参数

# 平顶阶段 (500-1300)
signal[up_duration:up_duration + flat_duration] = peak_value

# 衰减阶段 (1300-2048)
# 使用指数衰减函数
signal[up_duration + flat_duration:N] = peak_value * np.exp(- (t[up_duration + flat_duration:N] - (up_duration + flat_duration)) / 150)  # 150控制衰减速度


# 将信号输出到Eacc.txt文件
np.savetxt('Eacc.txt', signal)

# 绘制信号
plt.plot(t, signal)
plt.title('Generated Pulse Signal')
plt.xlabel('Sample Index')
plt.ylabel('Signal Value')
plt.xlim(0, N)
plt.ylim(0, peak_value)
plt.grid()
plt.show()