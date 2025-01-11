import numpy as np

# 生成长度为2048的随机信号，范围是0到10
N = 256
random_signal = np.random.uniform(low=0.0, high=10.0, size=N)

# 将随机信号输出到Piezo_previous.txt文件
np.savetxt('Piezo_previous.txt', random_signal)
