import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 传递函数的分子和分母系数
b = [1]
a = [1, 1]

# 计算频率响应
w, h = signal.freqs(b, a)

# 绘制幅频响应
plt.figure()
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.grid(which='both', axis='both')
plt.show()

# 绘制相频响应
angles = np.angle(h)
plt.figure()
plt.semilogx(w, angles)
plt.title('Phase response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Phase [radians]')
plt.grid(which='both', axis='both')
plt.show()
