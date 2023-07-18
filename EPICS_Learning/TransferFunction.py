# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt
#
# # 定义传递函数
# num = [1]
# den = [1, 1, 1]
#
# # 计算频率响应
# w, mag, phase = signal.bode((num, den))
#
# # 绘制Bode图
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.semilogx(w, mag)  # 绘制幅度响应
# plt.grid(True)
# plt.ylabel('Magnitude (dB)')
#
# plt.subplot(2, 1, 2)
# plt.semilogx(w, phase)  # 绘制相位响应
# plt.grid(True)
# plt.xlabel('Frequency (rad/s)')
# plt.ylabel('Phase (degrees)')
#
# # 绘制奈奎斯特图
# plt.figure()
# real, imag, freq = signal.freqresp((num, den))
# plt.plot(real, imag)
# plt.xlabel('Real')
# plt.ylabel('Imaginary')
# plt.grid(True)
#
# # 显示图形
# plt.show()




# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt
#
# # 定义传递函数
# num = [1]
# den = [1, 1, 1]
#
# # 计算频率响应
# w, mag, phase = signal.bode((num, den))
#
# # 绘制Bode图
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.semilogx(w, mag)  # 绘制幅度响应
# plt.grid(True)
# plt.ylabel('Magnitude (dB)')
#
# plt.subplot(2, 1, 2)
# plt.semilogx(w, phase)  # 绘制相位响应
# plt.grid(True)
# plt.xlabel('Frequency (rad/s)')
# plt.ylabel('Phase (degrees)')
#
# # 绘制奈奎斯特图
# plt.figure()
# w, h = signal.freqresp((num, den))
# plt.plot(h.real, h.imag)
# plt.xlabel('Real')
# plt.ylabel('Imaginary')
# plt.grid(True)
#
# # 显示图形
# plt.show()




# import numpy as np
# from scipy import signal
# import matplotlib.pyplot as plt
#
# # 定义传递函数
# num = [1]
# den = [1, 1, 1]
#
# # 计算频率响应
# w, mag, phase = signal.bode((num, den))
#
# # 绘制Bode图
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.semilogx(w, mag)  # 绘制幅度响应
# plt.grid(True)
# plt.ylabel('Magnitude (dB)')
#
# plt.subplot(2, 1, 2)
# plt.semilogx(w, phase)  # 绘制相位响应
# plt.grid(True)
# plt.xlabel('Frequency (rad/s)')
# plt.ylabel('Phase (degrees)')
#
# # 绘制Nyquist图
# plt.figure()
# signal.nyquist((num, den))
# plt.grid(True)
#
# # 显示图形
# plt.show()




import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 定义传递函数
num = [1]
den = [1, 1, 1]

# 计算频率响应
w, mag, phase = signal.bode((num, den))

# 绘制Bode图
plt.figure()
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)  # 绘制幅度响应
plt.grid(True)
plt.ylabel('Magnitude (dB)')

plt.subplot(2, 1, 2)
plt.semilogx(w, phase)  # 绘制相位响应
plt.grid(True)
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Phase (degrees)')

# 绘制Nyquist图
plt.figure()
t = np.linspace(0, 2 * np.pi, 1000)
H = (np.cos(t) + 1j * np.sin(t)) / (t + 1j)  # 传递函数的频率响应
plt.plot(H.real, H.imag)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)

# 显示图形
plt.show()