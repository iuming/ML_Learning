import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义二阶微分方程的函数
def f(y, t, w, Q, k, E):
    y1, y2 = y
    dy1_dt = y2
    dy2_dt = -w**2*y1 - (w/Q)*y2 - w**2*k*E**2
    return [dy1_dt, dy2_dt]

# 定义初始条件
y0 = [0, 0]  # 初始位置和速度

# 定义时间范围
t = np.linspace(0, 10, 1000)

# 定义参数
w = 1.0
Q = 2.0
k = 0.5
E = 1.0

# 求解二阶微分方程
sol = odeint(f, y0, t, args=(w, Q, k, E))

# 提取解的位置
y = sol[:, 0]

# 绘制解的曲线
plt.plot(t, y)

# 添加标题和坐标轴标签
plt.title("Solution of y'' + (w/Q)y' + w^2y = -w^2kE^2")
plt.xlabel("t")
plt.ylabel("y")

# 显示图形
plt.show()