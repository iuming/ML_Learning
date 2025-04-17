import gymnasium as gym
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


class MultiSinEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.max_steps = 1000000
        self.dt = 2 * np.pi / (self.max_steps - 1)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-2.0, -1.0, -1.0]),
            high=np.array([2.0, 1.0, 1.0]),
            shape=(3,),
            dtype=np.float32
        )

        # 渲染设置
        self.render_mode = render_mode
        self.fig = None
        self.line = None
        self.history_t = []
        self.history_ft = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.step_count = 0
        self.history_t = []
        self.history_ft = []
        observation = np.array([
            np.sin(self.t),  # f(t)
            np.sin(self.t),  # sin(t)
            np.cos(self.t)  # cos(t)
        ], dtype=np.float32)
        return observation, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # 多sin函数
        current_sin = np.sin(self.t) + 0.5 * np.sin(2 * self.t) + 0.25 * np.sin(3 * self.t)

        # 归一化处理
        min_value = -1.75  # np.sin(self.t) + 0.5 * np.sin(2 * self.t) + 0.25 * np.sin(3 * self.t) 的最小值
        max_value = 1.75   # np.sin(self.t) + 0.5 * np.sin(2 * self.t) + 0.25 * np.sin(3 * self.t) 的最大值
        current_sin = (current_sin - min_value) / (max_value - min_value) * 2 - 1.25 * np.sin(3 * self.t)

        # 构建观测
        obs = np.array([
            current_sin + action[0],  # f(t)
            current_sin,  # sin(t)
            np.cos(self.t)  # cos(t)
        ], dtype=np.float32)

        # 记录数据
        self.history_t.append(self.t)
        self.history_ft.append(obs[0].item())

        # 计算奖励
        reward = -np.abs(obs[0])

        # 更新状态
        self.t += self.dt
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        # 创建或更新图像
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 5))
            self.ax = self.fig.add_subplot(111)
            self.line, = self.ax.plot([], [], 'b-')
            self.ax.set_xlim(0, 2 * np.pi)
            self.ax.set_ylim(-2, 2)
            self.ax.set_xlabel('t')
            self.ax.set_ylabel('f(t)')
            plt.title('Sin Wave with Actions')

        # 更新数据
        self.line.set_data(self.history_t, self.history_ft)
        self.ax.relim()
        self.ax.autoscale_view()

        if self.render_mode == 'human':
            plt.draw()
            plt.pause(0.001)
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None