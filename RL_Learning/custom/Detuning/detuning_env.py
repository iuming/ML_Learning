import gymnasium as gym
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt


class PiezoDetuningEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        # 时间参数
        self.T = 0.04  # 40ms周期
        self.sampling_rate = 1e6
        self.max_steps = int(self.T * self.sampling_rate)
        self.dt = self.T / self.max_steps

        # 脉冲参数（调整参数量级以匹配动作范围）
        self.t_filling = 0.01
        self.t_flat = 0.03
        self.K = 1.0  # 原1e6改为1
        self.k = 0.5
        self.Eacc = 0.0
        self.tau = 0.005

        # 空间定义
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -2.0]),
            high=np.array([self.T, 2, 2.0]),
            shape=(3,),
            dtype=np.float32
        )

        # 渲染设置
        self.render_mode = render_mode
        self.fig = None
        self.line = None
        self.history_t = []
        self.history_ft = []
        self.current_cycle = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.step_count = 0
        self.phase_idx = 0
        self.Eacc = 0.0
        self.history_t = []
        self.history_ft = []
        self.current_cycle = 0

        return np.array([self.t, self.phase_idx, 0.0], dtype=np.float32), {}

    def _calculate_ft(self):
        if self.t < self.t_filling:
            self.phase_idx = 0
            return self.K * self.t ** 2
        elif self.t < self.t_flat:
            self.phase_idx = 1
            return self.k * self.Eacc ** 2
        else:
            self.phase_idx = 2
            decay_time = self.t - self.t_flat
            return self.k * self.Eacc ** 2 * np.exp(-decay_time / self.tau)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]

        # 更新Eacc（调整更新幅度）
        self.Eacc += 0.5 * action  # 放大动作影响

        # 计算f(t)
        ft = self._calculate_ft()

        # 计算补偿后的信号
        compensated = ft + action

        # 记录数据
        self.history_t.append(self.t)
        self.history_ft.append(compensated)

        # 计算奖励
        reward = -np.abs(compensated)

        # 更新状态
        self.t += self.dt
        self.step_count += 1

        # 周期结束判断
        terminated = False
        truncated = self.step_count >= self.max_steps
        if truncated:
            self.current_cycle += 1
            self.t = 0.0
            self.step_count = 0

        obs = np.array([self.t, self.phase_idx, compensated], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode is None:
            return

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(12, 6))
            self.ax = self.fig.add_subplot(111)
            self.line, = self.ax.plot([], [], 'b-')
            self.ax.set_xlim(0, self.T)
            self.ax.set_ylim(-2, 2)
            self.ax.set_xlabel('Time (s)')
            self.ax.set_ylabel('Compensated Signal')
            plt.title(f'Cycle {self.current_cycle} Compensation Effect')

        if self.step_count % 100 == 0:
            self.line.set_data(self.history_t, self.history_ft)
            self.ax.relim()
            self.ax.autoscale_view()
            self.ax.set_title(f'Cycle {self.current_cycle} Compensation Effect')

            if self.render_mode == 'human':
                plt.draw()
                plt.pause(0.001)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)