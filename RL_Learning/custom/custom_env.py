import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    def __init__(self):
        super(CustomEnv, self).__init__()
        self.x = None
        self.action_prev = 0
        self.t = 0
        self.max_steps = 200
        # 定义观察空间和动作空间
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def f(self, x):
        """自定义函数 f(x) = sin(x) + 0.5*cos(x)"""
        return np.sin(x) + 0.5 * np.cos(x)

    def reset(self, seed=None, options=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)  # 处理种子以确保可重复性
        self.x = np.random.uniform(-np.pi, np.pi)
        self.action_prev = 0
        self.t = 0
        observation = self.f(self.x) + self.x + self.action_prev
        return np.array([observation], dtype=np.float32), {}  # 返回观察值和信息字典

    def step(self, action):
        """执行一步环境更新"""
        action = action[0]  # 从动作数组中提取标量值
        reward = - (self.f(self.x) + self.x + action)
        self.x = self.x + 0.1 * action
        self.action_prev = action
        self.t += 1
        observation = self.f(self.x) + self.x + self.action_prev
        terminated = False  # 无特定终止条件
        truncated = self.t >= self.max_steps  # 时间限制
        info = {}
        return np.array([observation], dtype=np.float32), reward, terminated, truncated, info

    def render(self, mode='human'):
        """可选的渲染方法，此处未实现"""
        pass