import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

# 自定义环境
class CustomEnv(gym.Env):
    """自定义强化学习环境，观测空间为 F(x) = sin(x) + 0.5cos(x) + x + action，奖励为 -F(x)"""
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.x = 0.0
        self.current_step = 0
        self.max_steps = 200
        self.current_obs = None  # 用于记录当前观测值
        self.current_action = None  # 用于记录当前动作

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = 0.0
        self.current_step = 0
        observation = np.array([np.sin(self.x) + 0.5 * np.cos(self.x) + self.x + 0.0], dtype=np.float32)
        self.current_obs = observation  # 初始化观测值
        self.current_action = None  # 重置动作
        info = {}
        return observation, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action_value = action[0]
        self.x = self.x + action_value
        self.current_step += 1
        observation_value = np.sin(self.x) + 0.5 * np.cos(self.x) + self.x + action_value
        observation = np.array([observation_value], dtype=np.float32)
        reward = -observation_value
        terminated = False
        truncated = self.current_step >= self.max_steps
        info = {}
        self.current_obs = observation  # 更新当前观测值
        self.current_action = action  # 更新当前动作
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

# 自定义回调类
class DataCollectCallback(BaseCallback):
    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env  # 传入环境实例
        self.observations = []  # 存储观测值
        self.actions = []  # 存储动作

    def _on_step(self) -> bool:
        # 从环境中获取当前的观测值和动作
        if self.env.current_obs is not None and self.env.current_action is not None:
            self.observations.append(self.env.current_obs[0])  # 提取标量值
            self.actions.append(self.env.current_action[0])  # 提取标量值
        return True  # 继续训练

# 实例化环境和模型
env = CustomEnv()
callback = DataCollectCallback(env)
model = SAC("MlpPolicy", env, verbose=1)

# 训练模型
total_timesteps = 10000
model.learn(total_timesteps=total_timesteps, callback=callback)

print(f"训练完成，共进行了 {total_timesteps} 步。")

# 可视化数据
observations = np.array(callback.observations)
actions = np.array(callback.actions)

plt.figure(figsize=(10, 5))
plt.plot(observations, label='F(x)')
plt.xlabel('Step')
plt.ylabel('F(x)')
plt.title('F(x) over Steps')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(actions, label='Action', color='orange')
plt.xlabel('Step')
plt.ylabel('Action')
plt.title('Actions over Steps')
plt.legend()
plt.grid(True)
plt.show()

env.close()