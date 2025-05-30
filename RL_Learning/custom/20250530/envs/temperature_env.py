import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TemperatureControlEnv(gym.Env):
    """A custom environment for controlling room temperature."""
    
    def __init__(self, initial_temperature, target_temperature, max_steps, temperature_change, tolerance):
        super(TemperatureControlEnv, self).__init__()
        self.action_space = spaces.Discrete(2)  # 0: decrease temp, 1: increase temp
        self.observation_space = spaces.Box(low=0, high=30, shape=(1,), dtype=np.float32)
        self.initial_temperature = initial_temperature
        self.target_temperature = target_temperature
        self.max_steps = max_steps
        self.temperature_change = temperature_change
        self.tolerance = tolerance
        self.temperature = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.temperature = self.initial_temperature
        self.current_step = 0
        return np.array([self.temperature], dtype=np.float32), {}

    def step(self, action):
        if action == 0:
            self.temperature -= self.temperature_change
        else:
            self.temperature += self.temperature_change
        
        reward = -abs(self.temperature - self.target_temperature)
        self.current_step += 1
        
        done = self.current_step >= self.max_steps or abs(self.temperature - self.target_temperature) < self.tolerance
        truncated = self.current_step >= self.max_steps
        
        return np.array([self.temperature], dtype=np.float32), reward, done, truncated, {}

    def render(self):
        print(f"Current Temperature: {self.temperature:.1f}°C (Target: {self.target_temperature}°C)")