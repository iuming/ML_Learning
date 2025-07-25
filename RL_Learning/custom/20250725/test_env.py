import gymnasium as gym
import numpy as np
from sin_env import SinEnv

# 测试环境
print("Creating environment...")
env = SinEnv()

print("Resetting environment...")
obs, info = env.reset()
print(f"Initial observation: {obs}")
print(f"Observation shape: {obs.shape}")
print(f"Observation dtype: {obs.dtype}")
print(f"Any NaN in observation: {np.any(np.isnan(obs))}")
print(f"Any Inf in observation: {np.any(np.isinf(obs))}")

print("\nTesting a few steps...")
for i in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}:")
    print(f"  Action: {action}")
    print(f"  Observation: {obs}")
    print(f"  Reward: {reward}")
    print(f"  Any NaN in obs: {np.any(np.isnan(obs))}")
    print(f"  Any Inf in obs: {np.any(np.isinf(obs))}")
    print(f"  Any NaN in reward: {np.isnan(reward)}")
    print(f"  Any Inf in reward: {np.isinf(reward)}")
    
    if terminated or truncated:
        print("Episode ended")
        break

print("Environment test completed successfully!")
