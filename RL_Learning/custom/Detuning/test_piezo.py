import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from detuning_env import PiezoDetuningEnv
import matplotlib.pyplot as plt

# 环境注册
gym.register(id='PiezoDetuning-v0', entry_point=PiezoDetuningEnv)

# 加载最佳模型
best_model = PPO.load("./best_model/best_model", device='cuda')

# 创建测试环境
test_env = Monitor(gym.make('PiezoDetuning-v0', render_mode='human'))

# 运行测试
obs, _ = test_env.reset()
total_reward = 0
for _ in range(int(0.04 * 1e6)):  # 完整运行一个周期
    action, _ = best_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Total Reward: {total_reward:.2f}")
plt.ioff()
plt.show()