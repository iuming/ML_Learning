import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt

# 必须导入自定义环境模块以完成注册
from multisin_env import MultiSinEnv  # 确保路径正确

# 注册环境（如果未在模块中自动注册）
gym.register(id='MultiSinEnv-v0', entry_point=MultiSinEnv)

# 加载最佳模型（强制使用CPU）
best_model = PPO.load("./best_model/best_model", device='cuda')

# 创建测试环境（注意名称带-v0）
test_env = Monitor(gym.make('MultiSinEnv-v0', render_mode='human'))

# 评估
mean_reward, std_reward = evaluate_policy(
    best_model,
    test_env,
    n_eval_episodes=10,
    deterministic=True,
    render=False
)
print(f"Best Model Performance:")
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# 演示
obs, _ = test_env.reset()
total_reward = 0
for _ in range(1000):
    action, _ = best_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Demo Total Reward: {total_reward:.1f}")
plt.ioff()
plt.show()