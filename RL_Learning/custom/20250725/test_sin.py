import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
import matplotlib.pyplot as plt

# 必须导入自定义环境模块以完成注册
from sin_env import SinEnv  # 确保路径正确

# 注册环境（如果未在模块中自动注册）
gym.register(id='SinEnv-v0', entry_point=SinEnv)

# 加载最佳模型（强制使用CPU）
best_model = PPO.load("./best_model/best_model", device='cuda')

# 创建测试环境（注意名称带-v0）
test_env = Monitor(gym.make('SinEnv-v0', render_mode='human'))

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
actions = []
rewards = []
observations = []
step_count = 0
max_demo_steps = 32768  # 与环境的max_steps保持一致

for _ in range(max_demo_steps):
    action, _ = best_model.predict(obs, deterministic=True)
    # action = 0
    obs, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    observations.append(obs[3])
    actions.append(action)
    rewards.append(reward)
    step_count += 1
    
    if terminated or truncated:
        print(f"Episode finished at step {step_count}")
        break
    

print(f"Demo Total Reward: {total_reward:.1f}")
print(f"Average Reward per Step: {total_reward/step_count:.4f}")


# 绘制 action 和 reward (可能需要采样显示以避免图像过于密集)
sample_rate = max(1, len(actions) // 2000)  # 如果数据点太多，进行采样
sampled_indices = range(0, len(actions), sample_rate)
sampled_actions = [actions[i] for i in sampled_indices]
sampled_rewards = [rewards[i] for i in sampled_indices]
sampled_observations = [observations[i] for i in sampled_indices]
sampled_steps = list(sampled_indices)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(sampled_steps, sampled_actions)
plt.title(f'Actions (sampled every {sample_rate} steps)')
plt.xlabel('Step')
plt.ylabel('Action')

plt.subplot(3, 1, 2)
plt.plot(sampled_steps, sampled_rewards)
plt.title(f'Rewards (sampled every {sample_rate} steps)')
plt.xlabel('Step')
plt.ylabel('Reward')

plt.subplot(3, 1, 3)
plt.plot(sampled_steps, sampled_observations)
plt.title(f'Observations (sampled every {sample_rate} steps)')
plt.xlabel('Step')
plt.ylabel('Observation')

plt.tight_layout()

# 获取当前时间
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 获取模型的训练总步数和学习率
training_total_timesteps = best_model.n_steps * best_model.n_envs
learning_rate = best_model.learning_rate

# 构造文件名
file_name = f"results/test_results_{current_time}_steps_{training_total_timesteps}_lr_{learning_rate:.1e}.png"

# 保存图像
plt.savefig(file_name)
print(f"Plot saved as {file_name}")

plt.show()