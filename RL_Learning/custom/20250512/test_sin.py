import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from datetime import datetime
import matplotlib.pyplot as plt
from sin_env import SinEnv

# register env
gym.register(id='SinEnv-v0', entry_point='sin_env:SinEnv')

# load best model
model = PPO.load("./best_model/best_model", device='cpu')

# create test env
test_env = Monitor(gym.make('SinEnv-v0', render_mode='human'))

# evaluate
mean_reward, std_reward = evaluate_policy(
    model,
    test_env,
    n_eval_episodes=10,
    deterministic=True
)
print(f"Best Model Performance:")
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")

# demo rollout
obs, _ = test_env.reset()
actions, rewards, observs = [], [], []
for _ in range(10000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = test_env.step(action)
    actions.append(action[0])
    rewards.append(reward)
    observs.append(obs[3])
    if done:
        break

print(f"Demo Total Reward: {sum(rewards):.1f}")

# plot
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1); plt.plot(actions); plt.title('Actions')
plt.subplot(3,1,2); plt.plot(rewards); plt.title('Rewards')
plt.subplot(3,1,3); plt.plot(observs); plt.title('Observations')
plt.tight_layout()

# save
# ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# # steps = sum([test_env.num_envs * model.n_steps])
# steps = model.n_steps
# lr = model.learning_rate
# fname = f"results/test_results_{ts}_steps_{steps}_lr_{lr:.1e}.png"
# plt.savefig(fname)
# print(f"Plot saved as {fname}")
plt.show()
