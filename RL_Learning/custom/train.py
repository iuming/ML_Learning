from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from custom_env import CustomEnv  # 假设自定义环境定义在 custom_env.py 中

# 创建环境
env = DummyVecEnv([lambda: CustomEnv()])

# 创建 PPO 模型
model = PPO("MlpPolicy", env, verbose=16, device="cuda")

model.load("ppo_custom_env")

# 训练模型
model.learn(total_timesteps=100000)

# 保存模型
model.save("ppo_custom_env")

# 测试模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()