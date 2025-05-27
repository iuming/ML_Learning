"""
Name: env_cavity.py
version: 1.0

author: Ming Liu from IHEP
date: 2025-05-22

description:
This module defines a Gymnasium-compatible environment for simulating RF cavity systems,
intended for reinforcement learning research and development.

Features:
    - Simulates RF source, I/Q modulation, amplification, and cavity dynamics with microphonics.
    - Supports both pulsed and continuous wave (CW) operation modes.
    - Provides step, reset, render, and state management methods for RL workflows.

Dependencies:
    - gymnasium: RL environment interface.
    - numpy: Numerical operations.
    - matplotlib: Visualization.
    - llrflibs.rf_sim, llrflibs.rf_control: RF system simulation and control utilities.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
from env_cavity import Cavity_env


# 注册自定义环境
gym.register(id='Cavity_env-v0', entry_point='env_cavity:Cavity_env')

# 自定义提前停止回调
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, reward_threshold: float, check_freq: int, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 获取最近10个episode的平均奖励
            rewards = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            if len(rewards) > 0:
                mean_reward = np.mean(rewards)
                if mean_reward > self.reward_threshold:
                    print(f"Early stopping triggered! Mean reward {mean_reward:.2f} > {self.reward_threshold}")
                    return False  # 返回False会停止训练
        return True

# 创建环境
def make_env():
    env = gym.make('Cavity_env-v0', render_mode=None)
    return Monitor(env)

vec_env = make_vec_env(make_env, n_envs=4)

# 配置回调
eval_callback = EvalCallback(
    Monitor(gym.make('Cavity_env-v0')),
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    verbose=1
)

early_stop_callback = EarlyStoppingCallback(
    reward_threshold=-0.1,  # 当平均奖励超过-0.1时停止
    check_freq=1000         # 每1000步检查一次
)

# 初始化模型
model = PPO(
    "MlpPolicy",
    vec_env,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.001,
    tensorboard_log="./ppo_sin_tensorboard/"
)

# 训练模型
model.learn(
    total_timesteps=1_000_000,
    callback=[eval_callback, early_stop_callback],
    tb_log_name="ppo"
)

# 显式保存最终模型
model.save("ppo_sin_final")