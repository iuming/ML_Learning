import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np

# 注册自定义环境
gym.register(id='MultiSinEnv-v0', entry_point='multisin_env:MultiSinEnv')

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
    env = gym.make('MultiSinEnv-v0', render_mode=None)
    return Monitor(env)

vec_env = make_vec_env(make_env, n_envs=4)

# 配置回调
eval_callback = EvalCallback(
    Monitor(gym.make('MultiSinEnv-v0')),
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
    tensorboard_log="./ppo_multisin_tensorboard/"
)

# 训练模型
model.learn(
    total_timesteps=2_000_000,
    callback=[eval_callback, early_stop_callback],
    tb_log_name="ppo"
)

# 显式保存最终模型
model.save("ppo_sin_final")