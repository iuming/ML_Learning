import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from detuning_env import PiezoDetuningEnv

# 环境注册
gym.register(id='PiezoDetuning-v0', entry_point=PiezoDetuningEnv)

# 创建训练环境
def make_env():
    env = gym.make('PiezoDetuning-v0', render_mode=None)
    return Monitor(env)

vec_env = make_vec_env(make_env, n_envs=8)

# 配置评估回调
eval_callback = EvalCallback(
    Monitor(gym.make('PiezoDetuning-v0')),
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10000,
    deterministic=True,
    verbose=1
)

# 初始化模型
model = PPO(
    "MlpPolicy",
    vec_env,
    policy_kwargs={"net_arch": [128, 128]},
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    device='cuda',
    tensorboard_log="./piezo_tensorboard/"
)

# 开始训练
model.learn(
    total_timesteps=500_000,
    callback=eval_callback,
    tb_log_name="ppo"
)

# 保存最终模型
model.save("piezo_detuning_final")