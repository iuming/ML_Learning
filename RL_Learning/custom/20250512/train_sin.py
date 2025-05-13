import gymnasium as gym
import torch
from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import make_vec_env, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from sin_env import SinEnv

# register custom env
gym.register(id='SinEnv-v0', entry_point='sin_env:SinEnv')

# factory for vectorized, normalized env
def make_env():
    env = gym.make('SinEnv-v0')
    return Monitor(env)

vec_env = make_vec_env(make_env, n_envs=8)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

policy_kwargs = dict(
    net_arch=[256, 256],
    activation_fn=torch.nn.Tanh
)

model = PPO(
    "MlpPolicy",
    vec_env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=256,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.001,
    clip_range=0.2,
    verbose=1,
    tensorboard_log="./ppo_sin_tb/"
)

eval_callback = EvalCallback(
    vec_env,
    best_model_save_path="./best_model/",
    log_path="./logs/",
    eval_freq=10_000,
    n_eval_episodes=5,
    deterministic=True,
    render=False,
    verbose=1
)

model.learn(
    total_timesteps=2_000_000,
    callback=eval_callback,
    tb_log_name="PPO_sin"
)
model.save("ppo_sin_final")