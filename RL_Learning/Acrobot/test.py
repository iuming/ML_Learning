import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make('Acrobot-v1', render_mode="rgb_array")

model = PPO("MlpPolicy",
            env,
            verbose=4,
            device="cuda",
            tensorboard_log="./acrobot_tensorboard/",
            n_steps=4096,
            n_epochs=1024
            )

model.learn(total_timesteps=1e3,
            tb_log_name="first_run",
            progress_bar=True
            )

model.save("ppo_acrobot")

env.close()
