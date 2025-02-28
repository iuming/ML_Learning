# Program Name: example_gpu
# Author: Ming Liu
# Created Time: November, 9, 2024, 12:31 am

# Modified Time: February, 28, 2025, 20:07 pm
# Modified Author: Ming Liu
# Modified Notes: Add a new example of using GPU to train a model.

import gymnasium as gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1, device='cuda:0', tensorboard_log="./a2c_cartpole_tensorboard/")
# model.load("cartpole_a2c")
model.learn(total_timesteps=1000_000)
model.save("cartpole_a2c")

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    # print(reward)
    vec_env.render("human")
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()