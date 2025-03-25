import RFEnv

env = RFEnv.RFEnv()
obs, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # 随机动作
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        env.render()
        done = True
env.close()