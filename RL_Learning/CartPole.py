"""
Program Name: CartPole
Author: Liu Ming
Contact: ming-1018@foxmail.com
Version: 1.0
Created Date: 2023/12/12 下午3:13

Copyright (c) 2023 Liu Ming. All rights reserved.

Description: ...

Usage: Run the program and ...

Dependencies:
- Python 3.8 or above

Modifications:
- 2023/12/12 下午3:13: Initial Create.

"""

import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque

# 创建环境
env = gym.make('CartPole-v1')

# 设置随机种子
seed = 42
env.seed(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# 定义DQN模型
class DQN(tf.keras.Model):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(24, activation='relu')
        self.dense2 = tf.keras.layers.Dense(24, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_space, activation='linear')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

# 设置超参数
gamma = 0.95  # 折扣因子
epsilon = 1.0  # 探索率
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
memory = deque(maxlen=10000)

# 创建DQN模型
model = DQN(env.action_space.n)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

# 训练DQN模型
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    total_reward = 0
    for time in range(500):
        # 选择动作
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(state))
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        memory.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            break
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + gamma * np.amax(model.predict(next_state)[0]))
                target_f = model.predict(state)
                target_f[0][action] = target
                model.fit(state, target_f, epochs=1, verbose=0)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# 关闭环境
env.close()
