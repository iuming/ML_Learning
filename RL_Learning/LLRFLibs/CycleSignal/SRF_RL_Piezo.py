import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from llrflibs.rf_sim import *
from llrflibs.rf_control import *

class SuperconductingCavityEnv(gym.Env):
    def __init__(self, cycle_length=2048):
        super(SuperconductingCavityEnv, self).__init__()
        self.cycle_length = cycle_length  # 周期长度

        # 修改动作空间为一组信号
        self.action_space = gym.spaces.Box(
            low=-1e6, high=1e6, shape=(self.cycle_length,), dtype=np.float32
        )

        # 修改观测空间（假设每个时间步返回腔电压幅值、相位和piezo调整量）
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cycle_length, 3), dtype=np.float32
        )

        # 初始化其他参数（根据实际仿真模型调整）
        self.Ts = 1e-6  # 时间步长示例
        self.buf_size = 1000  # 缓冲区大小示例
        self.beta = 1.0  # 示例参数
        self.RL = 1.0  # 示例参数
        self.wh = 1e6  # 示例参数
        self.dw0 = 0.0  # 示例参数
        self.fsrc = 1e9  # 示例参数
        self.Asrc = 1.0  # 示例参数
        self.pha_src = 0.0  # 示例参数
        self.base_pul = np.ones(self.buf_size)  # 示例脉冲
        self.beam_pul = np.zeros(self.buf_size)  # 示例束流
        self.Ad = np.zeros((2, 2))  # 示例矩阵
        self.Bd = np.zeros((2, 1))  # 示例矩阵
        self.Cd = np.zeros((1, 2))  # 示例矩阵
        self.Dd = np.zeros((1, 1))  # 示例矩阵

        # 初始化数据收集列表
        self.vc_abs_list = []
        self.vc_phase_list = []
        self.dw_list = []
        self.action_list = []  # 新增：存储动作

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 重置状态
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.buf_id = 0
        self.dw_piezo_cycle = np.zeros(self.cycle_length)

        # 清空数据收集列表
        self.vc_abs_list = []
        self.vc_phase_list = []
        self.dw_list = []
        self.action_list = []  # 清空动作列表

        return self._get_obs_cycle(), {}

    def step(self, action):
        # action 是一个周期的piezo补偿信号，形状为 (cycle_length,)
        self.dw_piezo_cycle = action
        self.action_list.extend(action)  # 收集动作数据

        # 模拟一个周期
        obs_cycle = []
        dw_list = []
        for i in range(self.cycle_length):
            dw_piezo = self.dw_piezo_cycle[i]
            # 模拟一个时间步（这里调用仿真腔模型）
            dw_micr = 2.0 * np.pi * np.random.randn() * 10  # 示例微扰
            vb = -self.RL * self.beam_pul[self.buf_id]
            S0, self.pha_src = self.sim_rfsrc(self.fsrc, self.Asrc, self.pha_src, self.Ts)
            S1 = self.sim_iqmod(S0, pulsed=True, base_pul=self.base_pul, base_cw=1, buf_id=self.buf_id)
            S2 = self.sim_amp(S1, gain_dB=20 * np.log10(12e6))
            status, self.state_vc, vr, dw, self.state_m = self.sim_scav_step(
                self.wh, 0 , self.dw0 + dw_micr + dw_piezo, S2, vb, self.state_vc, self.Ts,
                beta=self.beta, state_m0=self.state_m, Am=self.Ad, Bm=self.Bd, Cm=self.Cd, Dm=self.Dd
            )
            if not status:
                raise RuntimeError("Simulation failed")
            self.buf_id += 1
            if self.buf_id >= self.buf_size:
                self.buf_id = 0
            obs_cycle.append(self._get_obs(i))
            dw_list.append(abs(dw))

            # 收集数据
            self.vc_abs_list.append(abs(self.state_vc) * 1e-6)
            self.vc_phase_list.append(np.angle(self.state_vc) * 180 / np.pi)
            self.dw_list.append(dw / (2 * np.pi))

        # 计算奖励：基于周期内失谐量的平均值
        avg_dw = np.mean(dw_list)
        reward = -avg_dw

        # 判断是否结束
        done = self.buf_id == 0
        truncated = False

        return np.array(obs_cycle), reward, done, truncated, {}

    def _get_obs(self, i):
        # 返回单步观测
        return np.array([
            abs(self.state_vc) * 1e-6,  # 腔电压幅值
            np.angle(self.state_vc) * 180 / np.pi,  # 腔电压相位
            self.dw_piezo_cycle[i]  # 当前时间步的piezo调整量
        ])

    def _get_obs_cycle(self):
        # 返回一个周期的初始观测
        return np.array([self._get_obs(i) for i in range(self.cycle_length)])

    # 仿真腔模型的占位函数
    def sim_rfsrc(self, fsrc, Asrc, pha_src, Ts):
        pha = pha_src + 2.0 * np.pi * fsrc * Ts
        return Asrc * np.exp(1j * pha), pha

    def sim_iqmod(self, S0, pulsed, base_pul, base_cw, buf_id):
        if pulsed:
            return S0 * base_pul[buf_id if buf_id < len(base_pul) else -1]
        else:
            return S0 * base_cw

    def sim_amp(self, S1, gain_dB):
        return S1 * 10 ** (gain_dB / 20)

    def sim_scav_step(self, wh, dw_piezo, dw_total, S2, vb, state_vc, Ts, **kwargs):
        # 示例实现，返回状态、腔电压、反射电压、失谐量和状态矩阵
        dw = dw_total - dw_piezo
        state_vc += (S2 + vb - dw * state_vc) * Ts
        vr = 0.0  # 占位
        state_m = kwargs["state_m0"]
        return True, state_vc, vr, dw, state_m

# 创建环境
env = SuperconductingCavityEnv()

# 创建PPO智能体
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard", device="cuda")

# 如果需要加载已有模型
model.load("superconducting_cavity_controller")

# 设置Matplotlib为交互模式
plt.ion()

# 创建图形，包含四个子图
fig, axs = plt.subplots(4, 1, figsize=(10, 10))
axs[0].set_title('Cavity Voltage Amplitude (MV)')
axs[1].set_title('Cavity Phase (deg)')
axs[2].set_title('Detuning (Hz)')
axs[3].set_title('Action (Piezo Adjustment)')  # 新增动作子图
for ax in axs:
    ax.set_xlabel('Time Step')

# 训练智能体
total_timesteps = 1000000
timesteps_per_episode = env.cycle_length
num_episodes = total_timesteps // timesteps_per_episode

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        if done or truncated:
            break

    # 更新曲线
    axs[0].plot(env.vc_abs_list, label=f'Episode {episode+1}')
    axs[1].plot(env.vc_phase_list, label=f'Episode {episode+1}')
    axs[2].plot(env.dw_list, label=f'Episode {episode+1}')
    axs[3].plot(env.action_list, label=f'Episode {episode+1}')  # 绘制动作

    # 在第一个episode后添加图例
    if episode == 0:
        for ax in axs:
            ax.legend()

    # 刷新图形
    fig.canvas.draw()
    fig.canvas.flush_events()

# 关闭交互模式并显示最终图形
plt.ioff()
plt.show()

# 保存模型
model.save("superconducting_cavity_controller")

# 评估智能体（可选）
obs, info = env.reset()
for _ in range(env.buf_size):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    print(f"Observation: {obs}, Reward: {rewards}")
    if done or truncated:
        break