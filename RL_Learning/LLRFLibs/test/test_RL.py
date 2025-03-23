import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# 导入LLRFLibsPy的相关模块
from llrflibs.rf_sim import *
from llrflibs.rf_control import *


# 定义超导腔环境
class SuperconductingCavityEnv(gym.Env):
    def __init__(self):
        super(SuperconductingCavityEnv, self).__init__()

        # 初始化参数
        self.Ts = 1e-6
        self.buf_size = 2048 * 8
        self.t_fill = 510
        self.t_flat = 1300
        self.f0 = 1.3e9
        self.beta = 1e4
        self.roQ = 1036
        self.QL = 3e6
        self.RL = 0.5 * self.roQ * self.QL
        self.wh = np.pi * self.f0 / self.QL
        self.ib = 0.008
        self.dw0 = 0

        # 初始化机械模式
        mech_modes = {'f': [280, 341, 460, 487, 618],
                      'Q': [40, 20, 50, 80, 100],
                      'K': [2, 0.8, 2, 0.6, 0.2]}
        self.status, self.Am, self.Bm, self.Cm, self.Dm = cav_ss_mech(mech_modes)
        self.status, self.Ad, self.Bd, self.Cd, self.Dd, _ = ss_discrete(self.Am, self.Bm, self.Cm, self.Dm, Ts=self.Ts,
                                                                         method='zoh', plot=False)

        # 初始化状态
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.buf_id = 0
        self.beam_pul = np.zeros(self.buf_size, dtype=complex)
        self.beam_pul[self.t_fill:self.t_flat] = self.ib

        # 定义动作空间和观测空间
        self.action_space = gym.spaces.Box(low=-1e6, high=1e6, shape=(1,), dtype=np.float32)  # dw_piezo范围
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,),
                                                dtype=np.float32)  # 观测空间：腔电压、腔相位、失谐

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.buf_id = 0
        self.dw_piezo = 0.0  # 初始化piezo调整量
        self.total_detuning = 0.0  # 初始化总失谐
        return self._get_obs(), {}

    # 在 step 方法中调用 sim_scav_step
    def step(self, action):
        self.dw_piezo = action[0]  # 动作：调整piezo
        dw_micr = 2.0 * np.pi * np.random.randn() * 10  # 微音器失谐
        vb = -self.RL * self.beam_pul[self.buf_id]  # 束流驱动电压

        # 仿真RF源和I/Q调制器
        fsrc = -460  # RF源频率偏移
        Asrc = 1.0  # RF源幅度
        pha_src = 0.0  # 初始相位
        S0, pha_src = self.sim_rfsrc(fsrc, Asrc, pha_src, self.Ts)  # RF源输出
        S1 = self.sim_iqmod(S0, pulsed=True, base_pul=self.beam_pul, base_cw=1, buf_id=self.buf_id)  # I/Q调制器输出
        S2 = self.sim_amp(S1, gain_dB=20 * np.log10(12e6))  # 放大器输出

        # 仿真腔响应
        status, self.state_vc, vr, dw, self.state_m = sim_scav_step(
            self.wh, self.dw_piezo, self.dw0 + dw_micr, S2, vb, self.state_vc, self.Ts,
            beta=self.beta, state_m0=self.state_m, Am=self.Ad, Bm=self.Bd, Cm=self.Cd, Dm=self.Dd,
            mech_exe=True
        )

        if not status:
            raise RuntimeError("Simulation failed")

        # 更新状态
        self.buf_id += 1
        if self.buf_id >= self.buf_size:
            self.buf_id = 0

        # 计算奖励：总失谐越小，奖励越大
        self.total_detuning += abs(dw)
        reward = -self.total_detuning

        # 判断是否结束
        done = self.buf_id == 0
        truncated = False

        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        # 返回观测值：腔电压、腔相位、失谐
        return np.array([abs(self.state_vc) * 1e-6, np.angle(self.state_vc) * 180 / np.pi, self.dw_piezo])

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def sim_rfsrc(self, fsrc, Asrc, pha_src, Ts):
        pha = pha_src + 2.0 * np.pi * fsrc * Ts
        return Asrc * np.exp(1j * pha), pha

    def sim_iqmod(self, sig_in, pulsed=True, base_pul=None, base_cw=0, buf_id=0):
        if pulsed:
            sig_out = sig_in * base_pul[buf_id if buf_id < len(base_pul) else -1]
        else:
            sig_out = sig_in * base_cw
        return sig_out

    def sim_amp(self, sig_in, gain_dB):
        return sig_in * 10.0 ** (gain_dB / 20.0)


# 创建环境
env = SuperconductingCavityEnv()

# 创建PPO智能体
model = PPO("MlpPolicy", env, verbose=1)

# 加载模型
model.load("superconducting_cavity_controller")

# 训练智能体
model.learn(total_timesteps=100000000)

# 保存模型
model.save("superconducting_cavity_controller")

# 评估智能体
obs, info = env.reset()
for _ in range(env.buf_size):
    action, _states = model.predict(obs)
    obs, rewards, done, truncated, info = env.step(action)
    print(f"Observation: {obs}, Reward: {rewards}")
    if done or truncated:
        break