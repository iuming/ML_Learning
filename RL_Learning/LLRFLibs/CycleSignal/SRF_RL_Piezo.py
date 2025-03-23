import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

from llrflibs.rf_sim import *
from llrflibs.rf_control import *


def sim_cav(half_bw, RL, dw_step0, detuning0, vf_step, state_vc, Ts, beta=1e4,
            state_m0=0, Am=None, Bm=None, Cm=None, Dm=None,
            pulsed=True, beam_pul=None, beam_cw=0, buf_id=0):
    # get the beam
    if pulsed:
        vb = -RL * beam_pul[buf_id if buf_id < len(beam_pul) else -1]
    else:
        vb = beam_cw

    # execute for one step
    status, vc, vr, dw, state_m = sim_scav_step(half_bw,
                                                dw_step0,
                                                detuning0,
                                                vf_step,
                                                vb,
                                                state_vc,
                                                Ts,
                                                beta=beta,
                                                state_m0=state_m0,
                                                Am=Am,
                                                Bm=Bm,
                                                Cm=Cm,
                                                Dm=Dm,
                                                mech_exe=True)
    state_vc = vc

    # return
    return vc, vr, dw, state_vc, state_m


class SuperconductingCavityEnv(gym.Env):
    def __init__(self, pulse_cycles=25):
        super().__init__()

        # 环境参数
        self.Ts = 1e-6  # 仿真时间步长
        self.pulse_cycles = pulse_cycles  # 每个episode的脉冲周期数
        self.current_cycle = 0  # 当前脉冲周期计数器

        # 按照正确的顺序初始化参数
        self._init_mechanical_modes()  # 必须先初始化机械模式
        self._init_cavity_parameters()  # 然后初始化腔体参数
        self._init_simulation_parameters()  # 最后初始化仿真参数

        # 定义动作和观测空间
        self.action_space = spaces.Box(
            low=-2 * np.pi * 100,  # -100 Hz
            high=2 * np.pi * 100,  # +100 Hz
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -2 * np.pi * 500]),
            high=np.array([20e6, np.pi, 2 * np.pi * 500]),  # 幅值、相位、失谐量
            dtype=np.float32
        )

        # 数据收集
        self.history = {
            'vc_amp': [],
            'vc_phase': [],
            'detuning': [],
            'actions': []
        }

    def _init_mechanical_modes(self):
        """初始化机械振动模式参数"""
        # 机械模式参数
        self.mech_modes = {
            'f': [280, 341, 460, 487, 618],
            'Q': [40, 20, 50, 80, 100],
            'K': [2, 0.8, 2, 0.6, 0.2]
        }

        # 状态空间模型
        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        if not status:
            raise RuntimeError("机械模式状态空间模型创建失败")

        status, self.Ad, self.Bd, self.Cd, self.Dd, _ = ss_discrete(
            Am, Bm, Cm, Dm, Ts=self.Ts, method='zoh', plot=False
        )
        if not status:
            raise RuntimeError("离散化状态空间模型创建失败")


    def _init_simulation_parameters(self):
        """初始化仿真参数（需要在机械模式之后初始化）"""
        # 原始仿真参数
        self.t_fill = 510
        self.t_flat = 1300
        self.buf_size = 2048 * 8
        self.base_pul = np.zeros(self.buf_size, dtype=complex)
        self.base_pul[:self.t_flat] = 1.0
        self.beam_pul = np.zeros(self.buf_size, dtype=complex)
        self.beam_pul[self.t_fill:self.t_flat] = 0.008

        # RF参数
        self.fsrc = -460  # Hz
        self.Asrc = 1.0  # V
        self.pha_src = 0.0  # rad
        self.gain_dB = 20 * np.log10(12e6)

        # 仿真状态
        self.buf_id = 0
        self.state_vc = 0.0 + 0j
        self.state_m = np.matrix(np.zeros(self.Bd.shape))

        # 状态空间模型
        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        status, self.Ad, self.Bd, self.Cd, self.Dd, _ = ss_discrete(
            Am, Bm, Cm, Dm, Ts=self.Ts, method='zoh', plot=False
        )

    def _init_cavity_parameters(self):
        # 腔体参数
        self.f0 = 1.3e9  # Hz
        self.beta = 1e4
        self.roQ = 1036  # Ohm
        self.QL = 3e6
        self.RL = 0.5 * self.roQ * self.QL
        self.wh = np.pi * self.f0 / self.QL
        self.dw0 = 0.0

    def reset(self, seed=None, options=None):
        # 重置仿真状态
        super().reset(seed=seed)

        self.buf_id = 0
        self.current_cycle = 0
        self.state_vc = np.complex64(0.0)  # 使用numpy复数类型
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.pha_src = 0.0
        self.dw_current = np.float32(0.0)  # 使用numpy浮点类型

        # 清空历史数据
        self.history = {k: [] for k in self.history}

        return self._get_obs(), {}

    def step(self, action):
        # 应用Piezo补偿
        dw_piezo = action[0]

        # 执行一个完整的脉冲周期
        cycle_rewards = []
        for _ in range(self.buf_size):
            # 执行单个时间步仿真
            self._simulation_step(dw_piezo)

            # 收集数据
            self._record_data()

            # 计算即时奖励
            cycle_rewards.append(-abs(self.dw_current / (2 * np.pi)))

        # 更新周期计数器
        self.current_cycle += 1

        # 计算总奖励
        reward = np.mean(cycle_rewards)

        # 检查终止条件
        terminated = self.current_cycle >= self.pulse_cycles
        truncated = False

        return (
            self._get_obs().flatten(),  # 确保输出形状是(3,)
            reward,
            terminated,
            truncated,
            {}
        )

    def _simulation_step(self, dw_piezo):
        # 生成微扰
        dw_micr = 2.0 * np.pi * self.np_random.normal(0, 10)

        # RF信号链
        S0, self.pha_src = self._sim_rfsrc()
        S1 = self._sim_iqmod(S0)
        S2 = self._sim_amp(S1)

        # 执行腔体仿真
        self.vc, vr, self.dw_current, self.state_vc, self.state_m = sim_cav(
            self.wh,
            self.RL,
            self.dw_current,  # 直接使用已初始化的属性
            self.dw0 + dw_micr + dw_piezo,
            S2,
            self.state_vc,
            self.Ts,
            beta=self.beta,
            state_m0=self.state_m,
            Am=self.Ad,
            Bm=self.Bd,
            Cm=self.Cd,
            Dm=self.Dd,
            pulsed=True,
            beam_pul=self.beam_pul,
            buf_id=self.buf_id
        )

        # 更新缓冲区ID
        self.buf_id = (self.buf_id + 1) % self.buf_size

    def _get_obs(self):
        vc_amp = np.abs(np.array([self.state_vc], dtype=complex))[0]
        vc_phase = np.angle(np.array([self.state_vc], dtype=complex))[0]
        dw = self.dw_current / (2 * np.pi)

        return np.array([vc_amp, vc_phase, dw], dtype=np.float32)

    def _record_data(self):
        self.history['vc_amp'].append(abs(self.state_vc) * 1e-6)
        self.history['vc_phase'].append(np.angle(self.state_vc) * 180 / np.pi)
        self.history['detuning'].append(self.dw_current / (2 * np.pi))
        self.history['actions'].append(self.dw_current / (2 * np.pi))

    def _sim_rfsrc(self):
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        return self.Asrc * np.exp(1j * pha), pha

    def _sim_iqmod(self, S0):
        return S0 * self.base_pul[self.buf_id]

    def _sim_amp(self, S1):
        return S1 * 10 ** (self.gain_dB / 20.0)

    def render(self, mode='human'):
        # 实时可视化
        if not hasattr(self, 'fig'):
            self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 8))
            plt.ion()
            plt.show()

        titles = ['Cavity Voltage (MV)', 'Phase (deg)', 'Detuning (Hz)', 'Piezo Adjustment (Hz)']
        for i, ax in enumerate(self.axs):
            ax.clear()
            ax.set_title(titles[i])
            ax.grid(True)

            if i < 3:
                data = list(self.history.values())[i]
                ax.plot(data[-1000:])
            else:
                ax.plot(self.history['actions'][-1000:])

        plt.pause(0.001)


# 训练配置
if __name__ == "__main__":
    env = SuperconductingCavityEnv()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./tb_logs/"
    )

    # 训练并定期保存模型
    try:
        model.learn(total_timesteps=1e6, progress_bar=True)
    except KeyboardInterrupt:
        pass

    model.save("sc_cavity_control")
    env.close()