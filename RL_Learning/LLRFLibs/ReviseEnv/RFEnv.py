import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium import spaces

from llrflibs.rf_sim import *
from llrflibs.rf_control import *

def sim_rfsrc(fsrc, Asrc, pha_src, Ts):
    pha = pha_src + 2.0 * np.pi * fsrc * Ts
    return Asrc*np.exp(1j*pha), pha

def sim_iqmod(sig_in, pulsed = True, base_pul = None, base_cw = 0, buf_id = 0):
    if pulsed:
        sig_out = sig_in * base_pul[buf_id if buf_id < len(base_pul) else -1]
    else:
        sig_out = sig_in * base_cw
    return sig_out

def sim_amp(sig_in, gain_dB):
    return sig_in * 10.0**(gain_dB / 20.0)


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

class RFSystemEnv(gym.Env):
    def __init__(self):
        super(RFSystemEnv, self).__init__()

        # --- 通用参数 ---
        self.Ts = 1e-6  # 仿真时间步长，秒
        self.t_fill = 510  # 腔体填充阶段长度，采样点
        self.t_flat = 1300  # 平顶阶段结束时间，采样点
        self.buf_size = 2048 * 8  # 缓冲区大小
        self.pulsed = False  # 脉冲或连续波模式

        # --- RF源参数 ---
        self.fsrc = -460  # 相对于载波的频率偏移，Hz
        self.Asrc = 1  # RF源幅值，V
        self.pha_src = 0  # RF源相位状态变量

        # --- IQ调制器参数 ---
        self.base_pul = np.zeros(self.buf_size, dtype=complex)  # 基带信号缓冲区
        self.base_pul[:self.t_flat] = 1.0  # 填充基带信号
        self.base_cw = 1  # 连续波基带标量

        # --- 放大器参数 ---
        self.gain_dB = 20 * np.log10(12e6)  # 放大器增益，dB

        # --- 腔体参数 ---
        self.mech_modes = {
            'f': [280, 341, 460, 487, 618],  # 机械模式频率
            'Q': [40, 20, 50, 80, 100],  # 机械模式品质因数
            'K': [2, 0.8, 2, 0.6, 0.2]  # 机械模式耦合系数
        }
        self.f0 = 1.3e9  # RF工作频率，Hz
        self.beta = 1e4  # 耦合因子
        self.roQ = 1036  # 腔体的r/Q，Ohm
        self.QL = 3e6  # 加载品质因数
        self.RL = 0.5 * self.roQ * self.QL  # 加载电阻，Ohm
        self.wh = np.pi * self.f0 / self.QL  # 半带宽，rad/s
        self.ib = 0.008  # 平均束流电流，A
        self.dw0 = 2 * np.pi * 0  # 初始失谐量，rad/s
        self.beam_pul = np.zeros(self.buf_size, dtype=complex)  # 脉冲束流缓冲区
        self.beam_pul[self.t_fill:self.t_flat] = self.ib  # 填充束流信号
        self.beam_cw = 0  # 连续波束流

        # --- 机械模式状态空间 ---
        status, self.Am, self.Bm, self.Cm, self.Dm = cav_ss_mech(self.mech_modes)
        status, self.Ad, self.Bd, self.Cd, self.Dd, _ = ss_discrete(
            self.Am, self.Bm, self.Cm, self.Dm, Ts=self.Ts, method='zoh'
        )

        # --- 状态变量 ---
        self.state_m = np.matrix(np.zeros(self.Bd.shape))  # 机械模式状态
        self.state_vc = 0.0  # 腔体电压状态
        self.buf_id = 0  # 缓冲区索引
        self.dw = 0  # 当前失谐量

        # --- 历史记录（用于reward和render） ---
        self.vc_history = []  # 腔体电压历史
        self.dw_history = []  # 失谐量历史
        self.dw_piezo_history = []  # 压电补偿历史

        # --- 动作空间 ---
        # action是dw_piezo，一个标量（每次step更新一个值）
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

        # --- 观测空间 ---
        # obs包括：腔体幅值(1)、相位(1)、失谐量(1)、beam_pul(1)、RF源参数(fsrc, Asrc, pha_src共3)
        # 这里假设beam_pul和RF源参数取当前值
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
        )

    def reset(self, seed=None):
        """重置环境到初始状态"""
        super().reset(seed=seed)
        self.pha_src = 0
        self.buf_id = 0
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_state_vc = 0.0
        self.dw = 0
        self.vc_history = []
        self.dw_history = []
        self.dw_piezo_history = []
        return self._get_obs(), {}

    def step(self, action):
        """执行一个时间步长的仿真"""
        dw_piezo = action[0]  # 提取动作（压电补偿）

        # --- RF源 ---
        S0, self.pha_src = sim_rfsrc(self.fsrc, self.Asrc, self.pha_src, self.Ts)

        # --- 更新缓冲区索引 ---
        if self.pulsed:
            self.buf_id += 1
            if self.buf_id >= self.buf_size:
                self.buf_id = 0

        # --- IQ调制 ---
        S1 = sim_iqmod(
            S0, pulsed=self.pulsed, base_pul=self.base_pul,
            base_cw=self.base_cw, buf_id=self.buf_id
        )

        # --- 放大器 ---
        S2 = sim_amp(S1, self.gain_dB)

        # --- 微扰 ---
        dw_micr = 2.0 * np.pi * np.random.randn() * 10  # 随机微扰

        # --- 腔体仿真 ---
        vc, vr, self.dw, self.state_vc, self.state_m = sim_cav(
            self.wh, self.RL, self.dw, self.dw0 + dw_micr + dw_piezo, S2, self.state_vc, self.Ts,
            beta=self.beta, state_m0=self.state_m, Am=self.Ad, Bm=self.Bd, Cm=self.Cd, Dm=self.Dd,
            pulsed=self.pulsed, beam_pul=self.beam_pul, beam_cw=self.beam_cw, buf_id=self.buf_id
        )

        # --- 收集历史数据 ---
        self.vc_history.append(vc)
        self.dw_history.append(self.dw)
        self.dw_piezo_history.append(dw_piezo)
        if len(self.vc_history) > self.buf_size:
            self.vc_history.pop(0)
            self.dw_history.pop(0)
            self.dw_piezo_history.pop(0)

        # --- 计算奖励 ---
        if len(self.dw_history) == self.buf_size:
            reward = -np.mean(np.abs(self.dw_history))  # buf_size内失谐量平均值的负数
        else:
            reward = 0  # 缓冲区未满时返回0

        # --- 获取观测值 ---
        obs = self._get_obs()

        # --- 判断是否结束 ---
        # 这里假设环境无固定终止条件，可根据需求修改
        terminated = False
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        vc_amp = float(np.abs(self.state_vc))
        vc_phase = float(np.angle(self.state_vc))
        dw = float(self.dw)
        beam_pul_val = self.beam_pul[self.buf_id] if self.pulsed else self.beam_cw
        beam_pul_amp = float(np.abs(beam_pul_val))

        return np.array([
            vc_amp, vc_phase, dw, beam_pul_amp,
            float(self.fsrc), float(self.Asrc), float(self.pha_src)
        ], dtype=np.float64)

    def render(self):
        """渲染buf_size内的信号"""
        if len(self.vc_history) < self.buf_size:
            print("Not enough data to render (buf_size not reached).")
            return

        # --- 提取buf_size内的数据 ---
        vc_array = np.array(self.vc_history[-self.buf_size:])
        dw_array = np.array(self.dw_history[-self.buf_size:])
        dw_piezo_array = np.array(self.dw_piezo_history[-self.buf_size:])

        # --- 绘制图像 ---
        plt.figure(figsize=(10, 8))

        # 腔体幅值
        plt.subplot(3, 1, 1)
        plt.plot(np.abs(vc_array) * 1e-6)
        plt.xlabel('Time (Ts)')
        plt.ylabel('Cavity Voltage (MV)')

        # 腔体相位
        plt.subplot(3, 1, 2)
        plt.plot(np.angle(vc_array) * 180 / np.pi)
        plt.xlabel('Time (Ts)')
        plt.ylabel('Cavity Phase (deg)')

        # 失谐量和压电补偿
        plt.subplot(3, 1, 3)
        plt.plot(dw_array / (2 * np.pi), label='Detuning')
        plt.plot(dw_piezo_array / (2 * np.pi), '--', label='dw_piezo')
        plt.xlabel('Time (Ts)')
        plt.ylabel('Detuning (Hz)')
        plt.legend()

        plt.tight_layout()
        plt.show(block=False)


# --- 测试环境 ---
if __name__ == "__main__":
    env = RFSystemEnv()
    obs, _ = env.reset()
    print("Initial observation:", obs)

    for _ in range(2048 * 16):  # 模拟一些步数
        action = np.random.randn(1) * 2 * np.pi * 500  # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        if _ % 1000 == 0:
            print(f"Step {_}: Reward = {reward}, Obs = {obs}")

    env.render()