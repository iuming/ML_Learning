import gymnasium as gym
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from llrflibs.rf_sim import *
from llrflibs.rf_control import *


class SinEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.max_steps = 2048 * 16
        self.dt = 2 * np.pi / (self.max_steps - 1)
        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1000.0, -1000.0, -360.0, -1000.0]),
            high=np.array([1000.0, 1000.0, 360.0, 1000.0]),
            shape=(4,),
            dtype=np.float32
        )

        self.action_history = [np.zeros(1) for _ in range(3)]

        # 渲染设置
        self.render_mode = render_mode
        self.fig = None
        self.line = None
        self.history_t = []
        self.history_ft = []

        # 通用参数
        self.Ts = 1e-6
        self.t_fill = 510
        self.t_flat = 1300

        # RF 源参数
        self.fsrc = -460
        self.Asrc = 1
        self.pha_src = 0

        # I/Q 调制器参数
        self.pulsed = True
        self.buf_size = 2048 * 8
        self.base_pul = np.zeros(self.buf_size, dtype=complex)
        self.base_cw = 1
        self.base_pul[:self.t_flat] = 1.0
        self.buf_id = 0

        # 放大器参数
        self.gain_dB = 20 * np.log10(12e6)

        # 腔参数
        self.mech_modes = {'f': [280, 341, 460, 487, 618],
                           'Q': [40, 20, 50, 80, 100],
                           'K': [2, 0.8, 2, 0.6, 0.2]}
        self.f0 = 1.3e9
        self.beta = 1e4
        self.roQ = 1036
        self.QL = 3e6
        self.RL = 0.5 * self.roQ * self.QL
        self.wh = np.pi * self.f0 / self.QL
        self.ib = 0.008
        self.dw0 = 2 * np.pi * 0

        self.beam_pul = np.zeros(self.buf_size, dtype=complex)
        self.beam_cw = 0
        self.beam_pul[self.t_fill:self.t_flat] = self.ib

        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        status, Ad, Bd, Cd, Dd, _ = ss_discrete(Am, Bm, Cm, Dm,
                                                Ts=self.Ts,
                                                method='zoh',
                                                plot=False,
                                                plot_pno=10000)
        self.state_m = np.matrix(np.zeros(Bd.shape))
        self.state_vc = 0.0
        self.Ad = Ad
        self.Bd = Bd
        self.Cd = Cd
        self.Dd = Dd

        self.sim_len = 2048 * 500
        self.pul_len = 2048 * 20
        self.dw = 0
        self.prev_dw = 0

    def sim_rfsrc(self):
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        return self.Asrc * np.exp(1j * pha), pha

    def sim_iqmod(self, sig_in):
        if self.pulsed:
            sig_out = sig_in * self.base_pul[self.buf_id if self.buf_id < len(self.base_pul) else -1]
        else:
            sig_out = sig_in * self.base_cw
        return sig_out

    def sim_amp(self, sig_in):
        return sig_in * 10.0 ** (self.gain_dB / 20.0)

    def sim_cav(self, vf_step, detuning):
        if self.pulsed:
            vb = -self.RL * self.beam_pul[self.buf_id if self.buf_id < len(self.beam_pul) else -1]
        else:
            vb = self.beam_cw

        status, vc, vr, dw, state_m = sim_scav_step(self.wh,
                                                    self.dw,
                                                    detuning,
                                                    vf_step,
                                                    vb,
                                                    self.state_vc,
                                                    self.Ts,
                                                    beta=self.beta,
                                                    state_m0=self.state_m,
                                                    Am=self.Ad,
                                                    Bm=self.Bd,
                                                    Cm=self.Cd,
                                                    Dm=self.Dd,
                                                    mech_exe=True)
        self.state_vc = vc
        return vc, vr, dw, self.state_vc, state_m

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.step_count = 0
        self.history_t = []
        self.history_ft = []
        
        self.pha_src = 0
        self.buf_id = 0
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.dw = 0
        self.prev_dw = 0  # 初始化prev_dw

        S0, self.pha_src = self.sim_rfsrc()
        S1 = self.sim_iqmod(S0)
        S2 = self.sim_amp(S1)
        dw_micr = 2.0 * np.pi * np.random.randn() * 10
        dw_piezo = 0
        vc, vr, self.dw, self.state_vc, self.state_m = self.sim_cav(S2, dw_piezo + dw_micr)

        if isinstance(vc, np.matrix):
            vc = vc.item()

        # 构建初始观测，使用实际的仿真结果
        vc_abs = abs(vc) if vc is not None else 0.0
        vr_abs = abs(vr) if vr is not None else 0.0
        vc_angle = np.angle(vc) if vc is not None else 0.0
        dw_val = self.dw if self.dw is not None else 0.0
        
        # 处理矩阵类型
        if isinstance(vc_abs, np.matrix):
            vc_abs = vc_abs.item()
        if isinstance(vr_abs, np.matrix):
            vr_abs = vr_abs.item()
        if isinstance(vc_angle, np.matrix):
            vc_angle = vc_angle.item()
        if isinstance(dw_val, np.matrix):
            dw_val = dw_val.item()
        
        # 确保所有值都是有效的数值
        vc_abs = np.nan_to_num(vc_abs, nan=0.0, posinf=1e3, neginf=-1e3)
        vr_abs = np.nan_to_num(vr_abs, nan=0.0, posinf=1e3, neginf=-1e3)
        vc_angle = np.nan_to_num(vc_angle, nan=0.0, posinf=np.pi, neginf=-np.pi)
        dw_val = np.nan_to_num(dw_val, nan=0.0, posinf=1e9, neginf=-1e9)
        
        # 构建观测值并限制范围
        obs_vc = np.clip(float(vc_abs * 1e-6), -1000.0, 1000.0)
        obs_vr = np.clip(float(vr_abs * 1e-6), -1000.0, 1000.0)
        obs_angle = np.clip(float(vc_angle * 180 / np.pi), -360.0, 360.0)
        obs_dw = np.clip(float(dw_val * 1e-3), -1000.0, 1000.0)
        
        observation = np.array([obs_vc, obs_vr, obs_angle, obs_dw], dtype=np.float32)
        action_history_array = np.concatenate(self.action_history)
        # action_history_array = action_history_array.flatten()
        action_history_array = np.ravel(action_history_array)

        return observation, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        current_sin = np.sin(50.0 * self.t)

        self.S0, self.pha_src = self.sim_rfsrc()


        if self.pulsed:
            self.buf_id += 1
            if self.buf_id >= self.pul_len:
                self.buf_id = 0

        self.S1 = self.sim_iqmod(self.S0)
        
        self.S2 = self.sim_amp(self.S1)
        
        self.dw_micr = 2.0 * np.pi * np.random.randn() * 0
        dw_piezo = 2 * np.pi * action[0] * 1e4
        
        self.vc, self.vr, self.dw, self.state_vc, self.state_m = self.sim_cav(self.S2, dw_piezo + self.dw_micr)
        
        

        self.action_history.pop(0)
        self.action_history.append(action)
        
        # 更新prev_dw用于下一步
        self.prev_dw = self.dw

        # 构建观测，添加NaN检查
        vc_abs = abs(self.vc) if self.vc is not None else 0.0
        vr_abs = abs(self.vr) if self.vr is not None else 0.0
        vc_angle = np.angle(self.vc) if self.vc is not None else 0.0
        dw_val = self.dw if self.dw is not None else 0.0
        
        # 处理矩阵类型
        if isinstance(vc_abs, np.matrix):
            vc_abs = vc_abs.item()
        if isinstance(vr_abs, np.matrix):
            vr_abs = vr_abs.item()
        if isinstance(vc_angle, np.matrix):
            vc_angle = vc_angle.item()
        if isinstance(dw_val, np.matrix):
            dw_val = dw_val.item()
        
        # 确保所有值都是有效的数值
        vc_abs = np.nan_to_num(vc_abs, nan=0.0, posinf=1e3, neginf=-1e3)
        vr_abs = np.nan_to_num(vr_abs, nan=0.0, posinf=1e3, neginf=-1e3)
        vc_angle = np.nan_to_num(vc_angle, nan=0.0, posinf=np.pi, neginf=-np.pi)
        dw_val = np.nan_to_num(dw_val, nan=0.0, posinf=1e9, neginf=-1e9)
        
        # 构建观测值并限制范围
        obs_vc = np.clip(float(vc_abs * 1e-6), -1000.0, 1000.0)
        obs_vr = np.clip(float(vr_abs * 1e-6), -1000.0, 1000.0)
        obs_angle = np.clip(float(vc_angle * 180 / np.pi), -360.0, 360.0)
        obs_dw = np.clip(float(dw_val * 1e-3), -1000.0, 1000.0)
        
        obs = np.array([obs_vc, obs_vr, obs_angle, obs_dw], dtype=np.float32)


        # obs = np.array([
        #     current_sin + action[0],
        #     current_sin,
        #     np.cos(self.t),
        #     np.sin(self.t)  # dw
        # ], dtype=np.float32)

        # 记录数据
        self.history_t.append(self.t)
        self.history_ft.append(obs[0].item())

        # 计算奖励，添加安全检查
        reward = -np.abs(obs[3]) # 这里的奖励是基于频率偏移量的绝对值
        reward = np.nan_to_num(reward, nan=-1.0, posinf=-1.0, neginf=-1.0)
        reward = np.clip(reward, -1000.0, 1000.0)  # 限制奖励范围

        # 更新状态
        self.t += self.dt
        self.step_count += 1

        terminated = False
        truncated = self.step_count >= self.max_steps
        # truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        # 创建或更新图像
        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 5))
            self.ax = self.fig.add_subplot(111)
            self.line, = self.ax.plot([], [], 'b-')
            self.ax.set_xlim(0, 2 * np.pi)
            self.ax.set_ylim(-2, 2)
            self.ax.set_xlabel('t')
            self.ax.set_ylabel('f(t)')
            plt.title('Sin Wave with Actions')

        # 更新数据
        self.line.set_data(self.history_t, self.history_ft)
        self.ax.relim()
        self.ax.autoscale_view()

        if self.render_mode == 'human':
            plt.draw()
            plt.pause(0.001)
        elif self.render_mode == 'rgb_array':
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None