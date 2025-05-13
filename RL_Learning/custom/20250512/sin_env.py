import gymnasium as gym
import numpy as np
from collections import deque
from typing import Optional
from llrflibs.rf_sim import *
from llrflibs.rf_control import *

class SinEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        # simulation parameters
        self.max_steps = 100_000
        self.dt = 2 * np.pi / (self.max_steps - 1)
        self.Ts = 1e-6
        self.t_fill = 510
        self.t_flat = 1300

        # action & observation spaces
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -360.0, -10.0]),
            high=np.array([20.0, 20.0, 360.0, 10.0]),
            dtype=np.float32
        )

        # history for rendering
        self.render_mode = render_mode
        self.history_t = deque(maxlen=1000)
        self.history_ft = deque(maxlen=1000)

        # RF source parameters
        self.fsrc = -460
        self.Asrc = 1.0
        self.pha_src = 0.0

        # pulsed/CW config
        self.pulsed = True
        self.buf_size = 2048 * 8
        self.base_pul = np.zeros(self.buf_size, dtype=np.float32)
        self.base_cw = 1.0
        self.base_pul[:self.t_flat] = 1.0
        self.buf_id = 0
        self.pul_len = 2048 * 20

        # amplifier gain
        self.gain_dB = 20 * np.log10(12e6)

        # cavity/mechanical parameters
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
        self.dw = 0.0
        self.prev_dw = 0.0

        # beam
        self.beam_pul = np.zeros(self.buf_size, dtype=np.complex64)
        self.beam_pul[self.t_fill:self.t_flat] = self.ib
        self.beam_cw = 0.0

        # precompute state-space
        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        status, Ad, Bd, Cd, Dd, _ = ss_discrete(
            Am, Bm, Cm, Dm,
            Ts=self.Ts,
            method='zoh',
            plot=False,
            plot_pno=10000
        )
        self.Ad = Ad.astype(np.float32)
        self.Bd = Bd.astype(np.float32)
        self.Cd = Cd.astype(np.float32)
        self.Dd = Dd.astype(np.float32)
        self.state_m = np.zeros((Bd.shape[0], 1), dtype=np.float32)
        self.state_vc = np.complex64(0.0)

        # reset counters
        self.sim_len = 2048 * 500
        self.step_count = 0
        self.t = 0.0

    def sim_rfsrc(self):
        # RF source: returns complex amplitude and phase
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        return np.complex64(self.Asrc * np.exp(1j * pha)), pha

    def sim_iqmod(self, sig_in: np.complex64) -> np.complex64:
        if self.pulsed:
            gate = self.base_pul[self.buf_id] if self.buf_id < self.buf_size else 0.0
            return sig_in * gate
        return sig_in * self.base_cw

    def sim_amp(self, sig_in: np.complex64) -> np.complex64:
        return sig_in * np.complex64(10**(self.gain_dB / 20.0))

    # def sim_cav(self, vf_step: np.complex64, detuning: float):
    #     vb = -self.RL * (self.beam_pul[self.buf_id] if self.pulsed else self.beam_cw)
    #     status, vc, vr, dw_new, state_m = sim_scav_step(
    #         self.wh,
    #         self.dw,
    #         detuning,
    #         vf_step,
    #         vb,
    #         self.state_vc,
    #         self.Ts,
    #         beta=self.beta,
    #         state_m0=self.state_m,
    #         Am=self.Ad,
    #         Bm=self.Bd,
    #         Cm=self.Cd,
    #         Dm=self.Dd,
    #         mech_exe=True
    #     )
    #     self.state_m = state_m.astype(np.float32)
    #     self.state_vc = np.complex64(vc)
    #     self.dw = float(dw_new)
    #     # dw_new may be a length‐1 array/matrix; extract its scalar value
    #     dw_val = np.asarray(dw_new).flatten()[0]
    #     self.dw = float(dw_val)
    #     return self.state_vc, vr, self.dw
    
    def sim_cav(self, vf_step: np.complex64, detuning: float):
        vb = -self.RL * (self.beam_pul[self.buf_id] if self.pulsed else self.beam_cw)
        status, vc, vr, dw_new, state_m = sim_scav_step(
            self.wh,
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
            mech_exe=True
        )
        # update mechanial state
        self.state_m = state_m.astype(np.float32)
        self.state_vc = np.complex64(vc)
        # Ensure dw_new is a scalar
        
        dw_arr = np.asarray(dw_new)
        self.dw = float(dw_arr.flatten()[0])
        return self.state_vc, vr, self.dw


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.t = 0.0
        self.buf_id = 0
        self.pha_src = 0.0
        self.dw = 0.0
        self.prev_dw = 0.0
        self.state_m.fill(0.0)
        self.state_vc = np.complex64(0.0)
        self.history_t.clear()
        self.history_ft.clear()

        # initial RF chain
        S0, self.pha_src = self.sim_rfsrc()
        S1 = self.sim_iqmod(S0)
        S2 = self.sim_amp(S1)
        vc, vr, self.dw = self.sim_cav(S2, 0.0)

        obs = np.array([
            float(np.abs(vc) * 1e-6),
            float(np.abs(vr) * 1e-6),
            float(np.angle(vc) * 180 / np.pi),
            float(self.dw * 1e-3)
        ], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        # clip action
        u = np.clip(action, self.action_space.low, self.action_space.high)[0]

        # update RF source
        S0, self.pha_src = self.sim_rfsrc()
        # if self.pulsed:
        #     self.buf_id = (self.buf_id + 1) % self.pul_len
        if self.pulsed:
            # wrap around the actual buffer length, not the pulse-length
            self.buf_id = (self.buf_id + 1) % self.buf_size
        S1 = self.sim_iqmod(S0)
        S2 = self.sim_amp(S1)

        # external detuning from action
        dw_piezo = 2 * np.pi * u * 1e4
        vc, vr, _ = self.sim_cav(S2, dw_piezo)

        # build observation
        obs = np.array([
            float(np.abs(vc) * 1e-6),
            float(np.abs(vr) * 1e-6),
            float(np.angle(vc) * 180 / np.pi),
            float(self.dw * 1e-3)
        ], dtype=np.float32)
        # guard against NaN/inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

        # reward: squared freq error
        err = self.dw * 1e-3
        reward = -err**2

        # record for render
        self.history_t.append(self.t)
        self.history_ft.append(obs[0])

        # update time and prev_dw
        self.t += self.dt
        self.prev_dw = self.dw

        done = self.step_count >= self.max_steps
        info = {}
        return obs, reward, False, done, info

    def render(self):
        if self.render_mode is None:
            return
        import matplotlib.pyplot as plt
        if not hasattr(self, 'ax'):
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.line, = self.ax.plot([], [], 'b-')
            self.ax.set_xlim(0, 2*np.pi)
            self.ax.set_ylim(-2, 2)
            self.ax.set_xlabel('t')
            self.ax.set_ylabel('f(t)')
            plt.title('Sin Wave with Actions')
        self.line.set_data(self.history_t, self.history_ft)
        self.ax.relim(); self.ax.autoscale_view()
        if self.render_mode=='human':
            plt.draw(); plt.pause(0.001)
        else:
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            return img.reshape(self.fig.canvas.get_width_height()[::-1]+(3,))