import numpy as np
import time
from utils.shared_memory import SharedMemoryManager
from llrflibs.rf_sim import *
from llrflibs.rf_control import *


class RFSimulator:
    def __init__(self):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.create_shared_memory("srf_data", size=4096)
        self.shm_manager.create_semaphore("srf_sem", initial_value=1)

        # General parameters
        self.Ts = 1e-6  # simulation time step, s

        # RF source parameters
        self.fsrc = -460  # offset frequency from carrier, Hz
        self.Asrc = 1  # RF source amplitude, V
        self.pha_src = 0  # state variable

        # I/Q modulator parameters
        self.pulsed = False  # pulsed or CW mode
        self.buf_size = 2048 * 8  # buffer size
        self.base_pul = np.zeros(self.buf_size, dtype=complex)  # buffer for baseband signal
        self.base_cw = 1  # complex CW baseband scalar input
        self.buf_id = 0  # id to get the buffer data

        # Amplifier parameters
        self.gain_dB = 20 * np.log10(12e6)

        # Cavity parameters
        self.mech_modes = {'f': [280, 341, 460, 487, 618],
                           'Q': [40, 20, 50, 80, 100],
                           'K': [2, 0.8, 2, 0.6, 0.2]}

        # derived parameters
        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        status, Ad, Bd, Cd, Dd, _ = ss_discrete(Am, Bm, Cm, Dm,
                                                Ts=self.Ts,
                                                method='zoh',
                                                plot=False,
                                                plot_pno=10000)

        self.f0 = 1.3e9  # RF operating frequency, Hz
        self.beta = 1e4
        self.roQ = 1036  # r/Q of the cavity, Ohm
        self.QL = 3e6  # loaded quality factor
        self.RL = 0.5 * self.roQ * self.QL  # loaded resistance (Linac convention), Ohm
        self.wh = np.pi * self.f0 / self.QL  # half bandwidth, rad/s
        self.ib = 0.008  # average beam current, A
        self.dw0 = 2 * np.pi * 000  # initial detuning, rad/s
        self.beam_pul = np.zeros(self.buf_size, dtype=complex)  # buffer for pulsed beam
        self.beam_cw = 0  # complex CW beam
        self.state_m = np.matrix(np.zeros(Bd.shape))  # state of the mechanical equation
        self.state_vc = 0.0  # state of cavity equation

        # RF system simulator
        self.sim_len = 2048 * 16
        self.pul_len = 2048 * 10

        self.sig_src = np.zeros(self.sim_len, dtype=complex)
        self.sig_iqm = np.zeros(self.sim_len, dtype=complex)
        self.sig_amp = np.zeros(self.sim_len, dtype=complex)
        self.sig_vc = np.zeros(self.sim_len, dtype=complex)
        self.sig_vr = np.zeros(self.sim_len, dtype=complex)
        self.sig_dw = np.zeros(self.sim_len, dtype=complex)

        self.dw = 0

    # simulate
    def sim_amp(self, sig_in, gain_dB):
        return sig_in * 10.0 ** (gain_dB / 20.0)

    def sim_rfsrc(self, fsrc, Asrc, pha_src, Ts):
        pha = pha_src + 2.0 * np.pi * fsrc * Ts
        return Asrc * np.exp(1j * pha), pha

    def sim_iqmod(self, sig_in, pulsed=True, base_pul=None, base_cw=0, buf_id=0):
        if pulsed:
            sig_out = sig_in * base_pul[buf_id if buf_id < len(base_pul) else -1]
        else:
            sig_out = sig_in * base_cw
        return sig_out

    def sim_cav(self, half_bw, RL, dw_step0, detuning0, vf_step, state_vc, Ts, beta=1e4,
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

    def calculate_next_state(self):
        # RF signal source
        S0, self.pha_src = self.sim_rfsrc(self.fsrc, self.Asrc, self.pha_src, self.Ts)

        # emulate the pulse
        if self.pulsed:
            self.buf_id += 1
            if self.buf_id >= self.pul_len:
                self.buf_id = 0

        # I/Q modulator
        S1 = self.sim_iqmod(S0, pulsed=self.pulsed, base_pul=self.base_pul, base_cw=self.base_cw,
                            buf_id=self.buf_id)

        # Amplifier
        S2 = self.sim_amp(S1, self.gain_dB)

        # Microphonics
        dw_micr = 2.0 * np.pi * np.random.randn() * 10

        # Cavity
        vc, vr, dw, self.state_vc, self.state_m = self.sim_cav(self.wh, self.RL, self.dw,
                                                              self.dw0 + dw_micr, S2, self.state_vc, self.Ts,
                                                              beta=self.beta, state_m0=self.state_m, Am=Ad, Bm=Bd,
                                                              Cm=Cd, Dm=Dd,
                                                              pulsed=self.pulsed, beam_pul=self.beam_pul,
                                                              beam_cw=self.beam_cw, buf_id=self.buf_id)

        # Collect the results
        self.sig_src = np.roll(self.sig_src, -1)
        self.sig_iqm = np.roll(self.sig_iqm, -1)
        self.sig_amp = np.roll(self.sig_amp, -1)
        self.sig_vc = np.roll(self.sig_vc, -1)
        self.sig_vr = np.roll(self.sig_vr, -1)
        self.sig_dw = np.roll(self.sig_dw, -1)
        self.sig_src[-1] = S0
        self.sig_iqm[-1] = S1
        self.sig_amp[-1] = S2
        self.sig_vc[-1] = vc
        self.sig_vr[-1] = vr
        self.sig_dw[-1] = dw

        return vc, vr, dw

    def run_simulation(self):
        while True:
            # 生成仿真数据
            vc, vr, dw = self.calculate_next_state()

            # 写入共享内存
            with self.shm_manager.semaphore:
                self.shm_manager.write_data(np.array([vc, vr, dw], dtype=np.complex128))

            # 控制周期
            time.sleep(self.Ts)
