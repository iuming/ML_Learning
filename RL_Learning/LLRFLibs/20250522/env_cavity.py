"""
Name: env_cavity.py
version: 1.0

author: Ming Liu from IHEP
date: 2025-05-22

description:
This module defines a Gymnasium-compatible environment for simulating RF cavity systems,
intended for reinforcement learning research and development.

Features:
    - Simulates RF source, I/Q modulation, amplification, and cavity dynamics with microphonics.
    - Supports both pulsed and continuous wave (CW) operation modes.
    - Provides step, reset, render, and state management methods for RL workflows.

Dependencies:
    - gymnasium: RL environment interface.
    - numpy: Numerical operations.
    - matplotlib: Visualization.
    - llrflibs.rf_sim, llrflibs.rf_control: RF system simulation and control utilities.
"""

import gymnasium as gym
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from llrflibs.rf_sim import *
from llrflibs.rf_control import *

def sim_iqmod(sig_in, pulsed = True, base_pul = None, base_cw = 0, buf_id = 0):
            if pulsed:
                sig_out = sig_in * base_pul[buf_id % len(base_pul)]
            else:
                sig_out = sig_in * base_cw
            return sig_out

def sim_amp(sig_in, gain_dB):
            return sig_in * 10.0**(gain_dB / 20.0) 

def sim_rfsrc(fsrc, Asrc, pha_src, Ts):
            pha = pha_src + 2.0 * np.pi * fsrc * Ts
            return Asrc*np.exp(1j*pha), pha

def sim_cav(half_bw, RL, dw_step0, detuning0, vf_step, state_vc, Ts, beta = 1e4,
                    state_m0 = 0, Am = None, Bm = None, Cm = None, Dm = None,
                    pulsed = True, beam_pul = None, beam_cw = 0, buf_id = 0):
            # get the beam
            if pulsed:
                vb = -RL * beam_pul[buf_id % len(beam_pul)]
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
                                                        beta      = beta,
                                                        state_m0  = state_m0, 
                                                        Am        = Am, 
                                                        Bm        = Bm, 
                                                        Cm        = Cm, 
                                                        Dm        = Dm,
                                                        mech_exe  = True)           
            state_vc = vc
            
            # return 
            return vc, vr, dw, state_vc, state_m


class Cavity_env(gym.Env):
    """
    Cavity environment for reinforcement learning simulations.
    This environment simulates the behavior of an RF cavity system.
    """

    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 30
    }

    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the Cavity environment.

        Parameters:
            render_mode (Optional[str]): The mode for rendering the environment.
        """
        super().__init__()

        self.action_space = gym.spaces.Box(
            low=-2.0, high=2.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([-20.0, -20.0, -360.0, -10.0]),
            high=np.array([20.0, 20.0, 360.0, 10.0]),
            shape=(4,),
            dtype=np.float32
        )


        # ---------------------------------------
        # General parameters
        # ---------------------------------------
        # parameters
        self.Ts = 1e-6                   # simulation time step, s

        # only for simulation here
        self.t_fill  = 510               # length of cavity filling stage, sample
        self.t_flat  = 1300              # end time of the flattop stage, sample

        # ---------------------------------------
        # RF source
        # ---------------------------------------
        # parameters
        self.fsrc = -460                # offset frequency from carrier, Hz
        self.Asrc = 1                    # RF source amplitude, V

        # simulate per time step
        self.pha_src = 0                 # state variable
        

        # ---------------------------------------
        # I/Q modulator
        # ---------------------------------------
        # parameters
        self.pulsed      = False                                     # pulsed or CW mode
        self.buf_size    = 2048 * 8                                  # buffer size
        self.base_pul    = np.zeros(self.buf_size, dtype = complex)  # buffer for baseband signal
        self.base_cw     = 1                                         # complex CW baseband scalar input  

        self.base_pul[:self.t_flat] = 1.0

        # simulate per step
        self.buf_id = 0                                              # id to get the buffer data
        

        # ---------------------------------------
        # amplifier
        # ---------------------------------------
        # parameters
        self.gain_dB = 20*np.log10(12e6)

        # simulate
        

        # ---------------------------------------
        # cavity
        # ---------------------------------------
        # parameters
        self.mech_modes = {'f': [280, 341, 460, 487, 618],
                    'Q': [40, 20, 50, 80, 100],
                    'K': [2, 0.8, 2, 0.6, 0.2]}

        self.f0   = 1.3e9                                 # RF operating frequency, Hz
        self.beta = 1e4
        self.roQ  = 1036                                  # r/Q of the cavity, Ohm
        self.QL   = 3e6                                   # loaded quality factor
        self.RL   = 0.5 * self.roQ * self.QL              # loaded resistance (Linac convention), Ohm
        self.wh   = np.pi * self.f0 / self.QL             # half bandwidth, rad/s
        self.ib   = 0.008                                 # average beam current, A
        self.dw0  = 2*np.pi*000                           # initial detuning, rad/s

        self.beam_pul = np.zeros(self.buf_size, dtype = complex)  # buffer for pulsed beam 
        self.beam_cw  = 0                                    # complex CW beam
        self.beam_pul[self.t_fill:self.t_flat] = self.ib

        # derived parameters
        status, self.Am, self.Bm, self.Cm, self.Dm = cav_ss_mech(self.mech_modes)
        status, self.Ad, self.Bd, self.Cd, self.Dd, _ = ss_discrete(self.Am, self.Bm, self.Cm, self.Dm, 
                                            Ts       = self.Ts, 
                                            method   = 'zoh', 
                                            plot     = False,
                                            plot_pno = 10000)

        # simulation
        self.state_m  = np.matrix(np.zeros(self.Bd.shape))        # state of the mechanical equation
        self.state_vc = 0.0                                       # state of cavity equation

        # ---------------------------------------
        # RF system simulator
        # ---------------------------------------
        self.sim_len = 2048 * 16
        self.pul_len = 2048 * 10

        self.sig_src = np.zeros(self.sim_len, dtype = complex)
        self.sig_iqm = np.zeros(self.sim_len, dtype = complex)
        self.sig_amp = np.zeros(self.sim_len, dtype = complex)
        self.sig_vc  = np.zeros(self.sim_len, dtype = complex)
        self.sig_vr  = np.zeros(self.sim_len, dtype = complex)
        self.sig_dw  = np.zeros(self.sim_len, dtype = complex)

        self.dw = 0
        self.prev_dw = 0

    def reset(self, seed = None, options = None):
        """
        Reset the environment to an initial state.

        Parameters:
            seed (Optional[int]): Random seed for reproducibility.
            options (Optional[dict]): Additional options for resetting the environment.

        Returns:
            tuple: Initial observation and info dictionary.
        """
        self.buf_id = 0
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.state_vc = 0.0
        self.dw = 0
        self.prev_dw = 0

        return np.zeros(4), {}
    
    def step(self, action):
        # Unpack environment variables for readability
        Ts = self.Ts
        fsrc = self.fsrc
        Asrc = self.Asrc
        pha_src = self.pha_src
        pulsed = self.pulsed
        base_pul = self.base_pul
        base_cw = self.base_cw
        buf_id = self.buf_id
        gain_dB = self.gain_dB
        wh = self.wh
        RL = self.RL
        dw = self.dw
        dw0 = self.dw0
        beta = self.beta
        state_m = self.state_m
        state_vc = self.state_vc
        Ad = self.Ad
        Bd = self.Bd
        Cd = self.Cd
        Dd = self.Dd
        beam_pul = self.beam_pul
        beam_cw = self.beam_cw
        pul_len = self.pul_len

        # Apply action (assume action is detuning adjustment)
        detuning_action = float(action[0])
        dw += detuning_action

        # RF signal source
        S0, pha_src = sim_rfsrc(fsrc, Asrc, pha_src, Ts)

        # Emulate the pulse
        buf_id += 1
        if buf_id >= pul_len:
            buf_id = 0

        # I/Q modulator
        S1 = sim_iqmod(
            S0,
            pulsed=pulsed,
            base_pul=base_pul,
            base_cw=base_cw,
            buf_id=buf_id
        )

        # Amplifier
        S2 = sim_amp(S1, gain_dB)

        # Microphonics noise
        dw_micr = 2.0 * np.pi * np.random.randn() * 10

        # Cavity simulation
        vc, vr, dw_new, state_vc, state_m = sim_cav(
            wh, RL, dw, dw0 + dw_micr, S2, state_vc, Ts,
            beta=beta,
            state_m0=state_m,
            Am=Ad,
            Bm=Bd,
            Cm=Cd,
            Dm=Dd,
            pulsed=pulsed,
            beam_pul=beam_pul,
            beam_cw=beam_cw,
            buf_id=buf_id
        )

        # Update environment state
        self.pha_src = pha_src
        self.buf_id = buf_id
        self.state_m = state_m
        self.state_vc = state_vc
        self.dw = dw_new
        self.prev_dw = dw

        # Observation: [Re(vc), Im(vc), dw, Re(vr)]
        # Ensure obs is a 1D float32 numpy array
        obs = np.array([
            float(np.real(vc)),
            float(np.imag(vc)),
            float(dw_new),
            float(np.real(vr))
        ], dtype=np.float32)

        # Reward: example, keep dw close to zero
        reward = -np.abs(dw_new)

        # Done: example, after a fixed number of steps
        terminated = False
        truncated = False

        info = {}

        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment.

        Parameters:
            mode (str): The mode for rendering the environment.
        """
        if mode == 'human':
            plt.plot(self.sig_src, label='Source Signal')
            plt.plot(self.sig_iqm, label='IQ Modulated Signal')
            plt.plot(self.sig_amp, label='Amplified Signal')
            plt.plot(self.sig_vc, label='Cavity Voltage')
            plt.plot(self.sig_vr, label='Cavity Resonance')
            plt.legend()
            plt.show()
        elif mode == 'rgb_array':
            # Create an RGB array for rendering
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img.fill(255)
            return img
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """
        Close the environment.
        """
        pass

    def seed(self, seed=None):
        """
        Set the random seed for the environment.

        Parameters:
            seed (Optional[int]): Random seed for reproducibility.
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def get_state(self):
        """
        Get the current state of the environment.

        Returns:
            np.ndarray: Current state of the environment.
        """
        return np.array([self.state_vc, self.state_m, self.dw], dtype=np.float32)
    
    def set_state(self, state):
        """
        Set the state of the environment.

        Parameters:
            state (np.ndarray): New state for the environment.
        """
        self.state_vc = state[0]
        self.state_m = state[1]
        self.dw = state[2]
