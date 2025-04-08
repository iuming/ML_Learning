import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from llrflibs.rf_sim import *
from llrflibs.rf_control import *

class SRFCavityEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super().__init__()
        
        # Define action and observation spaces
        # Action space: [piezo_compensation, rf_drive_amplitude, rf_drive_phase]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -np.pi]),  # Normalized piezo compensation, RF amplitude, RF phase
            high=np.array([1.0, 1.0, np.pi]),
            dtype=np.float32
        )

        # Observation space: [cavity_voltage_real, cavity_voltage_imag, detuning, 
        #                   rf_drive_real, rf_drive_imag, cavity_power]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        # Cavity parameters
        self.f0 = 1.3e9  # Resonance frequency in Hz
        self.QL = 1e7    # Loaded Q-factor
        self.Rs = 1e6    # Shunt impedance in ohms
        self.beta = 1e5  # Coupling factor
        
        # Mechanical mode parameters
        self.fmech = 100  # Mechanical resonance frequency in Hz
        self.Qmech = 1000  # Mechanical Q-factor
        self.Kmech = 1e-3  # Mechanical coupling factor
        
        # Time parameters
        self.dt = 1e-9    # Time step in seconds
        self.pulse_length = 1.0  # Pulse length in seconds
        self.steps_per_pulse = int(self.pulse_length / self.dt)
        
        # Visualization parameters
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        
        # Initialize cavity state
        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Initialize cavity state-space model
        # Create mechanical modes dictionary
        mech_modes = {
            'f': [self.fmech],  # frequencies of mechanical modes in Hz
            'Q': [self.Qmech],  # quality factors
            'K': [self.Kmech]   # K values in rad/s/(MV)^2
        }
        
        status, self.A, self.B, self.C, self.D = cav_ss_mech(
            mech_modes,  # mechanical modes dictionary
            lpf_fc=None  # optional low-pass cutoff frequency
        )
        
        if not status:
            raise ValueError("Failed to initialize cavity state-space model")
        
        # Initialize state variables
        self.x = np.zeros(self.A.shape[0], dtype=np.complex128)
        self.t = 0
        self.current_step = 0
        
        # Initialize history for visualization
        self.history = {
            'time': [],
            'vc_real': [],
            'vc_imag': [],
            'detuning': [],
            'rf_drive': [],
            'cavity_power': []
        }
        
        # Get initial observation
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        # Extract actions
        piezo_comp, rf_amp, rf_phase = action
        
        # Apply piezo compensation to mechanical detuning
        detuning = piezo_comp * self.Kmech
        
        # Create RF drive signal
        rf_drive = rf_amp * np.exp(1j * rf_phase)
        
        # Simulate one step
        self.x = sim_scav_step(
            self.x, self.A, self.B, self.C, self.D,
            detuning, self.dt
        )
        
        # Update time and step counter
        self.t += self.dt
        self.current_step += 1
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward components
        detuning_penalty = -abs(observation[2])  # Penalize detuning
        power_reward = observation[5]  # Reward for maintaining cavity power
        phase_penalty = -abs(np.angle(observation[0] + 1j * observation[1]))  # Penalize phase deviation
        
        # Combined reward
        reward = detuning_penalty + 0.1 * power_reward + 0.5 * phase_penalty
        
        # Check if episode is done
        terminated = self.current_step >= self.steps_per_pulse
        truncated = False
        
        # Update history
        self.history['time'].append(self.t)
        self.history['vc_real'].append(observation[0])
        self.history['vc_imag'].append(observation[1])
        self.history['detuning'].append(observation[2])
        self.history['rf_drive'].append(rf_drive)
        self.history['cavity_power'].append(observation[5])
        
        info = {
            'time': self.t,
            'step': self.current_step,
            'detuning': observation[2],
            'cavity_power': observation[5]
        }
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Get cavity voltage from state
        vc = self.C @ self.x + self.D
        
        # Extract real and imaginary parts
        vc_real = np.real(vc)
        vc_imag = np.imag(vc)
        
        # Calculate detuning from mechanical mode
        detuning = np.imag(self.x[-1])  # Last state is mechanical mode
        
        # Calculate RF drive components (from last action)
        rf_drive = self.history['rf_drive'][-1] if self.history['rf_drive'] else 0
        rf_drive_real = np.real(rf_drive)
        rf_drive_imag = np.imag(rf_drive)
        
        # Calculate cavity power
        cavity_power = np.abs(vc)**2 / (2 * self.Rs)
        
        return np.array([
            vc_real, vc_imag, detuning,
            rf_drive_real, rf_drive_imag,
            cavity_power
        ], dtype=np.float32)

    def render(self):
        if self.render_mode != "human":
            return
            
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('SRF Cavity Control')
            
        # Clear previous plots
        for ax in self.ax.flat:
            ax.clear()
            
        # Plot cavity voltage
        self.ax[0,0].plot(self.history['time'], self.history['vc_real'], label='Real')
        self.ax[0,0].plot(self.history['time'], self.history['vc_imag'], label='Imag')
        self.ax[0,0].set_title('Cavity Voltage')
        self.ax[0,0].legend()
        
        # Plot detuning
        self.ax[0,1].plot(self.history['time'], self.history['detuning'])
        self.ax[0,1].set_title('Detuning')
        
        # Plot RF drive
        rf_drive_amp = [np.abs(d) for d in self.history['rf_drive']]
        self.ax[1,0].plot(self.history['time'], rf_drive_amp)
        self.ax[1,0].set_title('RF Drive Amplitude')
        
        # Plot cavity power
        self.ax[1,1].plot(self.history['time'], self.history['cavity_power'])
        self.ax[1,1].set_title('Cavity Power')
        
        plt.tight_layout()
        plt.pause(0.01)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
