import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from llrflibs.rf_sim import *
from llrflibs.rf_control import *
import torch

# 优化GPU设置
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def sim_cav(half_bw, RL, dw_step0, detuning0, vf_step, state_vc, Ts, beta=1e4,
            state_m0=None, Am=None, Bm=None, Cm=None, Dm=None,
            pulsed=True, beam_pul=None, beam_cw=0, buf_id=0):
    """优化后的腔体仿真函数"""
    vb = -RL * beam_pul[buf_id] if pulsed else beam_cw
    status, vc, vr, dw, state_m = sim_scav_step(
        half_bw, dw_step0, detuning0, vf_step, vb, state_vc, Ts,
        beta=beta, state_m0=state_m0, Am=Am, Bm=Bm, Cm=Cm, Dm=Dm, mech_exe=True
    )
    return vc, vr, dw, vc, state_m  # 直接返回更新后的state_vc


class SuperconductingCavityEnv(gym.Env):
    def __init__(self, pulse_cycles=25, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # 环境参数
        self.Ts = 1e-6  # 仿真时间步长
        self.pulse_cycles = pulse_cycles  # 脉冲周期数
        self.buf_size = 2048 * 8  # 移动到类属性

        # 初始化顺序优化
        self._init_mechanical_modes()  # 机械模式初始化
        self._init_cavity_parameters()  # 腔体参数
        self._init_simulation_parameters()  # 仿真参数

        # 动作/观测空间
        self.action_space = spaces.Box(
            low=-2 * np.pi * 100,  # -100 Hz
            high=2 * np.pi * 100,  # +100 Hz
            shape=(1,),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -500]),  # 单位: MV, rad, Hz
            high=np.array([20e6, np.pi, 500]),
            dtype=np.float64
        )

        # 状态初始化
        self.reset()

    def _init_mechanical_modes(self):
        """机械模式初始化（仅执行一次）"""
        self.mech_modes = {
            'f': [280, 341, 460, 487, 618],
            'Q': [40, 20, 50, 80, 100],
            'K': [2, 0.8, 2, 0.6, 0.2]
        }

        # 生成状态空间模型
        status, Am, Bm, Cm, Dm = cav_ss_mech(self.mech_modes)
        if not status:
            raise RuntimeError("机械模式状态空间模型创建失败")

        # 离散化并保存所有参数
        status, self.Ad, self.Bd, self.Cd, self.Dd, _ = ss_discrete(
            Am, Bm, Cm, Dm, Ts=self.Ts, method='zoh'
        )
        self.Am, self.Bm, self.Cm, self.Dm = Am, Bm, Cm, Dm

    def _init_cavity_parameters(self):
        """腔体参数"""
        self.f0 = 1.3e9  # Hz
        self.beta = 1e4
        self.roQ = 1036  # Ohm
        self.QL = 3e6
        self.RL = 0.5 * self.roQ * self.QL
        self.wh = np.pi * self.f0 / self.QL
        self.dw0 = 0.0

    def _init_simulation_parameters(self):
        """仿真参数"""
        # 脉冲参数
        self.t_fill, self.t_flat = 510, 1300
        self.base_pul = np.zeros(self.buf_size, dtype=complex)
        self.base_pul[:self.t_flat] = 1.0
        self.beam_pul = np.zeros(self.buf_size, dtype=complex)
        self.beam_pul[self.t_fill:self.t_flat] = 0.008

        # RF信号参数
        self.fsrc = -460  # Hz
        self.Asrc = 1.0  # V
        self.gain_dB = 20 * np.log10(12e6)

        # 初始状态
        self.pha_src = 0.0
        self.state_vc = 0.0j
        self.state_m = np.zeros(self.Bd.shape, dtype=np.float64)
        self.dw_current = 0.0

        # 数据收集
        self.history = {'vc_amp': [], 'vc_phase': [], 'detuning': [], 'actions': []}

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.buf_id = 0
        self.current_cycle = 0
        self.state_vc = 0.0j
        self.state_m = np.zeros_like(self.state_m)
        self.pha_src = 0.0
        self.dw_current = 0.0
        self.history = {k: [] for k in self.history}
        return self._get_obs(), {}

    def step(self, action):
        """优化后的step函数：每个step对应单个时间步"""
        dw_piezo = action[0]
        self._simulation_step(dw_piezo)
        self._record_data()

        # 计算奖励
        reward = -abs(self.dw_current / (2 * np.pi))

        # 更新周期计数器
        self.current_cycle = self.buf_id // self.buf_size
        terminated = self.current_cycle >= self.pulse_cycles
        truncated = False

        return self._get_obs().flatten(), reward, terminated, truncated, {}

    def _simulation_step(self, dw_piezo):
        """执行单个时间步仿真"""
        # 微扰生成
        dw_micr = 2.0 * np.pi * self.np_random.normal(0, 10)

        # RF信号链
        S0, self.pha_src = self._sim_rfsrc()
        S1 = self._sim_iqmod(S0)
        S2 = self._sim_amp(S1)

        # 腔体仿真
        self.vc, _, self.dw_current, self.state_vc, self.state_m = sim_cav(
            self.wh, self.RL, self.dw_current, self.dw0 + dw_micr + dw_piezo,
            S2, self.state_vc, self.Ts, beta=self.beta, state_m0=self.state_m,
            Am=self.Ad, Bm=self.Bd, Cm=self.Cd, Dm=self.Dd,
            pulsed=True, beam_pul=self.beam_pul, buf_id=self.buf_id
        )

        self.buf_id = (self.buf_id + 1) % self.buf_size

    def _get_obs(self):
        """获取观测值"""
        vc_amp = abs(self.state_vc)
        vc_phase = np.angle(self.state_vc)
        dw_hz = self.dw_current / (2 * np.pi)
        return np.array([vc_amp, vc_phase, dw_hz], dtype=np.float64)

    def _record_data(self):
        """优化数据记录（移除print）"""
        self.history['vc_amp'].append(abs(self.state_vc) * 1e-6)
        self.history['vc_phase'].append(np.angle(self.state_vc) * 180 / np.pi)
        self.history['detuning'].append(self.dw_current / (2 * np.pi))
        self.history['actions'].append(self.dw_current / (2 * np.pi))

    def render(self, mode='human'):
        """优化渲染性能"""
        if not hasattr(self, 'fig'):
            self.fig, self.axs = plt.subplots(4, 1, figsize=(10, 8))
            plt.ion()
            plt.show()

        titles = ['Cavity Voltage (MV)', 'Phase (deg)', 'Detuning (Hz)', 'Piezo Adjustment (Hz)']
        for i, ax in enumerate(self.axs):
            ax.clear()
            ax.set_title(titles[i])
            ax.grid(True)
            data = list(self.history.values())[i][-1000:] if i < 3 else self.history['actions'][-1000:]
            ax.plot(data)
        plt.pause(0.001)

    def _sim_rfsrc(self):
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        return self.Asrc * np.exp(1j * pha), pha

    def _sim_iqmod(self, S0):
        return S0 * self.base_pul[self.buf_id]

    def _sim_amp(self, S1):
        return S1 * 10 ** (self.gain_dB / 20.0)

class TrainingCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, env: gym.Env, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_reward = -np.inf
        self.env = env

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 安全获取子环境数据
            histories = self.env.get_attr('history')
            avg_reward = np.mean(histories[0]['detuning'][-1000:])

            self.logger.record("train/avg_detuning", avg_reward)
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")

            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.model.save(f"{self.save_path}/best_model")

            # 记录训练曲线
            if len(histories[0]['vc_amp']) > 100:
                fig = self._create_metrics_figure(histories[0])
                self.logger.record("trajectory/figure", Figure(fig, close=True))

        return True

    def _create_metrics_figure(self, history):
        """创建监控图表"""
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))
        titles = ['Cavity Voltage (MV)', 'Phase (deg)', 'Detuning (Hz)', 'Piezo Adjustment (Hz)']
        for i, ax in enumerate(axs):
            ax.clear()
            ax.set_title(titles[i])
            data = list(history.values())[i][-1000:] if i < 3 else history['actions'][-1000:]
            ax.plot(data)
            ax.grid(True)
        plt.tight_layout()
        return fig


def run_trained_model(model_path="./sc_cavity_control_final.zip"):
    """运行训练好的模型"""
    env = SuperconductingCavityEnv(pulse_cycles=1)
    try:
        model = PPO.load(model_path, env=env)
    except FileNotFoundError:
        print("未找到预训练模型，使用新模型")
        model = PPO("MlpPolicy", env)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        if done: break

    # 可视化结果
    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    titles = ['Cavity Voltage (MV)', 'Phase (deg)', 'Detuning (Hz)', 'Piezo Adjustment (Hz)']
    for i, (ax, key) in enumerate(zip(axs, env.history.keys())):
        ax.plot(np.array(env.history[key], dtype=np.float32))
        ax.set_title(titles[i])
        ax.grid(True)
    plt.show()
    return env.history


if __name__ == "__main__":
    # 训练配置
    n_envs = 8  # 根据硬件调整
    env = SubprocVecEnv([lambda: SuperconductingCavityEnv() for _ in range(n_envs)])
    callback = TrainingCallback(check_freq=10000, save_path="./saved_models", env=env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log="./tb_logs/",
        device="cuda"
    )

    try:
        model.learn(total_timesteps=1e6, callback=callback, tb_log_name="ppo_sc_cavity")
    except KeyboardInterrupt:
        print("训练被用户中断")
    finally:
        model.save("sc_cavity_control_final")
        env.close()

    # 运行最终模型
    final_history = run_trained_model()
    np.savez('simulation_results.npz', **final_history)