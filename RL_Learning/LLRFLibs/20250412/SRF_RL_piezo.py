import numpy as np
import gymnasium as gym
from gymnasium import spaces
from rich.markup import render
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque

from llrflibs.rf_sim import *
from llrflibs.rf_control import *

import torch

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


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
    def __init__(self, pulse_cycles=25, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # 环境参数
        self.Ts = 1e-6  # 仿真时间步长
        self.pulse_cycles = pulse_cycles
        self.current_cycle = 0

        # 初始化顺序保持不变
        self._init_mechanical_modes()
        self._init_cavity_parameters()
        self._init_simulation_parameters()

        # 动作和观测空间定义保持不变
        self.action_space = spaces.Box(
            low=-2 * np.pi * 100,
            high=2 * np.pi * 100,
            shape=(1,),
            dtype=np.float64
        )

        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, -2 * np.pi * 500]),
            high=np.array([20e6, np.pi, 2 * np.pi * 500]),
            dtype=np.float64
        )

        # 使用deque限制历史记录长度
        max_history = 10_000  # 根据内存调整
        self.history = {
            'vc_amp': deque(maxlen=max_history),
            'vc_phase': deque(maxlen=max_history),
            'detuning': deque(maxlen=max_history),
            'actions': deque(maxlen=max_history)
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
        super().reset(seed=seed)

        # 重置状态
        self.buf_id = 0
        self.current_cycle = 0
        self.state_vc = 0.0 + 0j
        self.state_m = np.matrix(np.zeros(self.Bd.shape))
        self.pha_src = 0.0
        self.dw_current = 0.0

        # 清空历史记录
        for k in self.history:
            self.history[k].clear()

        return self._get_obs().flatten(), {}

    def step(self, action):
        dw_piezo = action[0]  # 确保动作是标量

        cycle_rewards = []
        for _ in range(self.buf_size):
            self._simulation_step(dw_piezo)
            self._record_data()  # 记录标量数据
            cycle_rewards.append(-abs(self.dw_current / (2 * np.pi)))

        self.current_cycle += 1

        return (
            self._get_obs().flatten(),
            np.mean(cycle_rewards),
            self.current_cycle >= self.pulse_cycles,
            False,
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

        # 在执行仿真后添加校验
        if not np.isfinite(self.vc):
            raise ValueError("检测到非法的腔体电压值")
        if abs(self.dw_current) > 2 * np.pi * 1e6:
            raise ValueError(f"失谐量超出范围: {self.dw_current / (2 * np.pi):.1f} Hz")

    def _get_obs(self):
        # 确保返回标量观测值
        return np.array([
            float(abs(self.state_vc)),  # 幅值 (V)
            float(np.angle(self.state_vc)),  # 相位 (rad)
            float(self.dw_current / (2 * np.pi))  # 失谐量 (Hz)
        ], dtype=np.float32)

    def _record_data(self):
        """确保记录标量数据"""
        try:
            # 显式转换为Python原生float类型
            self.history['vc_amp'].append(float(abs(self.state_vc) * 1e-6))  # 转换为MV
            self.history['vc_phase'].append(float(np.angle(self.state_vc) * 180 / np.pi))
            detuning = float(self.dw_current / (2 * np.pi))
            self.history['detuning'].append(detuning)
            self.history['actions'].append(detuning)  # 假设动作与失谐量量纲相同

            # 调试日志
            if len(self.history['vc_amp']) % 1000 == 0:
                print(f"记录数据点: {len(self.history['vc_amp'])}")
        except Exception as e:
            print(f"数据记录错误: {str(e)}")
            raise

    def _sim_rfsrc(self):
        pha = self.pha_src + 2.0 * np.pi * self.fsrc * self.Ts
        return self.Asrc * np.exp(1j * pha), pha

    def _sim_iqmod(self, S0):
        return S0 * self.base_pul[self.buf_id]

    def _sim_amp(self, S1):
        return S1 * 10 ** (self.gain_dB / 20.0)

    def render(self, mode='human'):
        # 添加数据有效性检查
        if not all(len(v) > 10 for v in self.history.values()):
            return

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

        # plt.pause(0.001)


class TrainingCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, env: gym.Env, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.env = env
        self.best_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self.model.save(f"{self.save_path}/model_{self.num_timesteps}")
            # 使用第一个子环境的 history 计算平均失谐量
            avg_reward = np.mean(self.env.envs[0].history['detuning'][-1000:])
            self.logger.record("train/avg_detuning", avg_reward)

            if len(self.env.envs[0].history['vc_amp']) > 100:
                fig = self._create_metrics_figure()
                self.logger.record("trajectory/figure", Figure(fig, close=True),
                                   exclude=("stdout", "log", "json", "csv"))

            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.model.save(f"{self.save_path}/best_model")

            # 检查数据维度
            hist = self.env.envs[0].history
            if any(not all(isinstance(x, float) for x in v) for v in hist.values()):
                print("警告: 检测到非浮点数历史数据")

        if self.n_calls % 100 == 0:
            self.env.envs[0].render()

        return True


def run_trained_model(model_path="./sc_cavity_control_final.zip"):
    env = SuperconductingCavityEnv(pulse_cycles=1)
    model = PPO.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)

    fig, axs = plt.subplots(4, 1, figsize=(12, 10))
    titles = ['腔体电压 (MV)', '相位 (度)', '失谐量 (Hz)', '压电调整量 (Hz)']
    keys = ['vc_amp', 'vc_phase', 'detuning', 'actions']

    for i, (ax, key) in enumerate(zip(axs, keys)):
        try:
            # 转换前检查数据类型
            raw_data = list(env.history[key])
            if not raw_data:
                print(f"无数据: {key}")
                continue

            # 处理可能的嵌套结构
            if isinstance(raw_data[0], (list, np.ndarray)):
                print(f"展开嵌套数据: {key}")
                raw_data = [item for sublist in raw_data for item in sublist]

            data = np.asarray(raw_data, dtype=np.float32).squeeze()
            ax.plot(data[-5000:])  # 仅显示最后5000个点
            ax.set_title(titles[i])
            ax.grid(True)
        except Exception as e:
            print(f"绘图错误 {key}: {str(e)}")

    plt.tight_layout()
    plt.show()
    env.close()
    return env.history


# 训练配置（保持不变）
if __name__ == "__main__":
    env = DummyVecEnv([lambda: SuperconductingCavityEnv(render_mode="human")])

    callback = TrainingCallback(
        check_freq=10000,
        save_path="./saved_models",
        env=env
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=16,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=2048,
        tensorboard_log="./tb_logs/",
        device="cuda"
    )

    try:
        model.learn(
            total_timesteps=1e1,
            callback=callback,
            progress_bar=True,
            tb_log_name="ppo_sc_cavity"
        )
    except KeyboardInterrupt:
        print("训练被用户中断")

    model.save("sc_cavity_control_final")
    env.close()

    final_history = run_trained_model()
    np.savez('simulation_results.npz', **final_history)