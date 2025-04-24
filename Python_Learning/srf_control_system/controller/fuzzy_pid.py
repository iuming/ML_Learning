# controller/fuzzy_pid.py
import numpy as np


class FuzzyPIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.error_history = []
        self.fuzzy_rules = self._initialize_fuzzy_rules()

    def _initialize_fuzzy_rules(self):
        # 实现模糊规则表（需根据实际系统调整）
        return np.array([
            [[-0.3, -0.1, 0.2], [-0.2, -0.05, 0.1], [-0.1, 0, 0.05]],
            [[-0.2, -0.05, 0.1], [0, 0, 0], [0.1, 0.05, -0.1]],
            [[-0.1, 0, 0.05], [0.1, 0.05, -0.1], [0.2, 0.1, -0.2]]
        ])

    def update(self, setpoint, measurement):
        error = setpoint - measurement
        de = error - self.error_history[-1] if self.error_history else 0
        self.error_history.append(error)

        # 模糊化
        e_fuzzy = self._fuzzify(error)
        de_fuzzy = self._fuzzify(de)

        # 模糊推理
        delta_Kp, delta_Ki, delta_Kd = self._infer_rules(e_fuzzy, de_fuzzy)

        # 反模糊化
        self.Kp += delta_Kp
        self.Ki += delta_Ki
        self.Kd += delta_Kd

        # PID计算
        integral = sum(self.error_history) * Ts
        derivative = (error - self.error_history[0]) / Ts
        return self.Kp * error + self.Ki * integral + self.Kd * derivative

    def _fuzzify(self, value):
        # 实现模糊化函数（需根据实际系统调整）
        return np.clip(np.digitize(value, bins=[-100, -50, 0, 50, 100]) - 1, 0, 2)

    def _infer_rules(self, e_fuzzy, de_fuzzy):
        # 实现模糊推理（需根据实际规则调整）
        return self.fuzzy_rules[e_fuzzy, de_fuzzy]