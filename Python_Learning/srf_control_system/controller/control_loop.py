# controller/control_loop.py
from utils.shared_memory import SharedMemoryManager
from controller.fuzzy_pid import FuzzyPIDController


class FrequencyControlSystem:
    def __init__(self):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.connect_shared_memory("srf_data")
        self.shm_manager.connect_semaphore("srf_sem")

        self.pid_controller = FuzzyPIDController(Kp=0.5, Ki=0.1, Kd=0.05)
        self.setpoint = 1.3e9  # 目标频率

    def run_control(self):
        while True:
            # 读取共享内存数据
            with self.shm_manager.semaphore:
                data = self.shm_manager.read_data()

            current_freq = np.abs(data[0])  # 假设vc的幅值代表频率

            # 计算控制量
            control_signal = self.pid_controller.update(self.setpoint, current_freq)

            # 写入控制参数（需在仿真器中实现接收接口）
            # ... 省略写入逻辑 ...

            # 控制周期
            time.sleep(self.Ts)