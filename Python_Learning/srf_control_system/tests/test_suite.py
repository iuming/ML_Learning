# tests/test_suite.py
import unittest
from simulator.rf_simulator import RFSimulator
from controller.fuzzy_pid import FuzzyPIDController

class TestSRFSystem(unittest.TestCase):
    def test_fuzzy_pid(self):
        pid = FuzzyPIDController(Kp=0.5, Ki=0.1, Kd=0.05)
        output = pid.update(1.3e9, 1.2e9)
        self.assertIsInstance(output, complex)

    def test_shared_memory(self):
        manager = SharedMemoryManager()
        manager.create_shared_memory("test_data", size=1024)
        manager.write_data(np.array([1+2j], dtype=np.complex128))
        self.assertEqual(manager.read_data()[0], 1+2j)