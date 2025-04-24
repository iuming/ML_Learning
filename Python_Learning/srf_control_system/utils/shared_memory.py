# utils/shared_memory.py
import posix_ipc
import mmap
import numpy as np

class SharedMemoryManager:
    def __init__(self):
        self.shm = None
        self.semaphore = None

    def create_shared_memory(self, name, size):
        self.shm = posix_ipc.SharedMemory(name, flags=posix_ipc.O_CREAT, size=size)
        self.memory = mmap.mmap(self.shm.fd, self.shm.size)

    def connect_shared_memory(self, name):
        self.shm = posix_ipc.SharedMemory(name)
        self.memory = mmap.mmap(self.shm.fd, self.shm.size)

    def create_semaphore(self, name, initial_value=1):
        self.semaphore = posix_ipc.Semaphore(name, flags=posix_ipc.O_CREAT, initial_value=initial_value)

    def connect_semaphore(self, name):
        self.semaphore = posix_ipc.Semaphore(name)

    def write_data(self, data):
        self.memory[:data.nbytes] = data.tobytes()

    def read_data(self):
        data = np.frombuffer(self.memory.read(), dtype=np.complex128)
        return data