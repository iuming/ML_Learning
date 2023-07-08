import numpy as np

if __name__ == '__main__':
    a = np.array([1, 2, 3])
    print("a:", a)
    b = np.array([[1,2],[3,4]])
    print("b:", b)
    c = np.array([1, 2, 3, 4, 5], ndmin = 2)
    print("c:", c)
    d = np.array([1, 2, 3], dtype=complex)
    print("d:", d)