import numpy as np

if __name__ == "__main__":
    a = np.arange(24)
    print(a.ndim)
    b = a.reshape(2, 4, 3)
    print(b.ndim)
    print()

    c = np.array([[1, 2, 3], [4, 5, 6]])
    print(c)
    print(c.shape)
    print()

    d = np.array([[1, 2, 3], [4, 5, 6]])
    d.shape = (3, 2)
    print(d)
    print()

    e = np.array([[1, 2, 3], [4, 5, 6]])
    f = e.reshape(3, 2)
    print(e)
    print(f)
    print()

    x = np.array([1, 2, 3, 4, 5], dtype=np.int8)
    print(x.itemsize)
    y = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    print(y.itemsize)
    print()

    g = np.array([1, 2, 3, 4, 5])
    print(g.flags)
    print()

    h = np.empty([3, 2], dtype=int, order='C')
    print(h)
    print()

    i = np.zeros(5)
    print(i)
    j = np.zeros((5,), dtype=int)
    print(j)
    k = np.zeros((2, 2), dtype=[('x', 'i4'), ('y', 'i4')])
    print(k)
    print()

    o = np.ones(5)
    print(o)
    p = np.ones([2, 2], dtype=int)
    print(p)
    print()

    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(arr)
    zeros_arr = np.zeros_like(arr)
    print(zeros_arr)
    ones_arr = np.ones_like(arr)
    print(ones_arr)
    print()

