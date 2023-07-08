import numpy as np

if __name__ =="__main__":
    a = np.array([1,2,3], dtype=bool);
    print("a:", a)
    b = np.dtype(np.int32)
    print(b)
    c = np.dtype('i4')
    print(c)
    d = np.dtype('<i4')
    print(d)
    e = np.dtype([('age', np.int8)])
    print(e)

    f = np.dtype([('age', np.int8)])
    g = np.array([(10,), (20,), (30,)], dtype=f)
    print(g)

    student = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
    print(student)

    student2 = np.dtype([('name', 'S20'), ('age', 'i1'), ('marks', 'f4')])
    h = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=student2)
    print(h)