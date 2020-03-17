import numpy as np

x1y = np.array([[2 / 3, 1 / 2], [1 / 3, 1 / 2]])
x2y = np.array([[2 / 3, 1 / 2], [1 / 3, 1 / 2]])
x1x2 = np.array([[2 / 7, 2 / 7], [2 / 7, 1 / 7]])
y = np.array([3 / 7, 4 / 7])


def main():
    s_um = 0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                s_um += cal(i, j, k)
    print(s_um)
    s_um = 0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                s_um += cal2(i, j, k)
    print(s_um)


def cal(a, b, c):
    return x1y[a][c] * x2y[b][c] * y[c] / ((y[0] * (x1y[a][0] * x2y[b][0])) + (y[1] * (x1y[a][1] * x2y[b][1])))


def cal2(a, b, c):
    return x1y[a][c] * x2y[b][c] * y[c] / x1x2[a][b]


if __name__ == '__main__':
    main()
