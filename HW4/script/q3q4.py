import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def data_plot(point, label):
    df = pd.DataFrame(np.concatenate([point, np.array([label]).T], axis=1),
                      columns=['x1', 'x2', 'class'])
    minus_pt = df[df['class'] == -1]
    one_pt = df[df['class'] == 1]
    minus_px, minus_py = minus_pt["x1"],  minus_pt["x2"]
    one_px, one_py = one_pt["x1"], one_pt["x2"]
    plt.scatter(minus_px, minus_py, c="red", label="-1")
    plt.scatter(one_px, one_py, c="blue", label="1")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()
    print(df)


def main():
    data = sio.loadmat('./data/hw04_data.mat')
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    x_trn = data["X_trn"]
    y_trn = data["y_trn"].T[0]
    x_tst = data["X_tst"]
    y_tst = data["y_tst"].T[0]
    data_plot(x_trn, y_trn)


if __name__ == '__main__':
    main()
