from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd


def logisticReg(X, y):
    log_reg = LogisticRegression()
    x = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    log_reg.fit(x, y)
    w = log_reg.coef_[:, 0:X.shape[1]][0]
    b = log_reg.coef_[0][X.shape[1]]
    return log_reg, w, b


def data_plot(point, label, w, b, title):
    df = pd.DataFrame(np.concatenate([point, np.array([label]).T], axis=1),
                      columns=['x1', 'x2', 'class'])
    zero_pt = df[df['class'] == 0]
    one_pt = df[df['class'] == 1]
    zero_px = zero_pt["x1"]
    zero_py = zero_pt["x2"]
    one_px = one_pt["x1"]
    one_py = one_pt["x2"]
    lx = np.linspace(df['x1'].min(), df['x1'].max(), num=100)
    ly = (- w[0] * lx + b) / w[1]
    plt.scatter(zero_px, zero_py, c="red", label="0")
    plt.scatter(one_px, one_py, c="yellow", label="1")
    plt.plot(lx, ly, label="Decision Boundary")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.title(title)
    plt.show()


def main():
    d1 = sio.loadmat('./data/hw03_dataset.mat')
    d1.pop('__header__')
    d1.pop('__version__')
    d1.pop('__globals__')
    # print(d1)
    X = d1["X_trn"]
    y = d1["Y_trn"].T[0]
    res, w, b = logisticReg(X, y)
    score = res.score(np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), y)
    print("Error: {}".format(score))
    print("w: {}, b: {}".format(w, b))
    data_plot(X, y, w, b, "trn")


if __name__ == "__main__":
    main()
