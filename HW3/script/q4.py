from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.io as sio


def logisticReg(X, y):
    log_reg = LogisticRegression()
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    log_reg.fit(X, y)
    return log_reg


def main():
    d1 = sio.loadmat('./data/hw03_dataset.mat')
    d1.pop('__header__')
    d1.pop('__version__')
    d1.pop('__globals__')
    # print(d1)
    X = d1["X_trn"]
    y = d1["Y_trn"].T[0]
    res = logisticReg(X, y)
    score = res.score(np.concatenate([X, np.ones((X.shape[0], 1))], axis=1), y)
    print("Error: {}".format(score))
    print("w: {}, b: {}".format(res.coef_[:, 0:2][0], res.coef_[0][2]))


if __name__ == "__main__":
    main()
