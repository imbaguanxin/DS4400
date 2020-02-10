import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as sio
import pandas as pd

# import matplotlib
# matplotlib.use('Qt5Agg')


def linear_reg(training, validation, test, s, phi):
    training_phi = phi(training)
    validation_phi = phi(validation)
    test_phi = phi(test)
    if s == 0:
        closed_form_reg(training_phi, validation_phi, test_phi)
    else:
        gredient_descent_reg(training_phi, validation_phi, test_phi)


def closed_form_reg(train, validate, test):
    X = np.array([[p[0]] for p in train])
    Y = np.array([[p[1]] for p in train])
    inv = np.linalg.inv(np.matmul(X.transpose(), X))
    theta_star = np.matmul(inv, np.matmul(X.transpose(), Y))
    theta_star = theta_star[0]


def calc_error(data, theta):
    pass


def gredient_descent_reg(train, validate, test):
    pass

def main():
    d1 = sio.loadmat('./data/dataset1.mat')
    # print(d1)
    x_tst = np.array(d1['X_tst'])
    print(x_tst)
    print(x_tst.T)
    inv = np.linalg.inv(np.matmul(x_tst.transpose(), x_tst))
    print(inv)

if __name__ == '__main__':
    main()

