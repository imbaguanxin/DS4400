import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io as sio
import pandas as pd

# import matplotlib
# matplotlib.use('Qt5Agg')

EPSILON = 0.001
MAX_ITER = 10000
RHO = 0.0001


def phi(data):
    return data


def linear_reg(data, s):
    if s == 0:
        return closed_form_reg(data)
    elif s == 1:
        return gradient_descent_reg(data)


def closed_form_reg(data):
    theta_star = closed_form_theta(data)
    err_trn, err_tst, err_val = calc_error(data, theta_star)
    return theta_star, err_trn, err_tst, err_val


def closed_form_theta(data):
    x_trn = data['X_trn']
    y_trn = data['Y_trn']
    inv = np.linalg.inv(np.matmul(x_trn.transpose(), x_trn))
    theta_star_mat = np.matmul(inv, np.matmul(x_trn.transpose(), y_trn))
    return theta_star_mat


def calc_error(data, theta):
    x_train = data['X_trn']
    x_test = data['X_tst']
    x_val = data['X_val']
    y_train = data['Y_trn']
    y_test = data['Y_tst']
    y_val = data['Y_val']
    err_train = calc_error_xytheta(x_train, y_train, theta)
    err_test = calc_error_xytheta(x_test, y_test, theta)
    err_val = calc_error_xytheta(x_val, y_val, theta)
    return err_train, err_test, err_val


def calc_error_xytheta(x_mat, y_mat, theta_mat):
    err_mat = np.subtract(y_mat, np.matmul(theta_mat.transpose(), x_mat.transpose()).transpose())
    # print("error dim: {}", err_mat.shape)
    return err_mat


def gradient_descent_reg(data, max_iter=MAX_ITER, rho=RHO, epsilon=EPSILON):
    x_train = data['X_trn']
    y_train = data['Y_trn']

    curr_theta = np.zeros((len(x_train[0]), 1))
    i = 1
    for i in range(0, max_iter):
        new_theta, new_dir = single_descent(curr_theta, rho, x_train, y_train)
        curr_theta = new_theta
        if np.linalg.norm(new_dir) < epsilon:
            print("The total number of iterations is: {}".format(i + 1))
            break
    if i >= max_iter - 1:
        print("Exceeds max iteration: {}; with learning rate: {}".format(max_iter, rho))
    err_trn, err_tst, err_val = calc_error(data, curr_theta)
    return curr_theta, err_trn, err_tst, err_val


def single_descent(curr_theta, rho, x, y):
    xt_x = np.matmul(x.transpose(), x)
    xt_y = np.matmul(x.transpose(), y)
    new_dir = 2 * (np.matmul(xt_x, curr_theta) - xt_y)
    theta = curr_theta - new_dir * rho
    return theta, new_dir


def main():
    d1 = sio.loadmat('./data/dataset_hw2.mat')
    # print(d1.keys())
    theta_star, err_trn, err_tst, err_val = linear_reg(d1, 0)
    print(theta_star)
    print(err_trn)
    print(err_tst)
    print(err_val)
    theta_star, err_trn, err_tst, err_val = linear_reg(d1, 1)
    print(theta_star)
    print(err_trn)
    print(err_tst)
    print(err_val)


if __name__ == '__main__':
    main()
