import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import scipy.io as sio
import pandas as pd

# import matplotlib
# matplotlib.use('Qt5Agg')

EPSILON = 0.01
MAX_ITER = 100000000
RHO = 0.001


def phi_func(row, n_degree):
    if n_degree == -1:
        return row
    else:
        result = np.array([])
        for i in row:
            for j in range(1, n_degree + 1):
                result = np.concatenate([result, [np.power(i, j)]])
        return np.concatenate([result, [1]])


def linear_reg(data, s, n_degree=-1, gd_rho=RHO):
    data_copy = data.copy()
    for key in ['X_trn', 'X_tst', 'X_val']:
        temp = data_copy[key]
        temp = np.apply_along_axis(phi_func, 1, temp, n_degree)
        data_copy[key] = temp
    if s == 0:
        return closed_form_reg(data_copy)
    elif s == 1:
        return gradient_descent_reg(data_copy, rho=gd_rho)


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
    err = np.power(np.linalg.norm(err_mat), 2) / len(err_mat)
    return err


def gradient_descent_reg(data, max_iter=MAX_ITER, rho=RHO, epsilon=EPSILON):
    x_train = data['X_trn']
    y_train = data['Y_trn']

    curr_theta = np.zeros((len(x_train[0]), 1))
    i = 1

    xtx, xty = xtx_xty(x_train, y_train)

    for i in range(0, max_iter):
        # new_theta, new_dir = single_descent(curr_theta, rho, x_train, y_train)

        new_theta, new_dir = descent_v2(xtx, xty, curr_theta, rho)

        curr_theta = new_theta
        if np.linalg.norm(new_dir) < epsilon:
            print("The total number of iterations is: {} with rho = {}".format(i + 1, rho))
            break
    if i >= max_iter - 1:
        print("Exceeds max iteration: {}; with learning rate: {}".format(max_iter, rho))
    err_trn, err_tst, err_val = calc_error(data, curr_theta)
    return curr_theta, err_trn, err_tst, err_val


def single_descent(curr_theta, rho, x, y):
    xt_x = np.matmul(x.transpose(), x)
    xt_y = np.matmul(x.transpose(), y)
    new_dir = (np.matmul(xt_x, curr_theta) - xt_y)
    theta = curr_theta - new_dir * rho
    return theta, new_dir


def descent_v2(xtx, xty, curr_theta, rho):
    new_dir = (np.matmul(xtx, curr_theta) - xty)
    return curr_theta - new_dir * rho, new_dir


def xtx_xty(x, y):
    xt_x = np.matmul(x.transpose(), x)
    xt_y = np.matmul(x.transpose(), y)
    return xt_x, xt_y


def main():
    d1 = sio.loadmat('./data/dataset_hw2.mat')
    # print(d1.keys())
    d1.pop('__header__')
    d1.pop('__version__')
    d1.pop('__globals__')
    # print(d1.keys())
    # arow = np.array([1, 2, 3, 4, 5])
    # test = phi_func(arow, 3)
    # print(test)
    # x_train = d1['X_trn']
    # train_test = np.apply_along_axis(phi_func, 1, x_train, 3)
    # print(x_train)
    # print(train_test)
    # print(d1.keys())
    for n in range(5, 6):
        print('n = {}'.format(n))
        print("closed form calculation:")
        theta_star, err_trn, err_tst, err_val = linear_reg(d1, 0, n_degree=n)
        print("theta = \n {}".format(theta_star))
        print("training error: {}".format(err_trn))
        print("testing error: {}".format(err_tst))
        print("validation error: {}".format(err_val))
    rho_list = [0.003, 0.0002, 0.000011, 0.0000005, 0.00000003, 0.01, 0.01, 0.01, 0.01]
    for n in range(5, 6):
        tic = time.perf_counter()
        print('n = {}'.format(n))
        learning_rate = rho_list[n - 1]
        print("gradient descent calculation:")
        theta_star, err_trn, err_tst, err_val = linear_reg(d1, 1, n_degree=n, gd_rho=learning_rate)
        print("theta = \n {}".format(theta_star))
        print("training error: {}".format(err_trn))
        print("testing error: {}".format(err_tst))
        print("validation error: {}".format(err_val))
        toc = time.perf_counter()
        print("running time: in {} seconds".format(toc - tic))


if __name__ == '__main__':
    main()
