import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.io as sio

EPSILON = 0.01
MAX_ITER = 10000000
RHO = 0.01


def phi_func(row, n_degree):
    if n_degree == -1:
        return row
    else:
        result = np.array([])
        for i in row:
            for j in range(1, n_degree + 1):
                result = np.concatenate([result, [np.power(i, j)]])
        return np.concatenate([result, [1]])


def feature_normalization(data):
    trn = data['X_trn']
    tst = data['X_tst']
    val = data['X_val']
    for i in range(0, len(trn[0])):
        avg = np.mean(trn[:, i])
        max_minus_min = np.max(trn[:, [i]]) - np.min(trn[:, [i]])
        trn[:, [i]] = (trn[:, [i]] - avg) / max_minus_min
        tst[:, [i]] = (tst[:, [i]] - avg) / max_minus_min
        val[:, [i]] = (val[:, [i]] - avg) / max_minus_min
    data['X_trn'] = trn
    data['X_tst'] = tst
    data['X_val'] = val
    return data


def linear_reg(data, s, n_degree=-1, gd_rho=RHO):
    data_copy = data.copy()
    for key in ['X_trn', 'X_tst', 'X_val']:
        data_copy[key] = np.apply_along_axis(phi_func, 1, data_copy[key], n_degree)
    # start regression:
    if s == 0:
        return (data_copy,) + closed_form_reg(data_copy)
    elif s == 1:
        return (data_copy,) + gradient_descent_reg(data_copy, rho=gd_rho)


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

    # curr_theta = np.zeros((len(x_train[0]), 1))
    curr_theta = np.full((len(x_train[0]), 1), 0)

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
    # print(new_dir)
    return curr_theta - new_dir * rho, new_dir


def xtx_xty(x, y):
    xt_x = np.matmul(x.transpose(), x)
    xt_y = np.matmul(x.transpose(), y)
    return xt_x, xt_y


def plot_regression(data, theta, title=""):
    # plot scatter point from data set
    x = data['x']
    y = data['y']
    x_first_col = x[:, [0]]
    plt.scatter(x_first_col, y)
    # plot line generated from theta
    theta_x = np.array([np.arange(-1, 1, 0.01)]).transpose()
    expanded_x = np.apply_along_axis(phi_func, 1, theta_x, len(theta) - 1)
    theta_y = np.matmul(expanded_x, theta)
    plt.plot(list(theta_x), list(theta_y))
    plt.title(title + " n = {}".format(theta.shape[0] - 1))
    plt.show()


def main():
    d1 = sio.loadmat('./data/dataset_hw2.mat')
    d1.pop('__header__')
    d1.pop('__version__')
    d1.pop('__globals__')
    d1 = feature_normalization(d1)

    closed_trn_err_list = []
    closed_tst_err_list = []
    closed_val_err_list = []
    gd_trn_err_list = []
    gd_tst_err_list = []
    gd_val_err_list = []
    for n in range(1, 10):
        print('n = {}'.format(n))
        print("closed form calculation:")
        data_copy, theta_star, err_trn, err_tst, err_val = linear_reg(d1, 0, n_degree=n)
        closed_trn_err_list.append(err_trn)
        closed_tst_err_list.append(err_tst)
        closed_val_err_list.append(err_val)
        print("theta transpose = \n {}".format(theta_star.transpose()))
        print("training error: {}".format(err_trn))
        print("testing error: {}".format(err_tst))
        print("validation error: {}".format(err_val))
        print("gradient descent calculation:")
        plot_regression({"x": data_copy['X_trn'], "y": data_copy["Y_trn"]}, theta_star, title="closed form training set")
        # plot_regression({"x": data_copy['X_tst'], "y": data_copy["Y_tst"]}, theta_star, title="closed form testing set")
        # plot_regression({"x": data_copy['X_val'], "y": data_copy["Y_val"]}, theta_star, title="closed form validation set")

        tic = time.perf_counter()
        data_copy, theta_star, err_trn, err_tst, err_val = linear_reg(d1, 1, n_degree=n)
        gd_trn_err_list.append(err_trn)
        gd_tst_err_list.append(err_tst)
        gd_val_err_list.append(err_val)
        print("theta transpose = \n {}".format(theta_star.transpose()))
        print("training error: {}".format(err_trn))
        print("testing error: {}".format(err_tst))
        print("validation error: {}".format(err_val))
        toc = time.perf_counter()
        print("running time: in {} seconds".format(toc - tic))
        plot_regression({"x": data_copy['X_trn'], "y": data_copy["Y_trn"]}, theta_star, title="gradient descent training set")
        # plot_regression({"x": data_copy['X_tst'], "y": data_copy["Y_tst"]}, theta_star, title="gradient descent testing set")
        # plot_regression({"x": data_copy['X_val'], "y": data_copy["Y_val"]}, theta_star, title="gradient descent validation set")

    # err_list = {"closed form training error": closed_trn_err_list,
    #             "closed form testing error": closed_tst_err_list,
    #             "closed form validating error": closed_val_err_list,
    #             "gradient descent training error": gd_trn_err_list,
    #             "gradient descent testing error": gd_tst_err_list,
    #             "gradient descent validating error": gd_val_err_list}
    # for err in ["closed form training error",
    #             "closed form testing error",
    #             "closed form validating error"]:
    #     plt.plot(np.arange(1, 10, 1), err_list[err])
    # plt.show()
    # for err in ["gradient descent training error",
    #             "gradient descent testing error",
    #             "gradient descent validating error"]:
    #     plt.plot(np.arange(1, 10, 1), err_list[err])
    # plt.show()


def test():
    trn = np.array([[1., -2., 3., 1],
                    [-1., 2., -3., 1],
                    [1., -2., -3., 1],
                    [-1., 2., 3., 1]])
    tst = np.array([[1., -1., 1., 1],
                    [1., 1., -1., 1]])
    val = np.array([[1., -1., 1., 1]])

    # # print(np.average(val[:, 1]))
    # trn, tst, val = feature_normalization(trn, tst, val)
    # print(trn)
    # print(tst)
    # print(val)


if __name__ == '__main__':
    main()
    # test()
