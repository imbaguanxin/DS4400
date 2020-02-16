import matplotlib.pyplot as plt
import time
import numpy as np
import scipy.io as sio

EPSILON = 0.01
MAX_ITER = 100000
RHO = 0.01


def phi_func(row, n_degree):
    """
    This is the phi function that apply to each row
    """
    if n_degree == -1:
        return row
    else:
        result = np.array([])
        for i in row:
            for j in range(1, n_degree + 1):
                result = np.concatenate([result, [np.power(i, j)]])
        return np.concatenate([result, [1]])


def feature_normalization(trn, tst, val):
    """
    This function normalize all training, testing and validation sets.
    It normalize the testing and validation sets with parameter computed from training set.
    """
    for i in range(0, len(trn[0]) - 1):
        avg = np.mean(trn[:, i])
        max_minus_min = np.max(trn[:, [i]]) - np.min(trn[:, [i]])
        trn[:, [i]] = (trn[:, [i]] - avg) / max_minus_min
        tst[:, [i]] = (tst[:, [i]] - avg) / max_minus_min
        val[:, [i]] = (val[:, [i]] - avg) / max_minus_min
    return trn, tst, val


def linear_reg(data, s, lam_da, n_degree=-1, gd_rho=RHO):
    """
    The main function of linear regression
    input s = 1 to do gradient descent and s = 0 to do closed form
    """
    data_copy = data.copy()
    for key in ['X_trn', 'X_tst', 'X_val']:
        data_copy[key] = np.apply_along_axis(phi_func, 1, data_copy[key], n_degree)
    # feature normalization:
    x_trn = data_copy['X_trn']
    x_tst = data_copy['X_tst']
    x_val = data_copy['X_val']
    x_trn, x_tst, x_val = feature_normalization(x_trn, x_tst, x_val)
    data_copy['X_trn'] = x_trn
    data_copy['X_tst'] = x_tst
    data_copy['X_val'] = x_val
    # start regression:
    if s == 0:
        return (data_copy,) + closed_form_reg(data_copy, lam_da)
    elif s == 1:
        return (data_copy,) + gradient_descent_reg(data_copy, lam_da, rho=gd_rho)


def closed_form_reg(data, lam_da):
    """
    wrapper of closed form regression.
    Calculate the learning result, then calculate the errors
    """
    theta_star = closed_form_theta(data, lam_da)
    err_trn, err_tst, err_val = calc_error(data, theta_star)
    return theta_star, err_trn, err_tst, err_val


def closed_form_theta(data, lam_da):
    """
    calculate optimal theta from training data and given lambda
    """
    x_trn = data['X_trn']
    y_trn = data['Y_trn']
    xtx = np.matmul(x_trn.transpose(), x_trn)
    l_id = lam_da * np.eye(len(xtx[0]), dtype=float)
    inv = np.linalg.inv(xtx + l_id)
    theta_star_mat = np.matmul(inv, np.matmul(x_trn.transpose(), y_trn))
    return theta_star_mat


def calc_error(data, theta):
    """
    Calculating errors of training, testing and validation sets
    """
    x_train = data['X_trn']
    x_test = data['X_tst']
    x_val = data['X_val']
    y_train = data['Y_trn']
    y_test = data['Y_tst']
    y_val = data['Y_val']
    err_train = calc_error_xytheta(x_train, y_train, theta)
    err_test = calc_error_xytheta(x_test, y_test, theta)
    err_val = calc_error_xytheta(x_val, y_val, theta)
    print("err val shape: {}".format(err_val.shape))
    return err_train, err_test, err_val


def calc_error_xytheta(x_mat, y_mat, theta_mat):
    """
    A helper that calculates the errors given PHI(X), Y and optimal theta
    """
    err_mat = np.subtract(y_mat, np.matmul(theta_mat.transpose(), x_mat.transpose()).transpose())
    err = np.power(np.linalg.norm(err_mat), 2) / len(err_mat)
    return err


def gradient_descent_reg(data, lamb_da, max_iter=MAX_ITER, rho=RHO, epsilon=EPSILON):
    """
    The main function that does the gradient descent.
    """
    x_train = data['X_trn']
    y_train = data['Y_trn']
    curr_theta = np.full((len(x_train[0]), 1), 0)
    i = 1
    for i in range(0, max_iter):
        new_theta, new_dir = single_descent(curr_theta, rho, x_train, y_train, lamb_da)
        curr_theta = new_theta
        if np.linalg.norm(new_dir) < epsilon:
            print("The total number of iterations is: {} with rho = {}".format(i + 1, rho))
            break
    if i >= max_iter - 1:
        print("Exceeds max iteration: {}; with learning rate: {}".format(max_iter, rho))
    err_trn, err_tst, err_val = calc_error(data, curr_theta)
    return curr_theta, err_trn, err_tst, err_val


def single_descent(curr_theta, rho, x, y, lamb_da):
    xt_x = np.matmul(x.transpose(), x)
    xt_y = np.matmul(x.transpose(), y)
    new_dir = (np.matmul(xt_x, curr_theta) - xt_y) + lamb_da * curr_theta
    theta = curr_theta - new_dir * rho
    return theta, new_dir


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
    plt.title(title)
    plt.show()


def main():
    d1 = sio.loadmat('./data/dataset_hw2.mat')
    d1.pop('__header__')
    d1.pop('__version__')
    d1.pop('__globals__')

    closed_val_err_list = []
    gd_val_err_list = []
    for lamb_da in [0.001, 0.01, 0.1, 1, 10, 100]:
        print('n = 7, lambda = {}'.format(lamb_da))
        # print("closed form calculation:")
        data_copy, theta_star, err_trn, err_tst, err_val = linear_reg(d1, 0, lamb_da, n_degree=7)
        closed_val_err_list.append(err_val)
        # print("theta transpose = \n {}".format(theta_star.transpose()))
        # print("validation error: {}".format(err_val))
        print("gradient descent calculation:")
        plot_regression({"x": data_copy['X_val'], "y": data_copy["Y_val"]}, theta_star, title="closed form validation set lambda = {}".format(lamb_da))

        data_copy, theta_star, err_trn, err_tst, err_val = linear_reg(d1, 1, lamb_da, n_degree=7)
        gd_val_err_list.append(err_val)
        # print("theta transpose = \n {}".format(theta_star.transpose()))
        print("validation error: {}".format(err_val))
        plot_regression({"x": data_copy['X_val'], "y": data_copy["Y_val"]}, theta_star, title="gradient descent validation set lambda = {}".format(lamb_da))


        # print(closed_val_err_list)
        # print(gd_val_err_list)
    # plot_regression({"x": data_copy['X_trn'], "y": data_copy["Y_trn"]}, theta_star)
    # plot_regression({"x": data_copy['X_tst'], "y": data_copy["Y_tst"]}, theta_star)
    # plot_regression({"x": data_copy['X_val'], "y": data_copy["Y_val"]}, theta_star)

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
