import matplotlib.pyplot as plt
import numpy as np

data_set = np.array(
    [[0.1, 0.15], [0.5, 0.4], [0.9, 0.85], [1.5, 1.62], [-0.2, -0.17], [-0.5, -0.42]]
)


def main():
    theta_star = calculate_theta()
    x = [p[0] for p in data_set]
    y = [p[1] for p in data_set]
    line_x = np.linspace(-1, 2, 2)
    plt.plot(line_x, line_x * theta_star)
    plt.scatter(x, y)
    plt.show()


def calculate_theta():
    """

    :return:
    """
    X = np.array([[p[0]] for p in data_set])
    Y = np.array([[p[1]] for p in data_set])
    inv = np.linalg.inv(np.matmul(X.transpose(), X))
    theta_star = np.matmul(inv, np.matmul(X.transpose(), Y))
    return theta_star[0][0]


if __name__ == "__main__":
    main()
