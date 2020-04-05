import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

sigmoid = 'sigmoid'
tanh = 'tanh'
relu = 'relu'
identity = 'identity'

module_dic = {
    sigmoid: nn.Sigmoid,
    tanh: nn.Tanh,
    relu: nn.ReLU,
    identity: nn.Identity
}


class Test_net(nn.Module):
    def __init__(self):
        super(Test_net, self).__init__()
        self.l1 = nn.Linear(2, 10)
        self.l1_f = nn.Sigmoid()
        self.l2 = nn.Linear(10, 2)
        self.l2_f = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.l1_f(x)
        x = self.l2(x)
        return self.l2_f(x)


class Net(nn.Module):

    def __init__(self, sl, activation):
        super(Net, self).__init__()
        self.layers = []
        self.activate_func = []
        if len(sl) == len(activation) + 1:
            for i in range(0, len(sl) - 1):
                from_dim = sl[i]
                to_dim = sl[i + 1]
                acti_func = activation[i]
                self.layers.append(nn.Linear(from_dim, to_dim))
                if acti_func in module_dic:
                    self.layers.append(module_dic[acti_func]())
                else:
                    raise ValueError(
                        """
                        activation function not supported!
                        Supported function: {}
                        Your function: {} on layer {}
                        """.format(module_dic.keys, acti_func, i + 1))
        else:
            raise ValueError(
                """
                The number of layer and the activation function doesn't match! (layer = activation + 1)
                number of layer:{}
                number of activation function:{}
                """.format(len(sl), len(activation))
            )

    def forward(self, x):
        for layer in self.layers:
            # print(x)
            x = layer(x)
        # print(x)
        return x


def test_build_nn(sl, activation):
    layers = []
    if len(sl) == len(activation) + 1:
        for i in range(0, len(sl) - 1):
            from_dim = sl[i]
            to_dim = sl[i + 1]
            acti_func = activation[i]
            layers.append(nn.Linear(from_dim, to_dim))
            if acti_func in module_dic:
                layers.append(module_dic[acti_func]())
            else:
                raise ValueError(
                    """
                    activation function not supported!
                    Supported function: {}
                    Your function: {} on layer {}
                    """.format(module_dic.keys, acti_func, i + 1))
        return nn.Sequential(*layers)
    else:
        raise ValueError(
            """
            The number of layer and the activation function doesn't match! (layer = activation + 1)
            number of layer:{}
            number of activation function:{}
            """.format(len(sl), len(activation))
        )


def data_plot(point, label):
    df = pd.DataFrame(np.concatenate([point, np.array([label]).T], axis=1),
                      columns=['x1', 'x2', 'class'])
    minus_pt = df[df['class'] == -1]
    one_pt = df[df['class'] == 1]
    minus_px, minus_py = minus_pt["x1"], minus_pt["x2"]
    one_px, one_py = one_pt["x1"], one_pt["x2"]
    plt.scatter(minus_px, minus_py, c="red", label="-1")
    plt.scatter(one_px, one_py, c="blue", label="1")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = sio.loadmat('./data/hw04_data.mat')
    data.pop('__header__')
    data.pop('__version__')
    data.pop('__globals__')
    x_trn = data["X_trn"]
    y_trn = data["y_trn"]
    x_tst = data["X_tst"]
    y_tst = data["y_tst"]
    data_plot(x_trn, y_trn.T[0])
    data_plot(x_tst, y_tst.T[0])

    x_trn_tensor = torch.tensor(x_trn).float()
    x_tst_tensor = torch.tensor(x_tst).float()
    y_trn_tensor = torch.tensor(y_trn.squeeze() + 1 / 2).long()
    y_tst_tensor = torch.tensor(y_tst.squeeze() + 1 / 2).long()
    print(len(x_trn_tensor))
    # print(y_trn_tensor)

    # net = Net([2, 10, 2], [relu, relu])
    net = test_build_nn([2, 10, 2], [relu, relu])
    print(net)
    # net = Test_net()
    # print(net)
    epoches = 20
    bs = 100
    lr = 0.001
    n = len(x_trn_tensor)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    for epoch in range(epoches):
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_trn_tensor[start_i:end_i]
            yb = y_trn_tensor[start_i:end_i]
            pred = net(xb)
            loss = criterion(pred, yb)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            # with torch.no_grad():
            #     for p in net.parameters():
            #         p -= p.grad * lr
            #     net.zero_grad()

    _, predi = torch.max(net(x_tst_tensor), 1)
    print(predi)
    print(y_tst_tensor)
    print()
    print(torch.abs(y_tst_tensor - predi).sum().float() / len(predi))
    # print(criterion(net(x_tst_tensor), y_tst_tensor))
    param = list(net.parameters())
    print(param)


if __name__ == '__main__':
    main()
    # test_net = Net([1, 10, 1], [sigmoid, relu])
    #
    # data = torch.randn(10, 1, 1).view([1, -1])
    # print(data)
    # test_net.zero_grad()
    # output = test_net(data)
    # target = torch.randn(1,1).view([1, -1])
    # print(target)
    # criterion = nn.CrossEntropyLoss()
    #
    # loss = criterion(output, target)
    # print(loss)
