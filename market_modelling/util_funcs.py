import numpy as np
import torch
from torch.autograd import Variable
from scipy.sparse import coo_matrix
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def prepare_data(df_replay, x, sample_size, train):
    # To prepare [x, bid, c, market]
    c = []
    mi1 = [0] * 301
    mi2 = [0] * 301
    bi = [0] * 301
    b = []
    m1 = []
    m2 = []

    ddpg_bids = df_replay['ddpg_bid'].values[:sample_size]
    market_price = df_replay['market'].values[:sample_size]
    # h = [0] * 301
    for bid, market in zip(ddpg_bids, market_price):
        bid = int(bid)
        market = min(int(market), 300) #max market price is 300
        if train:
            if bid > market:
                c.append([0])
            else:
                c.append([1])

        bi[:bid+1] = [1] * (bid+1)
        b.append(bi)

        mi1[market] = 1
        mi2[:market] = [1] * market
        m1.append(mi1)
        m2.append(mi2)

    x = np.asarray(x.iloc[:sample_size, :], dtype=np.int)
    m1 = np.asarray(m1, dtype=np.float32)
    m2 = np.asarray(m2, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if np.sum(m1) == 0 or np.sum(m2) == 0:
        print('error')
        sys.exit()

    if train:
        c = np.asarray(c, dtype=np.float32)
        print(x.shape, c.shape)
        # print the winning prob.
        win_prob = np.sum(c) / len(c)
        print('winning prob. in the replay file is %.2f ' % (win_prob))
    return x, c, b, m1, m2


def prepare_data_multi(df_replay, x, sample_size, train, agent_index):

        # To prepare [x, bid, c, market]
        c = []
        mi1 = [0] * 301
        mi2 = [0] * 301
        bi = [0] * 301
        b = []
        m1 = []
        m2 = []

        agent = 'ddpg_bid_' + str(agent_index)

        ddpg_bids = df_replay[agent].values[:sample_size]
        market_price = df_replay['market'].values[:sample_size]
        # h = [0] * 301
        for bid, market in zip(ddpg_bids, market_price):
            bid = int(bid)
            market = min(int(market), 300)  # min market price is 300
            if train:
                if bid > market:
                    c.append([0])
                else:
                    c.append([1])

            bi[:bid + 1] = [1] * (bid + 1)
            b.append(bi)

            mi1[market] = 1
            mi2[:market] = [1] * market
            m1.append(mi1)
            m2.append(mi2)

        x = np.asarray(x.iloc[:sample_size, :], dtype=np.int)
        m1 = np.asarray(m1, dtype=np.float32)
        m2 = np.asarray(m2, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)

        if np.sum(m1) == 0 or np.sum(m2) == 0:
            print('error')
            sys.exit()

        if train:
            c = np.asarray(c, dtype=np.float32)
            print(x.shape, c.shape)
            # print the winning prob.
            win_prob = np.sum(c) / len(c)
            print('winning prob. in the replay file is %.2f ' % (win_prob))
        return x, c, b, m1, m2



def prepare_data_test(x, sample_size):
    # To prepare x
    x = np.asarray(x.iloc[:sample_size, :], dtype=np.int)
    return x


def get_onehot_data(x, vocal_size):
    # get one hot encoded data
    x = x.numpy()
    feature_nums = x.shape[1]
    x_onehot = []
    for i in range(x.shape[0]):
        row = np.array([0] * vocal_size)
        col_index = [int(i) for i in x[i, :]]
        row[col_index] = [1] * feature_nums
        x_onehot.append(row)
    # sparse matrix
    # coo = coo_matrix((np.array(data), (np.array(row), np.array(col))), shape=(x.shape[0], vocal_size))
    # print(coo.toarray())
    # print(np.max(x_onehot[0]))
    # print(x_onehot[0])
    x_onehot = np.asarray(x_onehot, dtype=float)
    return torch.from_numpy(x_onehot).type(torch.FloatTensor)


def loss_full(c_batch, b_batch, m1_batch, m2_batch, ypred_batch, beta):

    lz = -(1 - c_batch) * torch.add(torch.sum(torch.log(1 - ypred_batch * m2_batch), dim=1),
                                           torch.log(torch.sum(ypred_batch * m1_batch, dim=1)))
    lz = lz.sum()
    lunc = - torch.sum((1-c_batch) * torch.log(1-torch.prod((1-ypred_batch*b_batch), dim=1)))
    lcen = - torch.sum(c_batch * torch.sum(torch.log(1-ypred_batch*b_batch), dim=1))
    lc = lunc + lcen
    l_full = beta * lz + (1 - beta) * lc
    l_full = torch.mean(l_full)

    return l_full


def plot_figures(figs, figure_path, camp, c0, r, file_prefix):
    i = 0
    for loss in figs:
        if i == 0:
            plt.figure()
            plt.plot(range(len(loss)), loss, label='train loss')

        elif i == 1:
            plt.plot(range(len(loss)), loss, label='validation loss')
            plt.legend(loc='upper right')
            plt.savefig(figure_path + camp + file_prefix + 'train_val.pdf')
        else:
            plt.figure()
            plt.plot(range(len(loss)), loss, label='validation ANLP')
            plt.savefig(figure_path + camp + file_prefix + '_val_anlp.pdf')
        i += 1



