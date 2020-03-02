import numpy as np
from numpy import genfromtxt
from sklearn.metrics import roc_auc_score
from math import exp, log, sqrt

np.random.seed(0)

DATA_PATH_PREFIX = './make-ipinyou-data/'

camps = ['2259']


class ftrl_proximal(object):
    def __init__(self, alpha=1., beta=1., L1=1., L2=1., D=70000, interaction=False):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield int(index)

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i + 1, L):
                    # one-hot encode interactions with hash trick
                    yield abs(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)



def split_camp_data(camp):
    camp_data_path = DATA_PATH_PREFIX + camp + '/train.yzx.txt'
    camp_data = genfromtxt(camp_data_path, delimiter=' ')
    click_price = camp_data[:, 0:2]
    Y = camp_data[:, 0:1]
    X = camp_data[:, 3:]
    Y = Y.reshape((Y.shape[0],))
    return X, Y, click_price


def train_ctr_estimator(X, Y):
    epoch = 3
    learner = ftrl_proximal(alpha=1., beta=1., L1=1.,
                            L2=1., D=70000, interaction=False)
    for e in range(epoch):
        loss = 0.
        count = 0
        for x, y in zip(X, Y):
            p = learner.predict(x)
            loss += logloss(p, y)
            count += 1
            learner.update(x, p, y)
        print('Epoch %d finished, logloss: %f' % (e, loss / count))
    proba = [learner.predict(x) for x in X]
    print(roc_auc_score(Y, proba))
    return proba


def write_ctr(camp, click_price, proba, X):
    camp_ctr_path = DATA_PATH_PREFIX + camp + '/train.ctr.txt'
    with open(camp_ctr_path, 'w') as fi:
        for cp, p, x in zip(click_price, proba, X):
            # print(cp, p, x)
            cp = list(map(int, cp))
            x = list(map(int, x))
            fi.write(' '.join(list(map(str, cp + [p] + x))) + '\n')
    print('Done {} ctr'.format(camp))


for camp in camps:
    X, Y, click_price = split_camp_data(camp)
    print('X, Y, click_price', X.shape, Y.shape, click_price.shape)
    proba = train_ctr_estimator(X, Y)
    write_ctr(camp, click_price, proba, X)

