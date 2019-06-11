import pandas as pd
import argparse

from data_loaders import *
from opponent_model import *
import pprint as pp
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, '../')
import master_config as config

compete_mode = config.compete_mode


def main(args):

    camp = args['camp']
    multi = args['multi']
    agent_num = int(args['agent_num'])
    noise = args['noise']

    q_type = args['q_func']
    seed = int(args['seed'])
    np.random.seed(seed=int(args['seed']))
    op = args['op']

    if not multi:
        replay_path = args['replay_dir']
    else:
        replay_path = '../log/bid_logs/marl/train_True/'

    index_file = args['data_path'] + camp + '/' + args['index_file']
    feature_index = pd.read_csv(index_file, sep='\t', header=None, index_col=False)
    vocal_size = feature_index.iloc[-1, 1] + 1
    c0 = str(args['budget'])
    r = int(args['reward'])

    file_prefix = '_c0_{}_r_{}_q_{}_a_{}_s_{}_n_{}_comp_{}'\
        .format(c0, r, q_type, agent_num, seed, noise, compete_mode)

    if args['train']:

        infile = args['data_path'] + camp + '/' + args['train_file']
        # infile = args['data_path'] + camp + '/' + 'test.txt'

        df_train = pd.read_csv(infile, sep=' ', header=None, index_col=False)
        # clicks = df_train.iloc[:, 0]
        # market_price = df_train.iloc[:, 1]
        x = df_train.iloc[:, 2:]
        num_features = x.shape[1]

        if args['sample_size'] != 'all':
            sample_size = int(args['sample_size'])
        else:
            sample_size = x.shape[0]

        df_replay = pd.read_csv(replay_path + camp + '_om_False_random_om_False' +
                                file_prefix + '.txt', header=0, index_col=False, sep='\t', dtype=np.float)


        # df_replay = pd.read_csv(replay_path + 'test.txt', header=0, index_col=False, sep='\t')

        # To get training data (mixed)
        if not multi:
            x, c, b, m1, m2 = prepare_data(df_replay, x, sample_size, args['train'])
        else:
            x, c, b, m1, m2 = prepare_data_multi(df_replay, x, sample_size, args['train'], args['agent_index'])

        # split train / validation set
        indices = range(x.shape[0])
        x_train, x_val, c_train, c_val, idx_train, idx_val = train_test_split(x, c, indices,
                                                                              test_size=0.2, random_state=42)
        b_train = b[idx_train]
        m1_train = m1[idx_train]
        m2_train = m2[idx_train]
        b_val = b[idx_val]
        m1_val = m1[idx_val]
        m2_val = m2[idx_val]

        # model training parameters
        params = {'batch_size': args['batch_size'], 'shuffle': True}

        model_path = args['model_path'] + 'agent_' + str(args['agent_index']) + '/'

        op1 = opponent(args['alpha'], args['beta'], args['epochs'], args['max_market'], args['h1'],
                       args['h2'], vocal_size, model_path, camp, args['sample_size'], num_features,
                       args['emb_dropout'], args['lin_dropout'], c0, r, agent_num, args['agent_index'],
                       torch.cuda.is_available(), args['op'])

        # load training data as tensors
        trainingdata = logData(x_train, c_train, b_train, m1_train, m2_train)
        training_generator = data.DataLoader(trainingdata, **params)

        # load validation data
        valdata = logData(x_val, c_val, b_val, m1_val, m2_val)
        val_generator = data.DataLoader(valdata, **params)

        print('start training')
        train_loss, val_loss, val_anlp = op1.train(training_generator, val_generator, file_prefix, camp)

        figs = [train_loss, val_loss, val_anlp]
        # if multi:
        figure_path = args['figure_dir'] + 'train_loss/agent_' + str(args['agent_index']) + '/'
        # else:
        # figure_path = args['figure_dir'] + 'train_loss/single_agent/'

        if not osp.exists(figure_path):
            os.makedirs(figure_path)

        plot_figures(figs, figure_path, camp, c0, r, file_prefix)

    else:
        print('begin test')
        index_file_path = '../log/best_opp_model_index/agent_' + str(args['agent_index']) + '/'
        with open(index_file_path + camp + '_best_model' + file_prefix + '.txt', 'r') as infile:
            modelindex = infile.read()
        best_model_index = modelindex.splitlines()[0]

        # test on the training data
        # df_replay_test = pd.read_csv(replay_path + camp + '_om_False_random_om_False' + file_prefix + '.txt',
        # header=0, index_col=False, sep='\t')

        # Load test data
        infile = args['data_path'] + camp + '/' + args['test_file']
        df_test = pd.read_csv(infile, sep=' ', header=None, index_col=False)
        x_test = df_test.iloc[:, 2:]
        sample_size = x_test.shape[0]
        # sample_size = 100000
        num_features = x_test.shape[1]

        # Load test lables
        # test on the trainning set
        # x, _, _, m1, m2 = prepare_data(df_replay_test, x_test, sample_size, False)

        x = prepare_data_test(x_test, sample_size)
        params = {'batch_size': args['test_batchsize'],
                  'shuffle': False}

        model_path = args['model_path'] + 'agent_' + str(args['agent_index']) + '/'

        op1 = opponent(args['alpha'], args['beta'], args['epochs'], args['max_market'], args['h1'],
                       args['h2'], vocal_size, model_path, camp, args['sample_size'], num_features,
                       args['emb_dropout'], args['lin_dropout'], c0, r, agent_num, args['agent_index'],
                       torch.cuda.is_available(), args['op'])

        # testdata = logData_test(x, m1, m2)
        testdata = logData_test(x)
        test_generator = data.DataLoader(testdata, **params)
        # anlp = op1.test(test_generator, best_model_index, c0, file_prefix, mode='train')
        mode = args['test_file'].split('.')[0]
        op1.test(test_generator, best_model_index, c0, file_prefix, mode=mode)
        # print('anlp in the test set is %.4f' % anlp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for model training')

    # Read train / test files
    parser.add_argument('--data-path', help='directory of training files', default='../data/make-ipinyou-data/')
    parser.add_argument('--train-file', help='name of training files', default='train.yzx.txt')
    parser.add_argument('--test-file', help='name of test files', default='test.yzx.txt')
    # parser.add_argument('--replay-dir', help='directory of replay files', default='../replay_logs/lin/')
    parser.add_argument('--replay-dir', help='directory of replay files', default='../log/bid_logs/train_True/')
    parser.add_argument('--budget', help='the proportion of the budget [0.5, 0.25, 0.625]', default=1/2)
    parser.add_argument('--index-file', help='name of the feature index file', default='featindex.txt')
    parser.add_argument('--model-path', help='directory to save models', default='../opp_models/')
    parser.add_argument('--figure-dir', help='directory to save figures', default='../figures/')
    parser.add_argument('--camp', help='camp ID', default='2259')


    # Training parameters
    parser.add_argument('--sample-size', help='training sample size [10000, 1000000, all]', default='all')
    parser.add_argument('--batch-size', help='batch size', default=128)
    parser.add_argument('--epochs', help='number of epochs', default=10)
    parser.add_argument('--test-batchsize', help='test batch size', default=128)

    # Initialize opponent training parameters
    parser.add_argument('--alpha', help='learning rate', default=1e-5)
    parser.add_argument('--beta', help='parameter to balance the censored and uncensored loss', default=0.25)
    parser.add_argument('--max-market', help='maximum market price', default=301)
    parser.add_argument('--h1', help='# of hidden units in the 1st hidden layer', default=100)
    parser.add_argument('--h2', help='# of hidden units in the 1st hidden layer', default=50)
    parser.add_argument('--emb-dropout', help='embedding layer dropout', default=0.1)
    parser.add_argument('--lin-dropout', help='linear layer dropout', default=0.1)

    # Training market model for multi agent ddpg
    parser.add_argument('--agent-index', help='agent index, get in the bash script, either [0-N] numbers or single', default=0)

    parser.add_argument('--multi', help='multiple agents or single', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--agent-num', help='number of agents in total', default=3)

    parser.add_argument('--noise', help='noise level', default=0.001)

    # Indicating train/test
    parser.add_argument('--train', help='train / test the model', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--reward', help='1: click, 2: pctr, 3:mix', default=2)
    parser.add_argument('--seed', help='random seed', default=0)
    parser.add_argument('--q-func', help='indi, concat', default='concat')
    parser.add_argument('--op', help='ffn, tf', default='ffn')

    args = vars(parser.parse_args())
    pp.pprint(args)
    main(args)
