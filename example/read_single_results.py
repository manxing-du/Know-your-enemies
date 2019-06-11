import glob
import csv

import sys
sys.path.append('../')
import master_config as config
import argparse
import numpy as np
import seaborn as sns
import pandas as pd

def main(args):

    # camps = config.camps
    c0 = args['budget']
    agent_num=int(args['agent_num'])
    r = args['reward']
    seed=args['seed']
    noise=float(args['noise'])
    # baseline_files = glob.glob('./cpm_budget_c0.5/*.tsv')
    random_om=int(config.random_om)
    q_type = config.q_func
    compete_mode = config.compete_mode

    baseline_files = []

    file_prefix = '{}_c0_{}_q_{}_a_{}_s_{}_n_{}_comp_{}' \
            .format(random_om, c0, q_type, agent_num, seed, noise, compete_mode)

    file_prefix2 = '{}_c0_{}_r_{}_q_{}_a_{}_s_{}_n_{}_comp_{}' \
            .format(random_om, c0, r, q_type, agent_num, seed, noise, compete_mode)

    # path_no_om = '../log/experiments/ddpg_False/' + str(r) + '/random_om0_c0_' + str(c0) + '_a_' + str(agent_num) + \
    #              '_s_' + str(seed) + '_n_' + str(noise) +'*'
    # path_om = '../log/experiments/ddpg_True/' + str(r) + '/random_om0_c0_' + str(c0) + '_a_' + str(agent_num) + \
    #           '_s_' + str(seed) + '_n_' + str(noise) + '*'
    # for camp in camps:

    camp = args['camp']

    path_no_om = '../log/experiments/ddpg_False/' + str(r) + '/' + camp + '/train_True' + '/random_om' + file_prefix + '*'
    path_om = '../log/experiments/ddpg_True/' + str(r) + '/' + camp + '/train_True' + '/random_om' + file_prefix + '*'

    # print(path_no_om)
    # print(path_om)

    baseline_files.append(sorted(glob.glob(path_no_om))[-1])
    baseline_files.append(sorted(glob.glob(path_om))[-1])

    exp_names = ['R_{}_O_F_C_{}_A_{}_S_{}_N_{}_COMP_{}_Q_{}'.format(r, c0, agent_num,seed,noise,compete_mode,q_type),
                 'R_{}_O_SUR_C_{}_A_{}_S_{}_N_{}_COMP_{}_Q_{}'.format(r, c0,agent_num,seed,noise,compete_mode,q_type)]

    log_file = open('../log/' + camp + '_summary_random_om_' + file_prefix2 + '_epi_all.txt', 'w')

    j = 0

    for file in baseline_files:

        file = file + ('/progress.txt')
        total_clks_ddpg = 0
        total_clks = 0
        total_win_ddpg = 0

        total_win_lins = np.zeros(agent_num - 1)
        total_clks_lins = np.zeros(agent_num - 1)

        lossQ = []
        QVals = []
        Losspi = []

        with open(file) as f:
            i = 0
            data_start = 1
            print(file)

            for line in f:
                if i >= data_start:
                    items = line.split('\t')
                    auc_win_ddpg = float(items[5].split('/')[0])

                    for z in range(0, len(items[5].split('/'))-1):
                        total_win_lins[z] += float(items[5].split('/')[z+1])

                    # for k in range((agent_num-1)):
                    #     auc_win_lin = float(items[5].split('/')[1+k])
                    #     clk_win_lin = float(items[7].split('/')[1+k])
                    #     total_clks_lin += clk_win_lin
                    #     total_auc_lin += auc_win_lin

                    clks = float(items[6])
                    clk_win_ddpg = float(items[7].split('/')[0])
                    for z in range(0, len(items[7].split('/'))-1):
                        total_clks_lins[z] += float(items[7].split('/')[z+1])

                    total_clks += clks
                    total_clks_ddpg += clk_win_ddpg
                    total_win_ddpg += auc_win_ddpg

                    i += 1

                    # lossQ.append(float(items[8]))
                    # QVals.append(float(items[9]))
                    # Losspi.append(float(items[10]))

                else:
                    i += 1
                    continue
            # exp_name = file.split(' ')[2]
            exp_name = exp_names[j]

            if j == 0:
                log = '{:55}\t{:>15}\t'.format('exp', 'total_clicks')
                log_file.write(log)

                for z in range(agent_num -1):
                    log_file.write('{:>15}\t'.format('clks_lin'))

                log = '{:>15}\t'.format('clks_ddpg')
                log_file.write(log)

                for z in range(agent_num - 1):
                    log_file.write('{:>15}\t'.format('wins_lin'))

                log = '{:>15}\n'.format('wins_ddpg')
                log_file.write(log)

            log = '{:55}\t{:>15}\t'.format(exp_name, total_clks)
            log_file.write(log)
            for z in range(agent_num - 1):
                log_file.write('{:>15}\t'.format(total_clks_lins[z]))

            log = '{:>15}\t'.format(total_clks_ddpg)
            log_file.write(log)

            for z in range(agent_num - 1):
                log_file.write('{:>15}\t'.format(total_win_lins[z]))

            log = '{:>15}\n'.format(total_win_ddpg)
            log_file.write(log)

            j += 1

    log_file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='provide arguments for the script')
    parser.add_argument('--budget', help='the proportion of the budget [0.5, 0.25, 0.625]', default=1/8)
    parser.add_argument('--camp', help='camp ID', default='2259')
    parser.add_argument('--agent-num', help='number of agents', default=3)
    parser.add_argument('--noise', help='noise level', default=0.001)
    parser.add_argument('--reward', help='reward', default=2)
    parser.add_argument('--seed', help='random seed', default=0)

    args = vars(parser.parse_args())
    main(args)

