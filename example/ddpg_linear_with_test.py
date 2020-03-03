import numpy as np
import time
import math
import argparse
import pprint as pp
import os
import sys
sys.path.insert(0, '../')

import master_config as config
from marl_bidding.data_loader import IPinyouDataLoader, SingleDataLoader
from marl_bidding.agent import LinearAgent, DDPGAgent
from marl_bidding.env import MultiAgentRTBBaseEnv

import os.path as osp
import _pickle as pickle
# import IPython

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = PROJECT_PATH + '/log'
MODEL_DIR = PROJECT_PATH + '/ddpg_models'

def main(args):

    om = args['om']
    random_om = args['random_om']
    reward_type = int(args['reward'])
    c0 = args['budget']

    seed = int(args['seed'])
    np.random.seed(seed=int(args['seed']))

    q_type = config.q_func
    agent_num = int(args['agent_num'])
    noise = float(args['noise'])

    compete_mode = config.compete_mode
    train_mode = args['train_mode']

    max_bid = 300.
    batch_size = 100
    start_steps = 1000
    # max_ep_len = 1000
    steps_per_epoch = 1000

    file_prefix = '_c0_{}_r_{}_q_{}_a_{}_s_{}_n_{}_comp_{}' \
        .format(c0, reward_type, q_type, agent_num, seed, noise, compete_mode)

    camp = args['camp']
    print('start processing %s' %camp)
    m_dim = int(max_bid) + 1

    market_file = []

    if train_mode:
        data_file = 'train.ctr.txt'
        if om:
            if not random_om:
                market_file.append('../predictions/agent_None/{}_{}'.format(camp, 'train') + file_prefix + '.txt')
                # data_file = 'train.ctr_mk' + file_prefix + '.txt'

    else:
        data_file = 'test.ctr.txt'
        if om:
            if not random_om:
                market_file.append('../predictions/agent_None/{}_{}'.format(camp, 'test') + file_prefix + '.txt')
                # data_file = 'test.ctr_mk' + file_prefix + '.txt'

    a_op = np.arange(0., max_bid+1, 1.) / max_bid

    bid_log_path = '../log/bid_logs/train_{}'.format(train_mode)
    if not osp.exists(bid_log_path):
        os.makedirs(bid_log_path)
    bid_log_file = bid_log_path + '/{}_om_{}_random_om_{}'.format(camp, om, random_om) + file_prefix + '.txt'

    bid_log = open(bid_log_file, 'w')
    bid_log_header = '{}\t'.format('ddpg_bid')
    bid_log.write(bid_log_header)
    for z in range(agent_num-1):
        bid_log.write('{}{}\t'.format('linear_bid_', z))
    bid_log.write('{}\t'.format('click'))
    bid_log.write('{}\n'.format('market'))

    if len(market_file) > 0:
        data_loader = IPinyouDataLoader(data_path='../data/make-ipinyou-data/', market_path=market_file[0],
                                        camp=camp, file=data_file, om=om, random_om=random_om)
    else:
        data_loader = IPinyouDataLoader(data_path='../data/make-ipinyou-data/', market_path=None,
                                        camp=camp, file=data_file, om=om, random_om=random_om)

    # data_loader = SingleDataLoader(data_path='../data/make-ipinyou-data/', camp=camp, file=data_file)
    data_loader.reset()
    data_loader.get_next(compete_mode)

    campaign_info = pickle.load(open('../data/make-ipinyou-data/' + camp + '/info.txt', 'rb'))
    imps_train = campaign_info['imp_train']
    imps_test = campaign_info['imp_test']

    if train_mode:
        epochs = math.ceil(imps_train / steps_per_epoch)
        # epochs = args['epochs']
    else:
        epochs = math.ceil(imps_test / steps_per_epoch)

    # budget per episode
    b_epi = float(campaign_info['cost_train'] / imps_train * c0 * steps_per_epoch)
    # b_epi = c0 * steps_per_epoch

    env = MultiAgentRTBBaseEnv(agent_num=agent_num,
                               data_loader=data_loader,
                               init_budget=b_epi,
                               market_price_dim=m_dim,
                               reward_schedule=reward_type,
                               max_bid_price=max_bid,
                               max_episode_steps=steps_per_epoch,
                               obs_noise_level=noise,
                               compete_mode=compete_mode,
                               om=om,
                               random_om=random_om,
                               seed=seed
                               )

    exp_name = 'ddpg_{}'.format(om)

    output_dir = LOG_DIR + '/experiments/{}/{}/{}/train_{}/random_om{}_c0_{}_q_{}_a_{}_s_{}_n_' \
                           '{}_comp_{}_{}'.format(exp_name, reward_type, camp, train_mode, int(random_om), str(c0),
                                                  q_type, agent_num, seed, noise, compete_mode, time.time())
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    logger_kwargs = {'output_dir': output_dir, 'exp_name': exp_name}

    # opponent action space (Normalized)

    model_path = MODEL_DIR + '/{}/{}/'.format(camp, c0)
    if not osp.exists(model_path):
        os.makedirs(model_path)

    model_name = model_path + 'model' + file_prefix + '_om_{}'.format(om) + '_random_om_{}'.format(random_om)

    ddpg_agent = DDPGAgent(env.observation_space, env.action_space, q_type, args['train_mode'], model_path, model_name,
                           batch_size, market_price_dim=m_dim, om=om, logger_kwargs=logger_kwargs, seed=seed)

    # todo use different lin_ratios
    # lin_agents = [LinearAgent(ratio=10. * (float(args['lin_ratio']) + i)) for i in range(agent_num - 1)]
    lin_agents = [LinearAgent(ratio=10. * (float(args['lin_ratio']) + i * 10)) for i in range(agent_num - 1)]
    # lin_agents = [LinearAgent(ratio=i) for i in range(agent_num - 1)]
    agents = [ddpg_agent] + lin_agents

    obses = env.reset(compete_mode)

    start_time = time.time()
    (obses, info), r, d, ep_ret, ep_len = env.reset(compete_mode), 0, False, np.array([0.] * agent_num), 0

    if train_mode:
        total_steps = min(steps_per_epoch * epochs, imps_train)
    else:
        total_steps = min(steps_per_epoch * epochs, imps_test)

    # Main loop: collect experience in env and update/log each epoch
    epoch = 0
    for t in range(int(total_steps)):
        if t % 1000 == 0:
            print(t)
        # Multi-Agent Sampler here
        if t > start_steps:
            actions = []
            for i, agent in enumerate(agents):
                action = agent.act(obses[i])
                actions.append(action)

        else:
            actions = np.array([env.action_space.sample()[0] for _ in range(agent_num)])

        # print('actions', actions)
        # Step the env
        obses2, r, d, info, price_paid, scaled_bid, log_click = env.step(actions, compete_mode)


        # record bid price from all the agents and the market price
        if type(scaled_bid[0]) is np.ndarray:
            ddpg_bid = scaled_bid[0][0]
        else:
            ddpg_bid = scaled_bid[0]
        bid_log.write('{}\t'.format(ddpg_bid))

        for i in range(agent_num-1):
            if type(scaled_bid[i+1]) is np.ndarray:
                lin_bid = scaled_bid[i+1][0]
            else:
                lin_bid = scaled_bid[i+1]
            bid_log.write('{}\t'.format(lin_bid))
        bid_log.write('{}\t'.format(log_click))
        bid_log.write('{}\n'.format(np.max(price_paid)))

        try:
            ep_ret += r
        except TypeError:
            print(r)
            print(price_paid)
            sys.exit()
        ep_len += 1

        for i, agent in enumerate(agents):
            if type(agent) is DDPGAgent:
                agent.logger.store(EpRet=ep_ret, EpLen=ep_len)

        # Ignore the 'done' signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        # d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        for i, agent in enumerate(agents):
            if type(agent) is DDPGAgent:
                agent.replay_buffer.store(obses[i], actions[i], r[i], obses2[i], d, info.get('x_market', None),
                                          info.get('x2_market', None), a_op)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
        obses = obses2

        if d == True:
            print('training step {}, reward {}'.format(t, ep_ret))
            if t > start_steps:
                for i, agent in enumerate(agents):
                    if type(agent) is DDPGAgent:
                        if train_mode:
                            agent.final_sess_save()

                        agent.logger.log_tabular('Epoch', epoch)
                        agent.logger.log_tabular('Initial-Budget-{}'.format(i),
                                                 '/'.join(map(str, env.agent_budget)))
                        agent.logger.log_tabular('Budget-Left-{}'.format(i),
                                                 '/'.join(map(str, env.agent_budget_left)))
                        agent.logger.log_tabular('Total0Auction', env.episode_steps)
                        agent.logger.log_tabular('Auction-Num-{}'.format(i), '/'.join(map(str, env.auction_num)))
                        agent.logger.log_tabular('Auction-Win-{}'.format(i), '/'.join(map(str, env.auction_win)))
                        agent.logger.log_tabular('Total-Click', env.click_num)
                        agent.logger.log_tabular('Click-Win-{}'.format(i), '/'.join(map(str, env.click_win)))
                        if train_mode:
                            agent.logger.log_tabular('LossQ', average_only=True)
                            agent.logger.log_tabular('QVals', average_only=True)
                            agent.logger.log_tabular('LossPi', average_only=True)
                            agent.logger.log_tabular('EpRet', with_min_and_max=True)
                        # agent.logger.log_tabular('EpLen', average_only=True)
                        agent.logger.log_tabular('Time', time.time() - start_time)
                        agent.logger.dump_tabular()
                        epoch += 1
            (obses, info), r, d, ep_ret, ep_len = env.reset(compete_mode), 0, False, np.array([0.] * agent_num), 0

        #if train_mode:
        if t >= start_steps:
            for agent in agents:
                    agent.train()
    bid_log.close()


if __name__ == '__main__':
    print('start running')

    parser = argparse.ArgumentParser(description='provide arguments for model training')
    parser.add_argument('--om', help='train ddpg with a survival model', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--train-mode', help='train ddpg with a survival model', default=True,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--budget', help='the proportion of the budget [0.5, 0.25, 0.625]', type=float, default=1/8)
    parser.add_argument('--camp', help='camp ID', default='2259')
    parser.add_argument('--epochs', help='limit the number of epochs in the test mode', default=400)
    parser.add_argument('--agent-num', help='number of agents in total', default=3)
    parser.add_argument('--random-om', help='use random market model or not', default=False,
                        type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--noise', help='noise level', default=0.001)
    parser.add_argument('--reward', help='1: click, 2: pctr, 3:mix', default=2)
    parser.add_argument('--lin-ratio', help='2259:50, 2997:10', default=10)
    parser.add_argument('--seed', help='random seed', default=0)

    args = vars(parser.parse_args())
    pp.pprint(args)

    main(args)
