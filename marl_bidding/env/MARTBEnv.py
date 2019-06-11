import numpy as np
import sys
from marl_bidding.env.base_env import RTBBaseEnv
import master_config as config
from gym.spaces import Box
from gym.utils import seeding
from copy import deepcopy

DEBUG = 1
# np.random.seed(seed=config.seed)

def print_progress(episodes, auctions, win_auctions, win_clicks, agent_budget_left):
    print("Episodes: {} | Win Auctions: {} | Clicks: {} | Budget Left: {}".format(
        episodes, win_auctions, win_clicks, agent_budget_left))

class MultiAgentRTBBaseEnv(RTBBaseEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, agent_num, data_loader, om=True, random_om=True, obs_noise_level=0., market_price_dim=10, init_budget=100000.,
                 reward_schedule=1, max_bid_price=300., max_episode_steps=100, episode_steps_ratio=0.,  compete_mode=1, seed=0):
        self.agent_num = agent_num
        self.done = False
        self.bid = None
        self.last_bid = None
        self.max_bid_price = max_bid_price
        self.market_price_dim = market_price_dim
        # todo update reward schedule
        self.reward_schedule = reward_schedule
        self.obs_noise_level = obs_noise_level
        self.episode_steps_ratio = episode_steps_ratio
        self.max_episode_steps = max_episode_steps
        self.data_loader = data_loader
        self.dataset_length = self.data_loader.get_dataset_length()
        if self.episode_steps_ratio > 0.:
            assert self.episode_steps_ratio < 1.
            self.max_episode_steps = int(np.ceil(self.dataset_length * self.episode_steps_ratio))

        self.om = om
        self.random_om = random_om

        # if reward_schedule is not None:
        #     self.reward_schedule = reward_schedule
        # elif reward_schedule == 1:
        #     self.reward_schedule = {'win': 0., 'click': 1., 'conversion': 1.,
        #                             'lose': 0., 'otherwise': 0.}
        # elif reward_schedule == 2:
        #     self.reward_schedule = {'win': 0., 'click': 'pctr', 'conversion': 1.,
        #                             'lose': 0., 'otherwise': 0.}
        # elif reward_schedule == 3:
        #     self.reward_schedule = {'win': '-cost', 'click': 1, 'conversion': 1.,
        #                             'lose': 0., 'otherwise': 0.}
        # else:
        #     self.reward_schedule = {'win': 0., 'click': 1., 'conversion': 1.,
        #                             'lose': 0., 'otherwise': 0.}

        self.auction_num = [0.] * self.agent_num
        self.auction_win = [0.] * self.agent_num
        self.click_num = 0.
        self.click_win = [0.] * self.agent_num
        self.conversion_win = [0.] * self.agent_num
        self.cost = [0.] * self.agent_num
        self.agent_budget = np.array([init_budget] * self.agent_num)
        self.agent_budget_left = deepcopy(self.agent_budget)

        self.episodes = 1
        self.episode_steps = 0
        self._seed(seed)
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()

        self.a_op = np.arange(0., max_bid_price + 1, 1.) / max_bid_price
        self.compete_mode = compete_mode

    def _get_market_p(self):
        random_dist = np.random.uniform(0., 1., self.market_price_dim)
        random_dist /= random_dist.sum()
        market_price = self.bid.get('market_price', random_dist)
        assert len(market_price) == self.market_price_dim, len(market_price)
        return market_price

    def step(self, actions, compete_mode):
        # assert self.action_space.contains(actions)
        self.last_action = actions
        bid_prices = actions
        self.episode_steps += 1
        done = self._check_done()
        rewards, price_paid, scaled_bid = self._compute_reward(self.bid, bid_prices, compete_mode)
        # scaled_bid = np.reshape(np.array(scaled_bid), (self.agent_num,))
        self.last_reward = rewards
        x_market = self._get_market_p()
        self.click_num += self.bid['click']
        self.last_bid = self.bid
        self.bid = self._get_next_bidding(compete_mode)
        x2_market = self._get_market_p()
        obs = self._convert_bid_to_obs(self.bid)

        return obs, rewards, done, {'x_market': x_market, 'x2_market': x2_market}, price_paid, scaled_bid, self.bid['click']

    def reset(self, compete_mode):
        print_progress(self.episodes, self.auction_num, self.auction_win, self.click_win, self.agent_budget_left)
        self.episodes += 1
        self.episode_steps = 0
        self.click_num = 0.
        self.click_win = [0.] * self.agent_num
        self.auction_num = [0.] * self.agent_num
        self.auction_win = [0.] * self.agent_num
        self.bid = self._get_next_bidding(compete_mode)
        x_market = self._get_market_p()
        self.agent_budget_left = deepcopy(self.agent_budget)
        obs = self._convert_bid_to_obs(self.bid)
        self.done = False
        return obs, {'x_market': x_market}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _action_space(self):
        """Returns a space object"""
        action_low = [0.]
        action_high = [1.]
        bid_price = Box(np.array(action_low), np.array(action_high))
        return bid_price

    def _observation_space(self):
        """Returns a space object"""
        obs_low = [0.] * 2
        obs_high = [1.] * 2
        pctrs = Box(np.array(obs_low), np.array(obs_high))
        return pctrs

    def _convert_bid_to_obs(self, bid):
        # obs_time = [1. - 1. * self.episode_steps / self.max_episode_steps ] * self.agent_num
        obs_budget = np.clip(self.agent_budget_left, 0, 100000000) / self.agent_budget
        pctrs = np.array([1.] * self.agent_num) * bid['pctr']
        # no noise for ddpg agent
        # pctrs[1:] = np.clip(pctrs[1:] * (1. + np.random.normal(size=self.agent_num-1) * self.obs_noise_level), 0., 1.)
        # add noise to all agents
        pctrs = np.clip(pctrs * (1. + np.random.normal(size=self.agent_num) * self.obs_noise_level), 0., 1.)
        # for i in range(len(pctrs)-1):
        #     pctrs[i+1] = np.clip(np.random.normal(0.4+i*0.4, 0.1, size=1), 0., 1.)
        obses = np.array(list(zip(pctrs, obs_budget)))
        obses = np.reshape(obses, (self.agent_num, 2))
        return obses
    
    def _validate_budget(self, bid_prices):
        for i, (bid_price, budget) in enumerate(zip(bid_prices, self.agent_budget_left)):
            if bid_price > budget:
                bid_prices[i] = budget
            else:
                self.auction_num[i] += 1
        return bid_prices

    def _compute_reward(self, bid, bid_prices, compete_mode):
        """Returns a computed scalar value based on the bidding result"""
        assert len(bid_prices) == self.agent_num

        paid_prices = [0.] * self.agent_num
        rewards = [0.] * self.agent_num

        bid_prices = np.array(bid_prices) * self.max_bid_price
        bid_prices = self._validate_budget(bid_prices)

        #print(bid_prices)

        index_max = np.argmax(bid_prices)

        highest_bid_price = bid_prices[index_max]

        if compete_mode == 1:
            market_price = bid['payprice']
            # second_bid_price = market_price
            if market_price > highest_bid_price:
                return np.array([0.] * self.agent_num), np.array([0.] * self.agent_num), bid_prices
            second_bid_price = np.partition(bid_prices, -2)[-2]
            paid_prices[index_max] = max(bid['payprice'], second_bid_price)

        else:
            # second_bid_price = market_price
            # if len(bid_prices) >= 2:
            try:
                second_bid_price = sorted(np.unique(bid_prices))[-2]
            except IndexError:
                second_bid_price = sorted(np.unique(bid_prices))[-1]
            if isinstance(second_bid_price, np.ndarray):
                second_bid_price = second_bid_price[0]
            paid_prices[index_max] = second_bid_price

        self.auction_win[index_max] += 1
        self.agent_budget_left = self.agent_budget_left - paid_prices
        # if self.agent_budget_left[index_max] < paid_prices[index_max]:
        #     index_max = 1 - index_max
        if self.reward_schedule == 1:
            if bid['click'] == 1.:
                self.click_win[index_max] += 1
                rewards[index_max] = 1.
        elif self.reward_schedule == 2:
            if bid['click'] == 1.:
                self.click_win[index_max] += 1
            rewards[index_max] = bid['pctr']
        elif self.reward_schedule == 3:
            if bid['click'] == 1.:
                self.click_win[index_max] += 1
                rewards[index_max] = 1.
            else:
                rewards[index_max] = bid['pctr']
        elif self.reward_schedule == 4:
            if bid['click'] == 1.:
                self.click_win[index_max] += 1
                rewards[index_max] = 400. - second_bid_price
            else:
                rewards[index_max] = bid['pctr'] * (400. - second_bid_price)

        rewards = np.reshape(rewards, (self.agent_num,))
        return rewards, paid_prices, bid_prices

    def _get_next_bidding(self, mode):
        """Get next bidding from data source"""
        return self.data_loader.get_next(mode)

    def _check_done(self):
        """Returns true if meet the ending conditions"""
        if self.episode_steps >= self.max_episode_steps or np.max(self.agent_budget_left) <= 10.:
            print('budget exhausted')
            return True
        else:
            return False

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = sys.stdout
        outfile.write('pCVR: {}, price paid: {}, is conversion: {},bid price: {}, reward: {}'.format(self.last_bid['estimated_cvr'],
            self.last_bid['price_paid'],
            self.last_bid['click'],
            self.last_action[0],
            self.last_reward))
        outfile.write('\n')
        if mode != 'human':
            return outfile
