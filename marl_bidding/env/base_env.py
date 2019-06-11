import math
import numpy as np
from gym.spaces import Box
from gym.utils import seeding
import logging
logger = logging.getLogger(__name__)


class RTBBaseEnv():
    def __init__(self, data_loader, max_bid_price=300, max_episode_steps=1000, reward_schedule=None, seed=0):
        self.done = False
        self.bid = None
        self.last_bid = None
        self.max_bid_price = max_bid_price
        if reward_schedule is not None:
            self.reward_schedule = reward_schedule
        else:
            self.reward_schedule = {'win': 1., 'click': 2., 'conversion': 5.,
                                    'lose': 0., 'otherwise': 0.}
        self.auction_num = 0
        self.auction_win = 0
        self.click_win = 0
        self.conversion_win = 0
        self.episodes = 1
        self.episode_steps = 0
        self.max_episode_steps = max_episode_steps
        self.data_loader = data_loader
        self._seed(seed)
        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

    def step(self, action, compete_mode):
        # assert self.action_space.contains(action)
        self.last_action = action
        bid_price = action[0]
        self.episode_steps += 1
        done = self._check_done()
        if action > 0.:
            self.auction_num += 1
        reward, price_paid = self._compute_reward(self.bid, bid_price, compete_mode)
        self.last_reward = reward
        self.last_bid = self.bid
        self.bid = self._get_next_bidding(compete_mode)
        # obs = self._convert_bid_to_obs(self.bid, price_paid)
        obs = self._convert_bid_to_obs(self.bid)
        return (obs, reward, done, {})

    def reset(self, compete_mode):
        self.episodes += 1
        self.episode_steps = 0
        self.data_loader.reset()
        self.bid = self._get_next_bidding(compete_mode)
        # obs = self._convert_bid_to_obs(self.bid, 0)
        obs = self._convert_bid_to_obs(self.bid)
        self.done = False
        return obs

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _action_space(self):
        """Returns a space object"""
        action_low = [0.0]
        action_high = [self.max_bid_price]
        bid_price = Box(np.array(action_low), np.array(action_high))
        return bid_price

    def _observation_space(self):
        """Returns a space object"""
        raise NotImplementedError

    def _convert_bid_to_obs(self, bid):
        raise NotImplementedError

    def _compute_reward(self, bid, bid_price, compete_mode):
        """Returns a computed scalar value based on the bidding result"""
        price_paid = 0.
        if bid_price >= bid['price_paid']:
            self.auction_win += 1
            price_paid = int(math.ceil(bid['price_paid']))
            reward = self.reward_schedule['win']
            if bid['is_click'] == 1.:
                self.click_win += 1
                reward = self.reward_schedule['click']
            if bid['is_conversion'] == 1.:
                self.conversion_win += 1
                reward = self.reward_schedule['conversion']
        else:
            reward = self.reward_schedule['lose']
        return reward, price_paid

    def _get_next_bidding(self, mode):
        """Get next bidding from data source"""
        return self.data_loader.get_next(mode)

    def _get_next_bidding_agent(self, mode, i):
        """Get next bidding from data source"""
        return self.data_loader[i].get_next(mode)

    def _check_done(self):
        """Returns true if meet the ending conditions"""
        if self.episode_steps >= self.max_episode_steps:
            return True
        else:
            return False
