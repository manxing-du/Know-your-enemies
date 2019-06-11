from .base import BaseAgent
import numpy as np

class LinearAgent(BaseAgent):

    def __init__(self, ratio=1.):
        self.ratio = ratio

    def act(self, obs):
        # if len(obs) == 2:
        #     return np.clip(obs[0] * self.ratio, 0., 1.)
        # else:
        #     return np.clip(obs[:, 0] * self.ratio, 0., 1.)
        return np.clip(np.random.normal(0.1+self.ratio*0.1, 0.05, size=1), 0., 1.)

    def train(self):
        pass
