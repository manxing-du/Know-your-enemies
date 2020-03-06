import numpy as np
from .base import BaseAgent


class RandomAgent(BaseAgent):

    def act(self, obs):
        return np.random.rand()

    def train(self):
        pass
