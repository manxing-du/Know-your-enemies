import numpy as np

class NNoise:
    """docstring for Normal_Decay_Noise"""
    def __init__(self,action_dimension, mu=0, decy=1.0-(4e-5), sigma=0.3):
        self.action_dimension = action_dimension
        self.mu = mu
        self.decy = decy
        self.sigma = sigma
        self.state = np.ones(self.action_dimension)
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension)

    def noise(self):
        self.state = self.state * self.decy
        dx = (self.mu) + self.sigma * np.random.randn(self.action_dimension)
        return self.state * dx