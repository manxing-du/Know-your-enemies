import numpy as np
import numpy.random as nr

class OUNoise:
    """docstring for OUNoise"""
    def __init__(self,action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx
        return self.state * 0.01



    # # ================================
    # #    EXPONENTIAL NOISE DECAY
    # # ================================
    # @staticmethod
    # def exp_decay(noise, decay_end):
    #     num_steps = noise.shape[0]
    #     # Check if decay ends before end of noise sequence
    #     assert(decay_end <= num_steps)
    #
    #     scaling = np.zeros(num_steps)
    #
    #     scaling[:decay_end] = 2. - np.exp(np.divide(np.linspace(1., decay_end, num=decay_end) * np.log(2.), decay_end))
    #
    #     return np.multiply(noise, scaling)
    #
    # @staticmethod
    # def ou_noise(theta, mu, sigma, num_steps, dt=1.):
    #     noise = np.zeros(num_steps)
    #
    #     # Generate random noise with mean 0 and variance 1
    #     white_noise = np.random.normal(0, 1, num_steps)
    #
    #     # Solve using Euler-Maruyama method
    #     for i in xrange(1, num_steps):
    #         noise[i] = noise[i - 1] + theta * (mu - noise[i - 1]) * \
    #                                             dt + sigma * np.sqrt(dt) * white_noise[i]
    #
    #     return noise
