import numpy as np
import copy
import random


# Ornstein-Uhlenbeck noise

class OUNoise:
    def __init__(self, size, mu=0., theta=0.15, sigma=0.05):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


#---------------------------------------------------------------------
