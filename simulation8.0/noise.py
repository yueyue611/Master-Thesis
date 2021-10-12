import numpy as np
from scipy.stats import norm


# Ornstein-Uhlenbeck noise
class OUNoise:
    def __init__(self, processes, mean=0.5, sigma=0.3, theta=0.15, dt=0.1, x_initial=None):
        self.processes = processes  # action_dim
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = self.reset()

    def __call__(self):
        dw = norm.rvs(scale=self.dt, size=self.processes)
        dx = self.theta * (self.mean - self.x_prev) * self.dt + self.sigma * dw
        x = self.x_prev + dx
        # store x into x_prev, make next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.ones(self.processes) * self.mean
        return self.x_prev


class GaussianNoise:
    def __init__(self, processes, mean=0.5, sigma=0.3):
        self.processes = processes  # action_dim
        self.mean = mean
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mean, self.sigma, size=self.processes)
