import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def main():
    dt = 0.1

    theta = 0.15  # friction: strength to pull towards the mean
    sigma = 0.3  # noise
    mu = 0.0  # global mean

    processes = 3
    samples = 1000

    X = np.zeros(shape=(samples, processes))
    for t in range(1, samples - 1):
        dw = norm.rvs(scale=dt, size=processes)  # W: Wierner process, dw: brownian velocity
        dx = theta * (mu - X[t]) * dt + sigma * dw
        X[t + 1] = X[t] + dx

    print(X)

    plt.figure()
    plt.plot(X)
    plt.title('Ornstein-Uhlenbeck Process')
    plt.legend(['A', 'B', 'C'])
    plt.show()


if __name__ == "__main__":
    main()
