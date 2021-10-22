import numpy as np
import matplotlib.pyplot as plt
from noise import OUNoise, GaussianNoise


def main():
    processes = 3
    samples = 1000

    mean = 0

    x_ou = np.zeros(shape=(samples, processes))
    x_gaussian = np.zeros(shape=(samples, processes))

    ounoise = OUNoise(processes, mean)
    gaunoise = GaussianNoise(processes, mean)

    for t in range(1, samples):
        x_ou[t] = ounoise()
        x_gaussian[t] = gaunoise()

    plt.figure(1)
    plt.plot(x_ou)
    plt.xlabel("t")
    plt.ylabel("$x_{t}$")
    plt.title('Ornstein-Uhlenbeck Process')
    plt.savefig('ounoise.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(2)
    plt.plot(x_gaussian)
    plt.xlabel("t")
    plt.ylabel("$x_{t}$")
    plt.title('Gaussian Process')
    plt.savefig('gaunoise.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
