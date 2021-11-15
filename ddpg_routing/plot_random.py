import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from config import Config


# parse string to float
def conv(s):
    try:
        s = float(s)
    except ValueError:
        pass
    return s


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


def main():
    experiment = Config.experiment
    total_episodes = Config.total_episodes
    total_steps = Config.total_steps
    noise_mode = Config.noise
    mode = Config.mode
    mode_select = Config.mode_select
    mode_flow_change = Config.mode_flow_change

    labels = ['TL=1.0']
    x_label = "traffic load (TL)"

    variance = 1

    m1 = [0 for _ in range(0, variance)]
    m2 = [0 for _ in range(0, variance)]
    m3 = [0 for _ in range(0, variance)]
    m4 = [0 for _ in range(0, variance)]

    median1 = [0 for _ in range(0, variance)]
    median2 = [0 for _ in range(0, variance)]
    median3 = [0 for _ in range(0, variance)]
    median4 = [0 for _ in range(0, variance)]

    st1 = [0 for _ in range(0, variance)]
    st2 = [0 for _ in range(0, variance)]
    st3 = [0 for _ in range(0, variance)]
    st4 = [0 for _ in range(0, variance)]

    for num in range(7, variance+7):
        reward_advance = [[] for _ in range(0, experiment)]
        reward_new = [[] for _ in range(0, experiment)]
        reward_old = [[] for _ in range(0, experiment)]
        reward_spf = [[] for _ in range(0, experiment)]

        for i in range(1, experiment + 1):
            with open(
                    "/home/tud/Github/Master-Thesis/ddpg_routing/csv/R{}{}/{}/CR_ No.{}, {}, {}, {}, {}, {}.csv".format(
                        num, num, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_advance:
                reader_advance = list(csv.reader(data_advance))
                for j in range(len(reader_advance)):
                    reward_advance[i - 1].append(conv(reader_advance[j][0]))

            with open(
                    "/home/tud/Github/Master-Thesis/ddpg_routing/csv/R{}{}new/{}/CR_ No.{}, {}, {}, {}, {}, {}.csv".format(
                        num, num, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_new:
                reader_new = list(csv.reader(data_new))
                for j in range(len(reader_new)):
                    reward_new[i - 1].append(conv(reader_new[j][0]))

            with open(
                    "/home/tud/Github/Master-Thesis/ddpg_routing/csv/R{}{}old/{}/CR_ No.{}, {}, {}, {}, {}, {}.csv".format(
                        num, num, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_old:
                reader_old = list(csv.reader(data_old))
                for j in range(len(reader_old)):
                    reward_old[i - 1].append(conv(reader_old[j][0]))

            with open(
                    "/home/tud/Github/Master-Thesis/ddpg_routing/csv/spf/spf, R, {}, {}.csv".format(num, num), 'r') as data_spf:
                reader_spf = list(csv.reader(data_spf))
                for j in range(len(reader_spf)):
                    reward_spf[i - 1].append(conv(reader_spf[j][0]))

        m1[num-7]= np.array(reward_advance).mean(axis=0)[0]
        median1[num-7] = np.median(reward_advance, axis=0)[0]

        m2[num-7] = np.array(reward_new).mean(axis=0)[0]
        median2[num-7] = np.median(reward_new, axis=0)[0]

        m3[num-7] = np.array(reward_old).mean(axis=0)[0]
        median3[num-7] = np.median(reward_old, axis=0)[0]

        m4[num-7] = np.array(reward_spf).mean(axis=0)[0]
        median4[num-7] = np.median(reward_spf, axis=0)[0]

    fig, ax = plt.subplots()
    x = np.arange(1)
    bar_width = 0.15

    ax.bar(x-bar_width, median1, bar_width, label="Advanced")
    ax.bar(x, median2, bar_width, label="New")
    ax.bar(x+bar_width, median3, bar_width, label="First")
    ax.bar(x+2*bar_width, median4, bar_width, label="OSPF")

    ax.set_xlabel('random case')
    ax.set_ylabel('median of avg. reward')
    ax.set_xticks(x + bar_width / 2)
    ax.set_xticklabels(('B'))
    ax.legend()
    fig.tight_layout()
    plt.savefig('randomB_median.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.bar(x-bar_width, m1, bar_width, label="Advance")
    ax2.bar(x, m2, bar_width, label="New")
    ax2.bar(x+bar_width, m3, bar_width, label="First")
    ax2.bar(x+2*bar_width, m4, bar_width, label="OSPF")

    ax2.set_xlabel('random case')
    ax2.set_ylabel('mean of avg. reward')
    ax2.set_xticks(x + bar_width / 2)
    ax2.set_xticklabels(('B'))
    ax2.legend()
    fig2.tight_layout()
    plt.savefig('randomB_mean.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
