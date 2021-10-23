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

    reward = [[] for _ in range(0, experiment)]
    converged_reward = [[] for _ in range(0, experiment)]

    for i in range(1, experiment + 1):
        with open(
<<<<<<< HEAD
                "/home/gaoyueyue/Github/Master-Thesis/simulation8.0/csv/tanh/{}/No.{}, {}, {}, {}, {}, {}.csv".format(
                    noise_mode, i+215, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_reward:
=======
                "/home/yueyue/Github/Master-Thesis/simulation8.0/csv/tanh/{}/No.{}, {}, {}, {}, {}, {}.csv".format(
                    noise_mode, i+600, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_reward:
>>>>>>> 40bc30fdb963b896b3dbc625c46c2514cd647f3d
            reader_reward = list(csv.reader(data_reward))
            for j in range(len(reader_reward)):
                reward[i - 1].append([conv(s) for s in reader_reward[j]])

        with open(
<<<<<<< HEAD
                "/home/gaoyueyue/Github/Master-Thesis/simulation8.0/csv/tanh/{}/CR: No.{}, {}, {}, {}, {}, {}.csv".format(
                    noise_mode, i+215, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_cr:
=======
                "/home/yueyue/Github/Master-Thesis/simulation8.0/csv/tanh/{}/CR: No.{}, {}, {}, {}, {}, {}.csv".format(
                    noise_mode, i+600, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_cr:
>>>>>>> 40bc30fdb963b896b3dbc625c46c2514cd647f3d
            reader_cr = list(csv.reader(data_cr))
            for j in range(len(reader_cr)):
                converged_reward[i - 1].append(conv(reader_cr[j][0]))

    # calculate confidence interval for reward
    data_reward = np.array(reward).transpose((1, 0, 2))
    expect_reward = [[] for _ in range(0, mode_select)]
    low_bound_reward = [0 for _ in range(0, mode_select)]
    high_bound_reward = [0 for _ in range(0, mode_select)]

    for i in range(0, mode_select):
        expect_reward[i].append(np.mean(data_reward[i], 0))
        low_bound_reward[i], high_bound_reward[i] = st.t.interval(
            0.95, len(data_reward[i]) - 1, loc=np.mean(data_reward[i], 0), scale=st.sem(data_reward[i]))
        print(expect_reward[i][-1][-1])

    colors = ['r', 'y', 'g', 'c', 'b', 'm', 'gray', 'orange', 'purple', 'pink']
    colors2 = ['gray', 'orange', 'purple', 'pink', 'r', 'y', 'g', 'c', 'b', 'm']
    labels_1 = ['TL=0.2', 'TL=0.3', 'TL=0.5', 'TL=0.7', 'TL=1.0', 'TL=1.2']
    labels_2 = ['QL=1', 'QL=2', 'QL=3']
    labels_1r = ['Random TL=0.2', 'Random TL=0.3', 'Random TL=0.5', 'Random TL=0.7', 'Random TL=1.0', 'Random TL=1.2']
    labels_2r = ['Random QL=1', 'Random QL=2', 'Random QL=3']

    if mode == "TL":
        labels = labels_1
        labelsr = labels_1r
    else:
        labels = labels_2
        labelsr = labels_2r

    plt.figure(figsize=(16, 9))
    x = np.linspace(0, total_episodes - 1, num=total_episodes)
    for i in range(mode_select):
        plt.plot(x, expect_reward[i][0], linewidth=0.5, linestyle='-', markersize=2, label=labels[i])
        plt.fill_between(x, low_bound_reward[i], high_bound_reward[i], alpha=0.5)
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    plt.grid()
    # plt.savefig('Avg. Episodic Reward.png', dpi=300, bbox_inches='tight')
    plt.show()

    figure1, axes1 = plt.subplots()
    axes1.boxplot(np.array(converged_reward), labels=labels, sym="o", vert=True, patch_artist=True)
    axes1.set_ylabel('Avg. Reward after Convergence')
    # plt.savefig('Avg. Reward after Convergence.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
