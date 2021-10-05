import csv
import matplotlib.pyplot as plt
import numpy as np
from config import Config


# parse string to float
def conv(s):
    try:
        s = float(s)
    except ValueError:
        pass
    return s


def main():
    experiment_num = 2

    total_episodes = Config.total_episodes
    total_steps = Config.total_steps
    mode = Config.mode
    mode_select = Config.mode_select
    mode_flow_change = Config.mode_flow_change

    reward = []

    with open(
            "/home/tud/Github/Master-Thesis/simulation8.0/csv/No.{}, {}, {}, {}, {}.csv".format(
                1, total_episodes, total_steps, mode, mode_flow_change), 'r') as data_reward:
        reader_reward = list(csv.reader(data_reward))
        for i in range(len(reader_reward)):
            reward.append([conv(s) for s in reader_reward[i]])

    converged_reward = [[] for _ in range(0, experiment_num)]
    for i in range(1, experiment_num + 1):
        with open(
                "/home/tud/Github/Master-Thesis/simulation8.0/csv/CR: No.{}, {}, {}, {}, {}.csv".format(
                    i, total_episodes, total_steps, mode, mode_flow_change), 'r') as data_cr:
            reader_cr = list(csv.reader(data_cr))
            print(reader_cr)
            for j in range(len(reader_cr)):
                converged_reward[i - 1].append(conv(reader_cr[j][0]))

    print(converged_reward)

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

    plt.figure(1)
    for i in range(mode_select):
        plt.plot(reward[i], color=colors[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    # plt.savefig('Avg. Episodic Reward.png', dpi=300, bbox_inches='tight')
    plt.show()

    figure1, axes1 = plt.subplots()
    axes1.boxplot(np.array(converged_reward), labels=labels_1, sym="o", vert=True, patch_artist=True)  # labels_1 or labels_2
    axes1.set_xlabel('Traffic Load')
    axes1.set_ylabel('Avg. Reward till Convergence')
    plt.show()


if __name__ == "__main__":
    main()
