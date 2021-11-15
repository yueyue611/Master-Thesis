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

    reward_advance = [[] for _ in range(0, experiment)]
    reward_new = [[] for _ in range(0, experiment)]
    reward_old = [[] for _ in range(0, experiment)]
    reward_spf = [[] for _ in range(0, experiment)]

    for i in range(1, experiment + 1):
        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/1000, QL=1/{}/CR_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_advance:
            reader_advance = list(csv.reader(data_advance))
            for j in range(len(reader_advance)):
                reward_advance[i - 1].append(conv(reader_advance[j][0]))

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/New, 100, QL=1/{}/CR_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_new:
            reader_new = list(csv.reader(data_new))
            for j in range(len(reader_new)):
                reward_new[i - 1].append(conv(reader_new[j][0]))

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/Old, 100, QL=1/{}/CR_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_old:
            reader_old = list(csv.reader(data_old))
            for j in range(len(reader_old)):
                reward_old[i - 1].append(conv(reader_old[j][0]))

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/spf/spf, 1000, 5.csv", 'r') as data_spf:
            reader_spf = list(csv.reader(data_spf))
            for j in range(len(reader_spf)):
                reward_spf[i - 1].append(conv(reader_spf[j][0]))

    labels_1 = ['TL=0.2', 'TL=0.3', 'TL=0.5', 'TL=0.7', 'TL=1.0', 'TL=1.2']
    labels_2 = ['QLU=1', 'QLU=2', 'QLU=3']

    if mode == "TL":
        labels = labels_1
        x_label = "traffic load (TL)"
    else:
        labels = labels_2
        x_label = "queue length unit (QLU)"

    m1 = np.array(reward_advance).mean(axis=0)
    median1 = np.median(reward_advance, axis=0)
    figure1, axes1 = plt.subplots()
    bp1 = axes1.boxplot(np.array(reward_advance), labels=labels, sym="o", vert=True, patch_artist=True, showmeans=True)
    axes1.set_xlabel(x_label)
    axes1.set_ylabel('avg. reward')
    for i, line in enumerate(bp1['medians']):
        x1, y1 = line.get_xydata()[1]
        text1 = 'M={:.2f}\n μ={:.2f}'.format(median1[i], m1[i])
        axes1.annotate(text1, xy=(x1, y1))
    plt.savefig('avg_reward_advance.png', dpi=300, bbox_inches='tight')
    plt.show()

    m2 = np.array(reward_new).mean(axis=0)
    # st2 = np.array(reward_new).std(axis=0)
    median2 = np.median(reward_new, axis=0)
    figure2, axes2 = plt.subplots()
    bp2 = axes2.boxplot(np.array(reward_new), labels=labels, sym="o", vert=True, patch_artist=True, showmeans=True)
    axes2.set_xlabel(x_label)
    axes2.set_ylabel('avg. reward')
    for i, line in enumerate(bp2['medians']):
        x2, y2 = line.get_xydata()[1]
        # text2 = 'm={:.2f}\n μ={:.2f}\n σ={:.2f}'.format(median2[i], m2[i], st2[i])
        text2 = 'M={:.2f}\n μ={:.2f}'.format(median2[i], m2[i])
        axes2.annotate(text2, xy=(x2, y2))
    plt.savefig('avg_reward_new.png', dpi=300, bbox_inches='tight')
    plt.show()

    # !!!!
    m3 = np.array(reward_old).mean(axis=0)
    median3 = np.median(reward_old, axis=0)
    figure3, axes3 = plt.subplots()
    bp3 = axes3.boxplot(np.array(reward_old), labels=labels, sym="o", vert=True, patch_artist=True, showmeans=True)
    axes2.set_xlabel(x_label)
    axes2.set_ylabel('avg. reward')
    for i, line in enumerate(bp3['medians']):
        x3, y3 = line.get_xydata()[1]
        text3 = 'M={:.2f}\n μ={:.2f}'.format(median3[i], m3[i])
        axes3.annotate(text3, xy=(x3, y3))
    plt.savefig('avg_reward_old.png', dpi=300, bbox_inches='tight')
    plt.show()

    m4 = np.array(reward_spf).mean(axis=0)
    median4 = np.median(reward_spf, axis=0)
    figure4, axes4 = plt.subplots()
    bp4 = axes4.boxplot(np.array(reward_spf), labels=labels, sym="o", vert=True, patch_artist=True, showmeans=True)
    axes4.set_xlabel(x_label)
    axes4.set_ylabel('avg. reward')
    for i, line in enumerate(bp4['medians']):
        x4, y4 = line.get_xydata()[1]
        text4 = 'M={:.2f}\n μ={:.2f}'.format(median4[i], m4[i])
        axes4.annotate(text4, xy=(x4, y4))
    plt.savefig('avg_reward_spf.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(median1, linewidth=1, linestyle='--', marker='o', markersize=4, label="Advance")
    #plt.plot(median2, linewidth=1, linestyle='--', marker='*', markersize=4, label="New")
    #plt.plot(median3, linewidth=1, linestyle='--', marker='^', markersize=4, label="First")
    plt.plot(median4, linewidth=1, linestyle='--', marker='+', markersize=4, label="OSPF")
    plt.xticks(range(0, len(labels)), labels)
    plt.legend()
    plt.xlabel("traffic load(TL)")
    plt.ylabel("avg. reward (median of 20 iterations)")
    plt.savefig('median1000.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    plt.plot(m1, linewidth=1, linestyle='--', marker='o', markersize=4, label="Advanced")
    #plt.plot(m2, linewidth=1, linestyle='--', marker='*', markersize=4, label="New")
    #plt.plot(m3, linewidth=1, linestyle='--', marker='^', markersize=4, label="First")
    plt.plot(m4, linewidth=1, linestyle='--', marker='+', markersize=4, label="OSPF")
    plt.xticks(range(0, len(labels)), labels)
    plt.legend()
    plt.xlabel("traffic load(TL)")
    plt.ylabel("avg. reward")
    plt.savefig('mean100.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
