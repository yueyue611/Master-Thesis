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
    r_delay = [[] for _ in range(0, experiment)]
    avg_delay = [[] for _ in range(0, experiment)]
    r_pkt = [[] for _ in range(0, experiment)]

    converged_ep = [[] for _ in range(0, experiment)]
    converged_reward = [[] for _ in range(0, experiment)]

    # !!!!!
    converged_ep2 = [[] for _ in range(0, experiment)]
    converged_reward2 = [[] for _ in range(0, experiment)]

    folder = "100, TL=1.0"
    # folder = "test"

    for i in range(1, experiment + 1):
        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_reward:
            reader_reward = list(csv.reader(data_reward))
            for j in range(len(reader_reward)):
                reward[i - 1].append([conv(s) for s in reader_reward[j]])

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/r_delay_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_r_delay:
            reader_r_delay = list(csv.reader(data_r_delay))
            for j in range(len(reader_r_delay)):
                r_delay[i - 1].append([conv(s) for s in reader_r_delay[j]])

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/avg_delay_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_avg_delay:
            reader_avg_delay = list(csv.reader(data_avg_delay))
            for j in range(len(reader_avg_delay)):
                avg_delay[i - 1].append([conv(s) for s in reader_avg_delay[j]])

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/r_pkt_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_r_pkt:
            reader_r_pkt = list(csv.reader(data_r_pkt))
            for j in range(len(reader_r_pkt)):
                r_pkt[i - 1].append([conv(s) for s in reader_r_pkt[j]])

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/CE_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_ce:
            reader_ce = list(csv.reader(data_ce))
            for j in range(len(reader_ce)):
                converged_ep[i - 1].append(conv(reader_ce[j][0]))

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/CR_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_cr:
            reader_cr = list(csv.reader(data_cr))
            for j in range(len(reader_cr)):
                converged_reward[i - 1].append(conv(reader_cr[j][0]))

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/CE2_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_ce2:
            reader_ce2 = list(csv.reader(data_ce2))
            for j in range(len(reader_ce2)):
                converged_ep2[i - 1].append(conv(reader_ce2[j][0]))

        with open(
                "/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/{}/CR2_ No.{}, {}, {}, {}, {}, {}.csv".format(
                    folder, noise_mode, i, total_episodes, total_steps, noise_mode, mode, mode_flow_change), 'r') as data_cr2:
            reader_cr2 = list(csv.reader(data_cr2))
            for j in range(len(reader_cr2)):
                converged_reward2[i - 1].append(conv(reader_cr2[j][0]))

    # calculate confidence interval for reward
    data_reward = np.array(reward).transpose((1, 0, 2))
    expect_reward = [[] for _ in range(0, mode_select)]
    low_bound_reward = [0 for _ in range(0, mode_select)]
    high_bound_reward = [0 for _ in range(0, mode_select)]

    # calculate confidence interval for r_delay
    data_r_delay = np.array(r_delay).transpose((1, 0, 2))
    expect_r_delay = [[] for _ in range(0, mode_select)]
    low_bound_r_delay = [0 for _ in range(0, mode_select)]
    high_bound_r_delay = [0 for _ in range(0, mode_select)]

    # calculate confidence interval for avg_delay
    data_avg_delay = np.array(avg_delay).transpose((1, 0, 2))
    expect_avg_delay = [[] for _ in range(0, mode_select)]
    low_bound_avg_delay = [0 for _ in range(0, mode_select)]
    high_bound_avg_delay = [0 for _ in range(0, mode_select)]

    # calculate confidence interval for r_pkt
    data_r_pkt = np.array(r_pkt).transpose((1, 0, 2))
    expect_r_pkt = [[] for _ in range(0, mode_select)]
    low_bound_r_pkt = [0 for _ in range(0, mode_select)]
    high_bound_r_pkt = [0 for _ in range(0, mode_select)]

    for i in range(0, mode_select):
        expect_reward[i].append(np.mean(data_reward[i], 0))
        low_bound_reward[i], high_bound_reward[i] = st.t.interval(
            0.95, len(data_reward[i]) - 1, loc=np.mean(data_reward[i], 0), scale=st.sem(data_reward[i]))

        expect_r_delay[i].append(np.mean(data_r_delay[i], 0))
        low_bound_r_delay[i], high_bound_r_delay[i] = st.t.interval(
            0.95, len(data_r_delay[i]) - 1, loc=np.mean(data_r_delay[i], 0), scale=st.sem(data_r_delay[i]))

        expect_avg_delay[i].append(np.mean(data_avg_delay[i], 0))
        low_bound_avg_delay[i], high_bound_avg_delay[i] = st.t.interval(
            0.95, len(data_avg_delay[i]) - 1, loc=np.mean(data_avg_delay[i], 0), scale=st.sem(data_avg_delay[i]))

        expect_r_pkt[i].append(np.mean(data_r_pkt[i], 0))
        low_bound_r_pkt[i], high_bound_r_pkt[i] = st.t.interval(
            0.95, len(data_r_pkt[i]) - 1, loc=np.mean(data_r_pkt[i], 0), scale=st.sem(data_r_pkt[i]))

    labels_1 = ['TL=0.2', 'TL=0.3', 'TL=0.5', 'TL=0.7', 'TL=1.0', 'TL=1.2']
    labels_2 = ['QLU=1', 'QLU=2', 'QLU=3']

    if mode == "TL":
        labels = labels_1
        x_label = "traffic load (TL)"
    else:
        labels = labels_2
        x_label = "queue length unit (QLU)"

    plt.figure()
    x = np.linspace(0, total_episodes - 1, num=total_episodes)
    for i in range(mode_select):
        plt.plot(x, expect_reward[i][0], linewidth=0.5, linestyle='-', markersize=2, label=labels[i])
        plt.fill_between(x, low_bound_reward[i], high_bound_reward[i], alpha=0.5)
    plt.legend()
    plt.xlabel("episode")
    plt.ylabel("avg. reward")
    plt.grid()
    # plt.savefig('Avg. Episodic Reward.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    for i in range(mode_select):
        plt.plot(x, expect_r_delay[i][0], linewidth=0.5, linestyle='-', markersize=2, label=labels[i])
        plt.fill_between(x, low_bound_r_delay[i], high_bound_r_delay[i], alpha=0.5)
    plt.legend()
    plt.xlabel("episode")
    plt.ylabel("avg. r_delay")
    plt.grid()
    # plt.savefig('Avg. Episodic r_delay.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    for i in range(mode_select):
        plt.plot(x, expect_avg_delay[i][0], linewidth=0.5, linestyle='-', markersize=2, label=labels[i])
        plt.fill_between(x, low_bound_avg_delay[i], high_bound_avg_delay[i], alpha=0.5)
    plt.legend()
    plt.xlabel("episode")
    plt.ylabel("avg. avg_delay")
    plt.grid()
    # plt.savefig('Avg. Episodic avg_delay.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure()
    for i in range(mode_select):
        plt.plot(x, expect_r_pkt[i][0], linewidth=0.5, linestyle='-', markersize=2, label=labels[i])
        plt.fill_between(x, low_bound_r_pkt[i], high_bound_r_pkt[i], alpha=0.5)
    plt.legend()
    plt.xlabel("episode")
    plt.ylabel("avg. r_pkt")
    plt.grid()
    # plt.savefig('Avg. Episodic r_pkt.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    m1 = np.array(converged_ep).mean(axis=0)
    median1 = np.median(converged_ep, axis=0)
    figure1, axes1 = plt.subplots()
    bp1 = axes1.boxplot(np.array(converged_ep), sym="o", vert=True, patch_artist=True, showmeans=True)
    axes1.set_xlabel(x_label)
    axes1.set_ylabel('number of episodes until convergence')
    for i, line in enumerate(bp1['medians']):
        x1, y1 = line.get_xydata()[1]
        text1 = 'M={:.2f}\n μ={:.2f}'.format(median1[i], m1[i])
        axes1.annotate(text1, xy=(x1, y1))
    # plt.savefig('Time until Convergence.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    m2 = np.array(converged_reward).mean(axis=0)
    # st2 = np.array(converged_reward).std(axis=0)
    median2 = np.median(converged_reward, axis=0)
    figure2, axes2 = plt.subplots()
    bp2 = axes2.boxplot(np.array(converged_reward), labels=labels, sym="o", vert=True, patch_artist=True, showmeans=True)
    axes2.set_xlabel(x_label)
    axes2.set_ylabel('avg. reward')
    for i, line in enumerate(bp2['medians']):
        x2, y2 = line.get_xydata()[1]
        # text2 = 'm={:.2f}\n μ={:.2f}\n σ={:.2f}'.format(median2[i], m2[i], st2[i])
        text2 = 'M={:.2f}\n μ={:.2f}'.format(median2[i], m2[i])
        axes2.annotate(text2, xy=(x2, y2))
    # plt.savefig('Avg. Reward after Convergence.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # !!!!
    m3 = np.array(converged_ep2).mean(axis=0)
    median3 = np.median(converged_ep2, axis=0)
    figure3, axes3 = plt.subplots()
    bp3 = axes3.boxplot(np.array(converged_ep2), labels=labels, sym="o", vert=True, patch_artist=True, showmeans=True)
    axes3.set_xlabel(x_label)
    axes3.set_ylabel('number of episodes until convergence')
    for i, line in enumerate(bp3['medians']):
        x3, y3 = line.get_xydata()[1]
        text3 = 'M={:.2f}\n μ={:.2f}'.format(median3[i], m3[i])
        axes3.annotate(text3, xy=(x3, y3))
    # plt.savefig('Time until Convergence.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    m4 = np.array(converged_reward2).mean(axis=0)
    median4 = np.median(converged_reward2, axis=0)
    figure4, axes4 = plt.subplots()
    bp4 = axes4.boxplot(np.array(converged_reward2), labels=labels, sym="o", vert=True, patch_artist=True, showmeans=True)
    axes4.set_xlabel(x_label)
    axes4.set_ylabel('avg. reward')
    for i, line in enumerate(bp4['medians']):
        x4, y4 = line.get_xydata()[1]
        text4 = 'M={:.2f}\n μ={:.2f}'.format(median4[i], m4[i])
        axes4.annotate(text4, xy=(x4, y4))
    # plt.savefig('Avg. Reward after Convergence.pdf', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
