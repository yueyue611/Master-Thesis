from itertools import product
import pandas as pd

from env import Env
from config import Config

flows = Config.flows
nodes = Config.nodes
max_bw = Config.max_bw
max_link_lt = Config.max_link_lt
case = Config.case
action_mode = Config.action_mode
env = Env(flows, nodes, max_bw, max_link_lt, case, action_mode)

a_delay = Config.a_delay
a_pkt_loss = Config.a_pkt_loss

traffic_load = Config.traffic_load
flows_tl = env.get_flow_tl(traffic_load)

queue_length = Config.queue_length[0]

optimal_path = []

# Advance
"""
all_path = env.get_all_paths()
for i in product(*all_path):
    optimal_path.append(i)
"""

# new
"""
optimal_path = [([0, 4], [0, 4], [0, 3, 4], [0, 1, 2, 4], [2, 4, 0]), ([0, 4], [0, 3, 4], [0, 3, 4], [0, 1, 2, 4], [2, 4, 0]),
                ([0, 4], [0, 1, 2, 4], [0, 3, 4], [0, 1, 2, 4], [2, 4, 0]), ([0, 4], [0, 4], [0, 3, 4], [0, 1, 2, 4], [2, 1, 0]),
                ([0, 4], [0, 3, 4], [0, 3, 4], [0, 1, 2, 4], [2, 4, 0]), ([0, 4], [0, 1, 2, 4], [0, 3, 4], [0, 1, 2, 4], [2, 4, 0])]
"""

# SPF
spf_Flag = True
if spf_Flag:
    paths = []
    for i in range(0, flows):
        paths = env.get_opt_path(env.latency)
    optimal_path.append(paths)
    r_spf = []

r = [[] for _ in range(len(traffic_load))]

for j in range(0, len(traffic_load)):
    for k in range(0, len(optimal_path)):
        flow_traffic = [flows_tl[j][i][2] for i in range(len(flows_tl[j]))]
        state = env.get_state(optimal_path[k], flow_traffic)
        reward, r_delay, avg_delay, r_pkt_loss = env.get_reward(state, optimal_path[k], queue_length, a_delay, a_pkt_loss)
        r[j].append([reward, optimal_path[k]])

for j in range(0, len(traffic_load)):
    r[j].sort(key=lambda x: (x[:][0]), reverse=True)
    print(r[j][0])
    print("\n")
    if spf_Flag:
        r_spf.append(r[j][0][0])

if spf_Flag:
    print(r_spf)
    folder = "spf"
    index = 0
    df = pd.DataFrame(r_spf)
    df.to_csv("/home/tud/Github/Master-Thesis/ddpg_routing/csv/{}/spf, {}, {}.csv"
              .format(folder, index, flows), header=False, index=False)




