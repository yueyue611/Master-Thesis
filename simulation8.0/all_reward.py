from itertools import product

from env import Env
from config import Config

flows = Config.flows
nodes = Config.nodes
max_bw = Config.max_bw
max_link_lt = Config.max_link_lt
env = Env(flows, nodes, max_bw, max_link_lt)

a_delay = Config.a_delay
a_pkt_loss = Config.a_pkt_loss

traffic_load = Config.traffic_load
flows_tl = env.get_flow_tl(traffic_load)

queue_length = Config.queue_length[2]

optimal_path = []
all_path = env.get_all_paths()

for i in product(*all_path):
    optimal_path.append(i)

print(optimal_path)

r = [[] for _ in range(len(traffic_load))]

for j in range(0, len(traffic_load)):
    for k in range(0, len(optimal_path)):
        flow_traffic = [flows_tl[j][i][2] for i in range(len(flows_tl[j]))]
        state = env.get_state(optimal_path[k], flow_traffic)
        reward, r_delay, avg_delay, r_pkt_loss = env.get_reward(state, optimal_path[k], queue_length, a_delay, a_pkt_loss)
        r[j].append([reward, optimal_path[k]])

for j in range(0, len(traffic_load)):
    r[j].sort(key=lambda x: (x[:][0]), reverse=True)
    print(r[j])

