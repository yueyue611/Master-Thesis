import numpy as np
import copy
import math
from collections import defaultdict


class Graph:
    def __init__(self, nodes):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        # number of vertices
        self.vertices = nodes
        self.edges = defaultdict(list)

        # initializes adjacency matrix of latency and bandwidth
        self.adj_mat_latency = [[np.Inf for _ in range(self.vertices)] for _ in range(self.vertices)]
        self.adj_mat_bandwidth = [[0 for _ in range(self.vertices)] for _ in range(self.vertices)]

    def add_edge(self, u, v, lt, bw):
        # Note: assumes edges are unidirectional
        self.edges[u].append(v)
        self.adj_mat_latency[u][v] = lt
        self.adj_mat_bandwidth[u][v] = bw

    # gets adjacency matrix of latency and bandwidth
    def adj_mat(self):
        return self.adj_mat_latency, self.adj_mat_bandwidth

    def dijkstra(self, source, end, weights):
        # shortest paths is a dict of nodes whose value is a tuple of (previous node, weight)
        shortest_paths = {source: (None, 0)}
        current_node = source
        visited = set()

        while current_node != end:
            visited.add(current_node)
            destinations = self.edges[current_node]
            weight_to_current_node = shortest_paths[current_node][1]

            for next_node in destinations:
                weight = weights[current_node][next_node] + weight_to_current_node
                if next_node not in shortest_paths:
                    shortest_paths[next_node] = (current_node, weight)
                else:
                    current_shortest_weight = shortest_paths[next_node][1]
                    if weight < current_shortest_weight:
                        shortest_paths[next_node] = (current_node, weight)

            next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
            if not next_destinations:
                return "Route Not Possible"

            # next node is the destination with the lowest weight
            current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

        # work back through destinations in shortest path
        path = []
        while current_node is not None:
            path.append(current_node)
            next_node = shortest_paths[current_node][0]
            current_node = next_node
        # reverse path
        path = path[::-1]
        return path


def get_opt_path(flows, weights):
    opt_path = []
    for i in range(len(flows)):
        opt_path.append(graph.dijkstra(flows[i][0], flows[i][1], weights))
    return opt_path


def get_state(total_switches, optimal_path, flow_traffic):
    state = np.zeros((total_switches, total_switches))
    for i in range(len(optimal_path)):
        for j in range(len(optimal_path[i]) - 1):
            state[optimal_path[i][j]][optimal_path[i][j + 1]] += flow_traffic[i]
    return state


# calculate propagation latency of each path
def get_path_latency_pro(optimal_path, latency):
    path_latency_pro = [0 for _ in range(len(optimal_path))]
    for i in range(len(optimal_path)):
        for j in range(len(optimal_path[i]) - 1):
            path_latency_pro[i] += latency[optimal_path[i][j]][optimal_path[i][j+1]]
    return path_latency_pro


# calculate the approximate queue latency of one link
def get_path_latency_que(link_bandwidth):
    path_latency_que = 1512 * 8 * 30 / (link_bandwidth * 1000)
    return path_latency_que


# calculates reward
def getting_reward(optimal_path, bandwidth):
    r = 0
    cost = [0 for _ in range(len(optimal_path))]
    path_latency_que = [0 for _ in range(len(optimal_path))]
    for path in range(len(optimal_path)):
        for i in range(len(optimal_path[path]) - 1):
            sum_bandwidth = state[optimal_path[path][i]][optimal_path[path][i + 1]] + state[optimal_path[path][i + 1]][optimal_path[path][i]]
            # if there is a congestion of one link
            if sum_bandwidth > bandwidth[optimal_path[path][i]][optimal_path[path][i + 1]]:
                path_latency_que[path] += get_path_latency_que(bandwidth[optimal_path[path][i]][optimal_path[path][i + 1]])
        cost[path] += (path_latency_pro[path] + path_latency_que[path]) ** 2
    print("path latency queue:", path_latency_que,
          "\ncost:", cost)
    # calculates reward as minus root mean square of the sum of all latencies
    r = -math.sqrt(sum(cost) / len(optimal_path))
    return r


# number of switches
total_switches = 6

# maximum bandwidth
max_bandwidth = 10

# link latency
latency_link = 10

# creates a graph for the above topology
graph = Graph(total_switches)

# adds information to graph edges: (source, destination, latency, bandwidth)
edges = [
    (0, 1, 2 * latency_link, max_bandwidth),
    (0, 2, latency_link, max_bandwidth),
    (0, 3, latency_link, max_bandwidth),
    (1, 5, latency_link, max_bandwidth),
    (2, 4, 3 * latency_link, max_bandwidth),
    (3, 4, latency_link, max_bandwidth),
    (3, 5, latency_link, max_bandwidth),
    (4, 2, latency_link, max_bandwidth),
    (4, 3, 2 * latency_link, max_bandwidth),
    (4, 5, latency_link, max_bandwidth),
    (5, 0, latency_link, max_bandwidth),
    (5, 2, 3 * latency_link, max_bandwidth),
    (5, 3, latency_link, max_bandwidth),
    (5, 4, latency_link, max_bandwidth)
]

for edge in edges:
    graph.add_edge(*edge)

# two adjacency matrices with the value of latency and bandwidth
latency, bandwidth = graph.adj_mat()
print("latency matrix:", latency,
      "\nbandwidth matrix:", bandwidth)

# number of flows
total_flows = 4

# flows[i] = [source, destination, traffic]
flows = [[0 for _ in range(3)] for _ in range(total_flows)]
for i in range(total_flows):
    node_i = np.random.randint(0, total_switches)
    node_j = np.random.randint(0, total_switches)
    while node_i == node_j:
        node_i = np.random.randint(0, total_switches)
        node_j = np.random.randint(0, total_switches)
    flows[i][0] = node_i
    flows[i][1] = node_j
    flows[i][2] = np.random.randint(1, max_bandwidth)
print("flows: ", flows)

# traffic load
traffic_load = [0.4, 0.6, 0.8, 1.0, 1.2]

# new flows according to different traffic load
flows_tl = [copy.deepcopy(flows) for _ in range(len(traffic_load))]
for i in range(len(traffic_load)):
    for j in range(total_flows):
        flows_tl[i][j][2] *= traffic_load[i]
print("flows_tl: ", flows_tl)

# optimal path for each flow
weights = latency
optimal_path = get_opt_path(flows, weights)
print("path:", optimal_path)

# !!!!!!!!!!!!! need modify
# here traffic load equal to 1
flow_traffic = [flows_tl[3][i][2] for i in range(len(flows_tl[3]))]
print("flow traffic:", flow_traffic)
# get one state
state = get_state(total_switches, optimal_path, flow_traffic)
print("state:", state)

# calculate propagation latency
path_latency_pro = get_path_latency_pro(optimal_path, latency)
print("propagation latency of each path: ", path_latency_pro)

# calculates reward
r = getting_reward(optimal_path, bandwidth)
print("reward r: ", r)

