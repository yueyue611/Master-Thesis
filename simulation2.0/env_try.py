import numpy as np
import copy
import math
from collections import defaultdict


class Graph:
    def __init__(self, nodes):
        # number of nodes
        self.vertices = nodes
        # a dict of all possible next nodes
        self.edges = defaultdict(list)

        # initialize adjacency matrix of weights, latency and bandwidth
        self.adj_mat_weights = [[0 for _ in range(self.vertices)] for _ in range(self.vertices)]
        self.adj_mat_latency = [[np.Inf for _ in range(self.vertices)] for _ in range(self.vertices)]
        self.adj_mat_bandwidth = [[0 for _ in range(self.vertices)] for _ in range(self.vertices)]

    def add_edge(self, u, v, lt, bw):
        # assume edges are unidirectional
        self.edges[u].append(v)
        self.adj_mat_weights[u][v] = 1
        self.adj_mat_latency[u][v] = lt
        self.adj_mat_bandwidth[u][v] = bw

    # get adjacency matrix of latency and bandwidth
    def adj_mat(self):
        return self.adj_mat_weights, self.adj_mat_latency, self.adj_mat_bandwidth

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


class Env:
    def __init__(self, flows, nodes, max_bw, max_link_lt):
        # number of switches
        self.total_switches = nodes
        # maximum bandwidth
        self.max_bandwidth = max_bw
        # max link latency
        self.max_link_latency = max_link_lt

        # creates a graph for the above topology
        self.graph = Graph(self.total_switches)

        # adds information to graph edges: (source, destination, latency, bandwidth)
        self.edges = [
            (0, 1, 1.0 * self.max_link_latency, self.max_bandwidth),
            (0, 2, 1.0 * self.max_link_latency, self.max_bandwidth),
            (0, 3, 1.0 * self.max_link_latency, self.max_bandwidth),
            (0, 5, 1.0 * self.max_link_latency, self.max_bandwidth),
            (1, 5, 1.0 * self.max_link_latency, self.max_bandwidth),
            (2, 4, 1.0 * self.max_link_latency, self.max_bandwidth),
            (3, 4, 1.0 * self.max_link_latency, self.max_bandwidth),
            (3, 5, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 2, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 3, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 5, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 1, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 3, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 4, 1.0 * self.max_link_latency, self.max_bandwidth)
        ]

        for edge in self.edges:
            self.graph.add_edge(*edge)

        # number of flows
        self.total_flows = flows

        # flows[i] = [source, destination, traffic]
        self.flows = [[0, 5, 6], [0, 5, 6], [0, 4, 6], [5, 0, 6], [3, 1, 6]]
        '''
        self.flows = [[0 for _ in range(3)] for _ in range(self.total_flows)]
        for i in range(self.total_flows):
            node_i = np.random.randint(0, self.total_switches)
            node_j = np.random.randint(0, self.total_switches)
            while node_i == node_j:
                node_i = np.random.randint(0, self.total_switches)
                node_j = np.random.randint(0, self.total_switches)
            self.flows[i][0] = node_i
            self.flows[i][1] = node_j
            self.flows[i][2] = np.random.randint(1, self.max_bandwidth)  # traffic can not be 0
        '''
        print("flows: ", self.flows)

        # adjacency matrices with the value of 1/0, latency and bandwidth
        self.weights, self.latency, self.bandwidth = self.graph.adj_mat()
        print("\nweights: ", self.weights,
              "\nlatency: ", self.latency,
              "\nbandwidth: ", self.bandwidth)

    def get_weights(self):
        return self.weights

    def observation_space(self):
        state_dim = self.total_switches ** 2
        action_dim = self.total_switches ** 2
        return state_dim, action_dim

    def get_flow_rl(self, traffic_load):
        # new flows according to different traffic load
        flows_tl = [copy.deepcopy(self.flows) for _ in range(len(traffic_load))]
        for i in range(len(traffic_load)):
            for j in range(self.total_flows):
                flows_tl[i][j][2] *= traffic_load[i]
        print("flows_tl: ", flows_tl)
        return flows_tl

    def get_opt_path(self, weights):
        opt_path = []
        for i in range(len(self.flows)):
            opt_path.append(self.graph.dijkstra(self.flows[i][0], self.flows[i][1], weights))
        return opt_path

    def get_state(self, weights, flow_traffic):
        optimal_path = self.get_opt_path(weights)
        state = np.zeros((self.total_switches, self.total_switches))
        for i in range(len(optimal_path)):
            for j in range(len(optimal_path[i]) - 1):
                state[optimal_path[i][j]][optimal_path[i][j + 1]] += flow_traffic[i]
        return state

    def reset(self, weights, flow_traffic):
        prev_state = self.get_state(weights, flow_traffic)
        return prev_state

    def step(self, action, flow_traffic):
        action_w = np.array(action).reshape((self.total_switches, self.total_switches))
        state = self.get_state(action_w, flow_traffic)
        reward = self.get_reward(state, action_w)
        return state, reward

    # calculates reward
    def get_reward(self, state, action):
        optimal_path = self.get_opt_path(action)
        path_latency_pro = self.get_path_latency_pro(optimal_path)
        r = 0
        cost = [0 for _ in range(len(optimal_path))]
        path_latency_que = [0 for _ in range(len(optimal_path))]
        for path in range(len(optimal_path)):
            for i in range(len(optimal_path[path]) - 1):
                sum_bandwidth = state[optimal_path[path][i]][optimal_path[path][i + 1]] + \
                                state[optimal_path[path][i + 1]][optimal_path[path][i]]
                # if there is a congestion of one link, calculate the approximate queue latency of one link
                if sum_bandwidth > self.bandwidth[optimal_path[path][i]][optimal_path[path][i + 1]]:
                    path_latency_que[path] += 1512 * 8 * 30 / \
                                              (self.bandwidth[optimal_path[path][i]][optimal_path[path][i + 1]] * 1000)
            cost[path] += (path_latency_pro[path] + path_latency_que[path]) ** 2
        print("optimal path: ", optimal_path,
              "\npath latency propagation: ", path_latency_pro,
              "\npath latency queue: ", path_latency_que,
              "\ncost:", cost)

        # calculates reward as minus root mean square of the sum of all latencies
        r = -math.sqrt(sum(cost) / len(optimal_path))
        print("reward: ", r)
        return r

    # calculate propagation latency of each path
    def get_path_latency_pro(self, optimal_path):
        path_latency_pro = [0 for _ in range(len(optimal_path))]
        for i in range(len(optimal_path)):
            for j in range(len(optimal_path[i]) - 1):
                path_latency_pro[i] += self.latency[optimal_path[i][j]][optimal_path[i][j + 1]]
        return path_latency_pro



