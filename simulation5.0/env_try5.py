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
            (1, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (1, 5, 1.0 * self.max_link_latency, self.max_bandwidth),
            (2, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (2, 4, 1.0 * self.max_link_latency, self.max_bandwidth),
            (3, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (3, 5, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 2, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 5, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 1, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 3, 1.0 * self.max_link_latency, self.max_bandwidth),
            (5, 4, 1.0 * self.max_link_latency, self.max_bandwidth)
        ]

        for edge in self.edges:
            self.graph.add_edge(*edge)

        # number of edges/links
        self.total_edges = len(self.edges)

        # adjacency matrices with the value of 1/0, latency and bandwidth
        self.weights, self.latency, self.bandwidth = self.graph.adj_mat()
        print("\nweights: ", self.weights,
              "\nlatency: ", self.latency,
              "\nbandwidth: ", self.bandwidth)

        # number of flows
        self.total_flows = flows

        # flows[i] = [source, destination, traffic]
        # self.flows = [[0, 8, 10], [0, 8, 10], [0, 8, 10], [8, 0, 10], [8, 0, 10]]
        # self.flows = [[0, 8, 10], [1, 2, 10], [4, 1, 10], [2, 7, 10], [3, 7, 10]]
        self.flows = [[0, 5, 10], [2, 5, 10], [1, 2, 10], [1, 3, 10]]
        # self.flows = self.get_flows(self.total_flows)

    def get_flows(self, number):
        flows = [[0 for _ in range(3)] for _ in range(number)]
        for i in range(number):
            [node_i, node_j] = np.random.choice(self.total_switches, 2, replace=False)
            flows[i][0] = node_i
            flows[i][1] = node_j
            flows[i][2] = np.random.randint(1, self.max_bandwidth)  # traffic can not be 0
        return flows

    def new_flows(self, mode_flow_change, changed_flows_num):
        if mode_flow_change == "JOIN":
            # generate joining flows randomly
            new_flows = self.get_flows(changed_flows_num)
        elif mode_flow_change == "LEAVE":
            # select leaving flows randomly
            leave_flows_list = np.random.choice(self.total_flows, changed_flows_num, replace=False)
            new_flows = [val for n, val in enumerate(self.flows) if n not in leave_flows_list]
        else:
            new_flows = []
        return new_flows

    def update_flows(self, mode_flow_change, new_flows, old_flows):
        if mode_flow_change == "JOIN":
            self.flows.extend(new_flows)
        elif mode_flow_change == "LEAVE":
            self.flows = copy.deepcopy(new_flows)
        else:
            self.flows = copy.deepcopy(old_flows)
        self.total_flows = len(self.flows)
        print("flows: ", self.flows)

    def get_original(self):
        return self.weights, self.flows

    def observation_space(self):
        state_dim = self.total_switches ** 2
        action_dim = self.total_switches ** 2
        return state_dim, action_dim

    def get_flow_tl(self, traffic_load):
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

    # get number of flows per link
    def get_flow_per_link(self, optimal_path):
        flow_per_link = np.zeros((self.total_switches, self.total_switches))
        for i in range(len(optimal_path)):
            for j in range(len(optimal_path[i]) - 1):
                flow_per_link[optimal_path[i][j]][optimal_path[i][j + 1]] += 1
        # print("flow per link: ", flow_per_link)
        return flow_per_link

    def reset(self, weights, flow_traffic):
        prev_state = self.get_state(weights, flow_traffic)
        return prev_state

    def step(self, action, flow_traffic, queue_length_select, a_delay, a_pkt_loss):
        action_w = np.array(action).reshape((self.total_switches, self.total_switches))
        state = self.get_state(action_w, flow_traffic)
        reward, r_delay, avg_delay, r_pkt_loss = self.get_reward(state, action_w, queue_length_select, a_delay, a_pkt_loss)
        return state, reward, r_delay, avg_delay, r_pkt_loss

    # calculate delay
    def get_delay(self, state, optimal_path, queue_length_select):
        path_latency_pro = self.get_path_latency_pro(optimal_path)
        cost = [0 for _ in range(len(optimal_path))]
        cost2 = [0 for _ in range(len(optimal_path))]
        path_latency_que = [0 for _ in range(len(optimal_path))]
        for path in range(len(optimal_path)):
            for i in range(len(optimal_path[path]) - 1):
                sum_bandwidth = state[optimal_path[path][i]][optimal_path[path][i + 1]] + \
                                state[optimal_path[path][i + 1]][optimal_path[path][i]]
                # if there is a congestion of one link, calculate the approximate queue latency of one link
                if sum_bandwidth > self.bandwidth[optimal_path[path][i]][optimal_path[path][i + 1]]:
                    # calculate queueing latency for each path
                    # assume each flow has 10 packets
                    # queue length is 10*(#flows), in order to make sure there is no packet loss
                    path_latency_que[path] += 1512 * 8 * 50 * queue_length_select / \
                                              (self.bandwidth[optimal_path[path][i]][optimal_path[path][i + 1]] * 1000)
            # cost for each path
            cost[path] += (path_latency_pro[path] + path_latency_que[path]) ** 2
            cost2[path] = path_latency_pro[path] + path_latency_que[path]
        """
        print("optimal path: ", optimal_path,
              "\npath latency propagation: ", path_latency_pro,
              "\npath latency queue: ", path_latency_que,
              "\ncost:", cost)
        """
        # calculate delay component of reward as minus root mean square of the sum of all latencies
        r_delay = math.sqrt(sum(cost) / len(optimal_path))
        avg_delay = sum(cost) / len(optimal_path)
        #print("r_delay: ", r_delay)
        return r_delay, avg_delay

    # calculate packet loss rate
    def get_pkt_loss(self, state, optimal_path, queue_length_select):
        cost = []
        flow_per_link = self.get_flow_per_link(optimal_path)
        for i in range(self.total_switches - 1):
            for j in range(i + 1, self.total_switches):
                p = 0  # packet loss probability
                sum_bandwidth = state[i][j] + state[j][i]
                sum_flow = flow_per_link[i][j] + flow_per_link[j][i]
                if sum_bandwidth > self.bandwidth[i][j] and sum_flow > queue_length_select:
                    p = 1 - self.bandwidth[i][j] / sum_bandwidth
                # take non-zero p into account
                if p != 0:
                    cost.append(p ** 2)
        r_pkt_loss = math.sqrt(sum(cost) / self.total_edges)
        #print("r_pkt_loss: ", r_pkt_loss)
        return r_pkt_loss

    # calculate reward
    def get_reward(self, state, action, queue_length_select, a_delay, a_pkt_loss):
        optimal_path = self.get_opt_path(action)
        r_delay, avg_delay = self.get_delay(state, optimal_path, queue_length_select)
        r_pkt_loss = self.get_pkt_loss(state, optimal_path, queue_length_select)
        r = - a_delay * r_delay - a_pkt_loss * r_pkt_loss
        #r = - a_delay * r_delay
        #print("reward: ", r)
        return r, r_delay, avg_delay, r_pkt_loss

    # calculate propagation latency of each path
    def get_path_latency_pro(self, optimal_path):
        path_latency_pro = [0 for _ in range(len(optimal_path))]
        for i in range(len(optimal_path)):
            for j in range(len(optimal_path[i]) - 1):
                path_latency_pro[i] += self.latency[optimal_path[i][j]][optimal_path[i][j + 1]]
        return path_latency_pro

