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

    def print_all_paths_helper(self, s, d, visited, path, all_path):
        visited[s] = True
        path.append(s)

        if s == d:
            all_path.append(path.copy())
        else:
            for i in self.edges[s]:
                if not visited[i]:
                    self.print_all_paths_helper(i, d, visited, path, all_path)

        path.pop()
        visited[s] = False

    # Prints all paths from s to d
    def print_all_paths(self, s, d):
        visited = [False] * self.vertices

        path = []
        all_path = []

        # Call the recursive helper function to print all paths
        self.print_all_paths_helper(s, d, visited, path, all_path)
        return all_path


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
            (0, 3, 1.0 * self.max_link_latency, self.max_bandwidth),
            (0, 4, 1.0 * self.max_link_latency, self.max_bandwidth),
            (1, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (1, 2, 1.0 * self.max_link_latency, self.max_bandwidth),
            (2, 1, 1.0 * self.max_link_latency, self.max_bandwidth),
            (2, 4, 1.0 * self.max_link_latency, self.max_bandwidth),
            (3, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (3, 4, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 0, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 2, 1.0 * self.max_link_latency, self.max_bandwidth),
            (4, 3, 1.0 * self.max_link_latency, self.max_bandwidth)
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
        self.flows = [[0, 4, 10], [0, 4, 10], [0, 4, 10], [0, 4, 10], [2, 0, 10]]
        # self.flows = self.get_flows(self.total_flows)
        self.flows.sort(key=lambda x: (x[0], x[1], x[2]))

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
        return flows_tl

    # get optimal path using dijkstra
    def get_opt_path(self, weights):
        opt_path = []
        for i in range(len(self.flows)):
            opt_path.append(self.graph.dijkstra(self.flows[i][0], self.flows[i][1], weights))
        return opt_path

    # get optimal path for different groups
    # weights here is action
    def get_opt_path_advance(self, weights, flow_traffic):
        opt_path = []
        flows_group = [[self.flows[0][0], self.flows[0][1], [flow_traffic[0]]]]
        index = 0
        for i in range(1, self.total_flows):
            if self.flows[i][0] == flows_group[index][0] and self.flows[i][1] == flows_group[index][1]:
                flows_group[index][2].append(flow_traffic[i])
            else:
                flows_group.append([self.flows[i][0], self.flows[i][1], [flow_traffic[i]]])
                index += 1
        for j in range(0, len(flows_group)):
            if len(flows_group[j][2]) == 1:
                opt_path.append(self.graph.dijkstra(flows_group[j][0], flows_group[j][1], weights))
            else:
                all_paths_onepair = self.graph.print_all_paths(flows_group[j][0], flows_group[j][1])
                for k in range(0, len(all_paths_onepair)):
                    distance = 0
                    for l in range(0, len(all_paths_onepair[k])-1):
                        distance += weights[all_paths_onepair[k][l]][all_paths_onepair[k][l+1]]
                    all_paths_onepair[k].append(distance)
                # all possible paths are sorted according to the aggregated weights
                all_paths_onepair.sort(key=lambda x: (x[-1]))
                #print("all_paths_onepair_sort:", all_paths_onepair)
                path_id = 0
                start = 0
                opt_path.append(all_paths_onepair[path_id][:-1])  # [:-1]: exclude the last item distance
                for end in range(1, len(flows_group[j][2])):
                    if sum(flows_group[j][2][start:end+1]) <= self.max_bandwidth:
                        opt_path.append(all_paths_onepair[path_id][:-1])
                    else:
                        path_id = (path_id + 1) % len(all_paths_onepair)
                        opt_path.append(all_paths_onepair[path_id][:-1])
                        start = end
        #print("path: ", opt_path)
        return opt_path

    # get state
    def get_state(self, paths, flow_traffic):
        state = np.zeros((self.total_switches, self.total_switches))
        for i in range(len(paths)):
            for j in range(len(paths[i]) - 1):
                state[paths[i][j]][paths[i][j + 1]] += flow_traffic[i]
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
        optimal_path = self.get_opt_path_advance(action_w, flow_traffic)  # modify!!!
        state = self.get_state(optimal_path, flow_traffic)
        reward, r_delay, avg_delay, r_pkt_loss = self.get_reward(state, optimal_path, queue_length_select, a_delay, a_pkt_loss)
        return state, reward, r_delay, avg_delay, r_pkt_loss

    # calculate propagation latency of each path
    def get_path_latency_pro(self, optimal_path):
        path_latency_pro = [0 for _ in range(len(optimal_path))]
        for i in range(len(optimal_path)):
            for j in range(len(optimal_path[i]) - 1):
                path_latency_pro[i] += self.latency[optimal_path[i][j]][optimal_path[i][j + 1]]
        return path_latency_pro

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
                    # assume each flow has 50 packets
                    # queue length is 50*(#flows), in order to make sure there is no packet loss
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
                # more than num flows on the same link would result in packet loss
                num = queue_length_select
                if sum_bandwidth > self.bandwidth[i][j] and sum_flow > num:
                    p = 1 - self.bandwidth[i][j] / sum_bandwidth
                # take non-zero p into account
                if p != 0:
                    cost.append(p ** 2)
        r_pkt_loss = math.sqrt(sum(cost) / self.total_edges)
        #print("r_pkt_loss: ", r_pkt_loss)
        return r_pkt_loss

    # calculate reward
    def get_reward(self, state, optimal_path, queue_length_select, a_delay, a_pkt_loss):
        r_delay, avg_delay = self.get_delay(state, optimal_path, queue_length_select)
        r_pkt_loss = self.get_pkt_loss(state, optimal_path, queue_length_select)
        r = - a_delay * r_delay - a_pkt_loss * r_pkt_loss
        #r = - a_delay * r_delay
        #print("reward: ", r)
        return r, r_delay, avg_delay, r_pkt_loss

    # get all paths for each flow
    def get_all_paths(self):
        all_paths = [[] for i in range(0, self.total_flows)]
        for i in range(0, self.total_flows):
            all_paths[i] = self.graph.print_all_paths(self.flows[i][0], self.flows[i][1])
        return all_paths


