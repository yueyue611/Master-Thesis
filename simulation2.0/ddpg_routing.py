import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import defaultdict
import itertools
from enum import Enum
import copy


class Graph:
    def __init__(self, nodes):
        # number of vertices
        self.vertices = nodes

        # a dict of all possible next nodes
        self.edges = defaultdict(list)

        # initialize adjacency matrix of latency and bandwidth
        self.adj_mat_latency = [[np.Inf for _ in range(self.vertices)] for _ in range(self.vertices)]
        self.adj_mat_bandwidth = [[0 for _ in range(self.vertices)] for _ in range(self.vertices)]

    def add_edge(self, u, v, lt, bw):
        # edges are unidirectional
        self.edges[u].append(v)
        self.adj_mat_latency[u][v] = lt
        self.adj_mat_bandwidth[u][v] = bw

    # get adjacency matrix of latency and bandwidth
    def adj_mat(self):
        return self.adj_mat_latency, self.adj_mat_bandwidth

    # implement dijkstra's shortest path algorithm
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
                return "Path not available"

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

    # recursive function helps to obtain all possible paths
    def all_paths_recur(self, s, d, visited, paths_cur, paths_all):

        # marks the current node as visited and store it in current path
        visited[s] = True
        paths_cur.append(s)

        # if the current node is destination, save the current found path to the path list
        if s == d:
            paths_all.append(paths_cur.copy())
        else:
            # otherwise recurs for all adjacent vertices
            for i in self.edges[s]:
                if not visited[i]:
                    self.all_paths_recur(i, d, visited, paths_cur, paths_all)

        # removes current node from current path and mark it as unvisited
        paths_cur.pop()
        visited[s] = False

    # prints all possible paths from s to d
    def all_paths(self, s, d):

        # first marks all nodes as not visited
        visited = [False for _ in range(self.vertices)]

        # stores the current single path and all paths
        paths_cur = []
        paths_all = []

        # calls the recursive function to obtain all paths
        self.all_paths_recur(s, d, visited, paths_cur, paths_all)
        return paths_all


# get optimal path according to dijkstra
def get_opt_path(graph, flows, weights):
    opt_path = []
    for i in range(len(flows)):
        opt_path.append(graph.dijkstra(flows[i][0], flows[i][1], weights))
    return opt_path


# get state: traffic matrix
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


def choose_action(possible_actions, possible_q, exploration_mode):
    # uses epsilon greedy
    if exploration_mode.value == 0:
        if np.random.uniform(0, 1) < epsilon:
            # selects a  random action
            action = possible_actions[np.random.randint(0, len(possible_actions))]
        else:
            # greedy, chooses the highest valued action
            action = possible_actions[np.argmax(possible_q)]
    # uses softmax:
    if exploration_mode.value == 1:
        total = sum([np.exp(-1 / (possible_q[i] * temperature)) for i in range(len(possible_actions))])
        probs = [(np.exp(-1 / (possible_q[i] * temperature)) / total) for i in range(len(possible_actions))]
        action = np.random.choice(possible_actions, p=probs)
        print(action)
    return action


def updating_q_value(q, state, next_state, reward, action):
    if q[state, action] == np.NINF:
        q[state, action] = 0
    q[state, action] = q[state, action] + learning_rate * (reward + gamma * np.max(q[next_state, :]) - q[state, action])
    # print("q[{}, {}]: {}".format(state, action, q[state, action]))


def training(total_states, r_matrix, exploration_mode):
    q = np.matrix(np.full((total_states, total_states), np.NINF))
    for episode in range(total_episodes):
        # random initial state
        state = np.random.randint(0, total_states)
        step = 0
        while step < max_steps:
            possible_actions = []
            possible_q = []
            for action in range(total_states):
                # we can not choose actions that one flow change can not achieve
                # all actions are possible when direct change
                if r_matrix[state, action] is not None:
                    possible_actions.append(action)
                    possible_q.append(q[state, action])
            # chooses action
            action = choose_action(possible_actions, possible_q, exploration_mode)
            # updates Q value
            next_state = action
            reward = r_matrix[state, action]
            updating_q_value(q, state, next_state, reward, action)
            # goes to the next state
            state = next_state
            step += 1
    return q


def test(total_states, q):
    # random initial state
    state = np.random.randint(0, total_states)
    step = 0
    state_transition = [state]
    while step < max_steps:
        # chooses action with the optimal policy
        action = np.argmax(q[state])
        # arrives to the next state
        state = action
        state_transition.append(state)
        step += 1
    return state_transition


class ExplorationMode(Enum):
    GREEDY = 0
    SOFTMAX = 1


class ActionMode(Enum):
    ONE_FLOW_CHANGE = 0
    DIRECT_CHANGE = 1


# Q-learning parameters
gamma = 0.9
learning_rate = 0.8
epsilon = 0.8
temperature = 0.0005
total_episodes = 100
max_steps = 100


def main():
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
        graph.add_edges(*edge)

    # two adjacency matrices with the value of latency and bandwidth
    latency, bandwidth = graph.adj_mat()
    print("latency matrix:", latency,
          "\nbandwidth matrix:", bandwidth)

    # number of flows
    total_flows = 4

    # flows[i] = [source, destination, traffic]
    flows = [[np.random.randint(0, total_switches) for _ in range(3)] for _ in range(total_flows)]
    for i in range(total_flows):
        flows[i][2] = np.random.randint(0, max_bandwidth)
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

    # here traffic load equal to 1
    flow_traffic = [flows_tl[3][i][2] for i in range(len(flows_tl[3]))]
    print("flow traffic:", flow_traffic)
    state = get_state(total_switches, optimal_path, flow_traffic)
    print("state:", state)

    # bitrate of flows: f0: 3 Mbit/s, f1: 6 Mbit/s, f2: 9 Mbit/s
    # flows_rate = [3, 3, 3, 6, 9]

    # direct change or one flow change
    # action_mode = ActionMode.DIRECT_CHANGE  # ONE_FLOW_CHANGE or DIRECT_CHANGE
    # print("action mode:", action_mode.name)

    # exploration method
    # exploration_mode = ExplorationMode.SOFTMAX  # GREEDY or SOFTMAX
    # print("exploration mode:", exploration_mode.name)

    # all flows are routed from switch0 to switch3
    # s = 0
    # d = 4
    # finds all paths
    # paths = graph.all_paths(s, d)
    # print("all possible paths: ", paths)
    # number of paths
    # total_paths = len(paths)
    # number of state space
    # total_states = total_paths ** total_flows

    # state = [[0 for _ in range(total_switches)] for _ in range(total_switches)]
    # for i in range(total_flows):
    #    state[flows[i][0]][flows[i][1]] = flows[i][2]


    # calculates maximum bandwidth, propagation latency, approximate queue latency of each path
    path_bandwidth = get_path_bandwidth(total_paths, paths, bandwidth)
    path_latency_pro = get_path_latency_pro(total_paths, paths, latency)
    path_latency_que = get_path_latency_que(total_paths, path_bandwidth)
    print("maximum bandwidth of each path: ", path_bandwidth,
          "\npropagation latency of each path: ", path_latency_pro,
          "\nqueue latency of each path: ", path_latency_que)

    # marks all state space
    # each tuple defines a routing path of a flow
    # e.g. state[0]=(0, 1, 0) represents that flow0 and flow2 choose path0, flow1 chooses path1
    state = list(itertools.permutations([i for i in range(total_paths)] * total_flows, total_flows))
    state_matrix = sorted(set(state))
    print("number of state space: ", total_states,
          "\nState Space:", state_matrix)

    # calculates reward matrix
    r, r_matrix = getting_reward(total_paths, total_states, total_flows, flows_rate, state_matrix,
                                 path_bandwidth, path_latency_pro, path_latency_que, action_mode)
    print("reward r: ", r)

    # training
    q = training(total_states, r_matrix, exploration_mode)
    print("Q-table:\n", q)

    # test
    state_transition = test(total_states, q)
    print("state transition: ", state_transition)


if __name__ == '__main__':
    main()
