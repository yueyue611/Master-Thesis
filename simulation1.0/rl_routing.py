import numpy as np
import math
from collections import defaultdict
import itertools
from enum import Enum


# Q-learning parameters
gamma = 0.9
learning_rate = 0.8
epsilon = 0.8
temperature = 0.0005
total_episodes = 100
max_steps = 100


class Graph:
    def __init__(self, nodes):
        # number of vertices
        self.vertices = nodes

        # sets up a default dictionary to store graph
        self.graph = defaultdict(list)

        # initializes adjacency matrix of latency and bandwidth
        self.adj_mat_latency = [[np.Inf for _ in range(self.vertices)] for _ in range(self.vertices)]
        self.adj_mat_bandwidth = [[0 for _ in range(self.vertices)] for _ in range(self.vertices)]

    # adds directed edges to graph u --> v
    def add_edges(self, u, v, lt, bw):
        # connects u to v, updates the weighted edges
        self.graph[u].append(v)
        self.adj_mat_latency[u][v] = lt
        self.adj_mat_bandwidth[u][v] = bw

    # gets adjacency matrix of latency and bandwidth
    def adj_mat(self):
        return self.adj_mat_latency, self.adj_mat_bandwidth

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
            for i in self.graph[s]:
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


# calculates maximum bandwidth of each path
def get_path_bandwidth(total_paths, paths, bandwidth):
    path_bandwidth = [0 for _ in range(total_paths)]
    for i in range(total_paths):
        for j in range(len(paths[i]) - 1):
            if bandwidth[paths[i][j]][paths[i][j + 1]] > path_bandwidth[i]:
                path_bandwidth[i] = bandwidth[paths[i][j]][paths[i][j + 1]]
    return path_bandwidth


# calculates propagation latency of each path
def get_path_latency_pro(total_paths, paths, latency):
    path_latency_pro = [0 for _ in range(total_paths)]
    for i in range(total_paths):
        for j in range(len(paths[i]) - 1):
            path_latency_pro[i] += latency[paths[i][j]][paths[i][j+1]]
    return path_latency_pro


# calculates the approximate queue latency of each path
def get_path_latency_que(total_paths, path_bandwidth):
    path_latency_que = [0 for _ in range(total_paths)]
    for i in range(total_paths):
        path_latency_que[i] = 1512 * 8 * 30 / (path_bandwidth[i] * 1000)
    return path_latency_que


# help function
def get_index(a, item):
    return [index for (index, value) in enumerate(a) if value == item]


# calculates reward matrix R
def getting_reward(total_paths, total_states, total_flows, flows_rate, state_matrix, path_bandwidth,
                   path_latency_pro, path_latency_que, action_mode):
    # stores reward value of each state
    r = [0 for _ in range(total_states)]
    for state in range(total_states):
        cost = [0 for _ in range(total_paths)]
        for path in range(total_paths):
            # checks which flow chooses this path
            flow_index = get_index(state_matrix[state], path)
            flow_rate_sum = 0
            for (i, j) in enumerate(flow_index):
                flow_rate_sum += flows_rate[j]
            # if there is a congestion
            if flow_rate_sum > path_bandwidth[path]:
                for i in range(len(flow_index)):
                    cost[path] += (path_latency_pro[path] + path_latency_que[path]) ** 2
            else:
                for i in range(len(flow_index)):
                    cost[path] += path_latency_pro[path] ** 2
        # calculates reward as minus root mean square of the sum of all latencies
        r[state] = -math.sqrt(sum(cost) / total_flows)

    # reward matrix
    r_matrix = np.matrix(np.full((total_states, total_states), None))
    index = np.arange(0, total_flows)
    for state_cur in range(total_states):
        for state_next in range(total_states):
            # one flow change
            if action_mode.value == 0:
                # if the next state is reachable: no transition or only one flow changes
                if len(index[np.array(state_matrix[state_cur]) != np.array(state_matrix[state_next])]) <= 1:
                    r_matrix[state_cur, state_next] = r[state_next]
            # direct change
            else:
                r_matrix[state_cur, state_next] = r[state_next]
    # print("reward matrix: ", r_matrix)
    return r, r_matrix


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


def main():
    #                      sw1
    #  h00   10Mbit/s /10ms    20ms\ 10Mbit/s   h30
    #  h01 -------- sw0             sw3 ------- h31
    #  h02   10Mbit/s \5ms      15ms/ 10Mbit/s   h32
    #                      sw2
    # number of switches
    total_switches = 5

    # bitrate of flows: f0: 3 Mbit/s, f1: 6 Mbit/s, f2: 9 Mbit/s
    flows_rate = [3, 3, 3, 6, 9]
    # number of flows
    total_flows = len(flows_rate)

    # creates a graph for the above topology
    graph = Graph(total_switches)
    # adds information to graph edges: (source, destination, latency, bandwidth)
    graph.add_edges(0, 1, 10, 10)
    graph.add_edges(0, 2, 5, 10)
    graph.add_edges(0, 4, 20, 10)
    graph.add_edges(1, 3, 20, 10)
    graph.add_edges(2, 3, 15, 10)
    graph.add_edges(2, 4, 15, 10)
    graph.add_edges(3, 4, 10, 10)

    # direct change or one flow change
    action_mode = ActionMode.DIRECT_CHANGE  # ONE_FLOW_CHANGE or DIRECT_CHANGE
    print("action mode:", action_mode.name)

    # exploration method
    exploration_mode = ExplorationMode.SOFTMAX  # GREEDY or SOFTMAX
    print("exploration mode:", exploration_mode.name)

    # all flows are routed from switch0 to switch3
    s = 0
    d = 4
    # finds all paths
    paths = graph.all_paths(s, d)
    print("all possible paths: ", paths)
    # number of paths
    total_paths = len(paths)
    # number of state space
    total_states = total_paths ** total_flows

    # two adjacency matrices with the value of latency and bandwidth
    latency, bandwidth = graph.adj_mat()
    print("latency:", latency,
          "\nbandwidth:", bandwidth)

    #                 path0:sw0-sw1-sw3     path1:sw0-sw2-sw3
    # link capacity:    10 Mbit/s              10 Mbit/s
    # link latency:       30ms                    20ms

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
