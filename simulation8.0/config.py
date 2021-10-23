class Config:
    # number of flows
    flows = 5

    # number of nodes
    nodes = 5  # consistent with graph

    # link bandwidth (Mbit/s)
    max_bw = 10

    # link latency (ms)
    max_link_lt = 10

    # action boundary
    upper_bound = 1
    lower_bound = 0.01

    # noise type
    noise = "OU"
    # noise = "Gaussian"

    # Learning rate for actor-critic models
    critic_lr = 0.002
    actor_lr = 0.001

    # discount factor for future rewards
    gamma = 0.99
    # used to update target networks
    tau = 0.005

    # weights for determining reward
    a_delay = 1.0
    a_pkt_loss = 100

    # number of episodes, steps
    total_episodes = 300
    total_steps = 100

    # traffic load
    traffic_load = [0.2, 0.3, 0.5, 0.7, 1.0, 1.2]
    # traffic_load = [0.7, 1.0, 1.2]

    # queue length
    queue_length = [1, 2, 3]

    # select mode
    mode = "TL"  # "TL" or "QL"
    if mode == "TL":
        mode_select = len(traffic_load)
    else:
        mode_select = len(queue_length)

    # JOIN or LEAVE after N episodes
    N = 50
    mode_flow_change = "NONE"  # "JOIN" or "LEAVE" or "NONE"
    changed_flows_num = 2

    # number of experiments
    experiment = 1
