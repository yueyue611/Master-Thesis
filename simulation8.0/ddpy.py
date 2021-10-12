import math
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from env import Env
from config import Config
from noise import OUNoise, GaussianNoise


flows = Config.flows
nodes = Config.nodes
max_bw = Config.max_bw
max_link_lt = Config.max_link_lt
env = Env(flows, nodes, max_bw, max_link_lt)

state_dim, action_dim = env.observation_space()
print("Size of State Space ->  {}".format(state_dim))
print("Size of Action Space ->  {}".format(action_dim))

upper_bound = Config.upper_bound
lower_bound = Config.lower_bound


# replay buffer
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # max number of previous experiences to store
        self.buffer_capacity = buffer_capacity
        # number of tuples to train on
        self.batch_size = batch_size
        # experience counter
        self.experience_counter = 0

        # use different np.arrays for each tuple element
        # instead of experience = (state, action, reward, next_state, done)
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim))
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim))

    # add (s,a,r,s') observation
    def record(self, state, action, reward, next_state):
        if self.experience_counter < self.buffer_capacity:
            index = self.experience_counter
        else:
            # if buffer_capacity is exceeded, new experience will replace old ones randomly
            index = random.randint(0, self.buffer_capacity - 1)

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state

        self.experience_counter += 1

    # decorate with tf.function
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model,
               target_actor_model, target_critic_model, actor_optimizer, critic_optimizer, gamma):
        # train and update critic and actor network
        # open a GradientTape
        with tf.GradientTape() as tape:
            target_actions = target_actor_model(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic_model([next_state_batch, target_actions], training=True)
            critic_value = critic_model([state_batch, action_batch], training=True)
            # minimize critic loss
            critic_loss = tf.math.reduce_mean(tf.math.square(critic_value - y))

        # get gradients of weights wrt the loss
        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        # Update the weights
        critic_optimizer.apply_gradients(zip(critic_grad, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # maximize actor loss
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(zip(actor_grad, actor_model.trainable_variables))

    # randomly sample batch_size examples and update parameters
    def learn(self, actor_model, critic_model, target_actor_model, target_critic_model, actor_optimizer,
              critic_optimizer, gamma):
        # get sampling range
        record_range = min(self.experience_counter, self.buffer_capacity)
        # randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch, actor_model, critic_model,
                    target_actor_model, target_critic_model, actor_optimizer, critic_optimizer, gamma)


@tf.function
# update target parameters slowly based a hyperparameter `tau` between 0-1(<< 1)
def update_target_weights(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(tau * b + (1 - tau) * a)


class ActorNetwork:
    def __init__(self):
        self.HIDDEN1_UNITS = 16
        self.HIDDEN2_UNITS = 32
        self.h_activation = "relu"
        self.activation = "sigmoid"

    # create actor network
    def create_actor_network(self):
        inputs = layers.Input(shape=(state_dim,))
        h1 = layers.Dense(self.HIDDEN1_UNITS, activation=self.h_activation, name='a_h1')(inputs)
        h2 = layers.Dense(self.HIDDEN2_UNITS, activation=self.h_activation, name='a_h2')(h1)
        outputs = layers.Dense(action_dim, activation=self.activation, name='a_out')(h2)  # 0-1

        # outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model


class CriticNetwork:
    def __init__(self):
        self.HIDDEN1_UNITS = 16
        self.HIDDEN2_UNITS = 32
        self.h_activation = "relu"
        self.activation = "sigmoid"

    # create critic network
    def create_critic_network(self):
        state_inputs = layers.Input(shape=(state_dim,))
        h1_s = layers.Dense(self.HIDDEN1_UNITS, activation=self.h_activation, name='c_h1_s')(state_inputs)
        state_outputs = layers.Dense(self.HIDDEN2_UNITS, activation=self.h_activation, name='c_out_s')(h1_s)

        action_inputs = layers.Input(shape=(action_dim,), name='c_in_a')
        action_outputs = layers.Dense(self.HIDDEN2_UNITS, activation=self.h_activation, name='c_out_a')(action_inputs)

        concat = layers.Concatenate()([state_outputs, action_outputs])
        h1 = layers.Dense(self.HIDDEN1_UNITS, activation=self.h_activation, name='c_h1')(concat)
        h2 = layers.Dense(self.HIDDEN2_UNITS, activation=self.h_activation, name='c_h2')(h1)
        outputs = layers.Dense(action_dim, activation='linear', name='c_out')(h2)
        model = tf.keras.Model([state_inputs, action_inputs], outputs)
        return model


def policy(actor_model, state, noise, weights_original, indicator, exploration_rate=1.0):
    weights = np.array(weights_original+weights_original).reshape(1, nodes ** 2 * 2)
    sampled_actions = actor_model(state)
    noise = noise()
    noise_dec = exploration_rate * noise
    # exploration: add decreased noise
    if indicator == 1 and exploration_rate > 0 and np.random.randint(0, 1) < 0.7:
        action_plus_n = sampled_actions + noise_dec
        # print("plus: ", action_plus_n)
        action_minus_n = sampled_actions - noise_dec
        # print("minus: ", action_minus_n)
        sampled_actions = np.where((action_plus_n < upper_bound) & (action_plus_n > lower_bound),
                                   action_plus_n, action_minus_n)
    # test or exploitation
    else:
        sampled_actions = sampled_actions.numpy()
    # make sure action is within bounds, set 0 to link weights in action when there is no link
    legal_actions = np.where((weights == 1), sampled_actions, 0).clip(lower_bound, upper_bound)
    action = np.squeeze(legal_actions)
    # print("legal actions: ", legal_actions)
    return action


def mode_selection(mode_select, len_traffic_load, queue_length, flows_tl, ft):
    if mode_select == len_traffic_load:
        index1 = 0  # queue_length = 1
        queue_length_select = queue_length[index1]
        flow_traffic = [flows_tl[ft][i][2] for i in range(len(flows_tl[ft]))]
    else:
        index2 = -2  # traffic load = 1.0
        flow_traffic = [flows_tl[index2][i][2] for i in range(len(flows_tl[index2]))]
        queue_length_select = queue_length[ft]
    return flow_traffic, queue_length_select


"""
training
"""


def main():
    start = time.time()

    noise_mode = Config.noise
    if noise_mode == "OU":
        noise = OUNoise(action_dim)
    elif noise_mode == "Gaussian":
        noise = GaussianNoise(action_dim)

    # Learning rate for actor-critic models
    critic_lr = Config.critic_lr
    actor_lr = Config.actor_lr

    # optimizer implements the Adam algorithm
    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    # discount factor for future rewards
    gamma = Config.gamma
    # used to update target networks
    tau = Config.tau

    a_delay = Config.a_delay
    a_pkt_loss = Config.a_pkt_loss

    total_episodes = Config.total_episodes
    total_steps = Config.total_steps

    # set up different traffic load level
    traffic_load = Config.traffic_load
    len_traffic_load = len(traffic_load)

    # set up different queue length
    queue_length = Config.queue_length
    len_queue_length = len(queue_length)

    # original link weights matrix
    weights_original, flows_original = env.get_original()
    print("original flows: ", flows_original)

    # store all paths
    all_paths = env.get_all_paths()
    print("all_paths:", all_paths)

    # measure performance according to queue length or traffic load
    mode = Config.mode  # "TL" or "QL"
    mode_select = Config.mode_select

    # determine if there is joining or leaving flows after N episodes
    N = Config.N
    mode_flow_change = Config.mode_flow_change
    changed_flows_num = Config.changed_flows_num
    new_flows = env.new_flows(mode_flow_change, changed_flows_num)

    experiment = Config.experiment

    for ex in range(9, experiment + 9):
        # to store reward history of each episode
        ep_reward_list = [[] for i in range(mode_select)]
        ep_r_delay_list = [[] for i in range(mode_select)]
        ep_r_pkt_loss_list = [[] for i in range(mode_select)]
        ep_avg_delay_list = [[] for i in range(mode_select)]

        # random case
        ep_reward_random_list = [[] for i in range(mode_select)]
        ep_r_delay_random_list = [[] for i in range(mode_select)]
        ep_r_pkt_loss_random_list = [[] for i in range(mode_select)]
        ep_avg_delay_random_list = [[] for i in range(mode_select)]

        # to store average reward history of last few episodes
        avg_reward_list = [[] for j in range(mode_select)]
        avg_r_delay_list = [[] for j in range(mode_select)]
        avg_r_pkt_loss_list = [[] for j in range(mode_select)]
        avg_avg_delay_list = [[] for j in range(mode_select)]

        opt_path_list = [[] for k in range(mode_select)]

        ep_converged = [0 for k in range(mode_select)]
        reward_converged = [0 for k in range(mode_select)]

        for ft in range(mode_select):
            actor_model = ActorNetwork().create_actor_network()
            critic_model = CriticNetwork().create_critic_network()
            target_actor_model = ActorNetwork().create_actor_network()
            target_critic_model = CriticNetwork().create_critic_network()

            # make the weights equal initially
            target_actor_model.set_weights(actor_model.get_weights())
            target_critic_model.set_weights(critic_model.get_weights())

            # create buffer
            buffer = Buffer(1000, 64)

            # initialization
            env.update_flows("NONE", new_flows, flows_original)
            flows_tl = env.get_flow_tl(traffic_load)
            flow_traffic, queue_length_select = mode_selection(mode_select, len_traffic_load, queue_length, flows_tl, ft)
            prev_state = env.reset(weights_original, flow_traffic)
            print("Initialization",
                  "\nflows_tl: ", flows_tl,
                  "\nflow_traffic: ", flow_traffic,
                  "\nprev_state", prev_state)

            exploration_rate = 1.0

            for ep in range(total_episodes):
                print("case {}: episode {} begins!".format(ft, ep))

                episodic_reward = 0
                episodic_r_delay = 0
                episodic_avg_delay = 0
                episodic_r_pkt_loss = 0

                episodic_reward_random = 0
                episodic_r_delay_random = 0
                episodic_avg_delay_random = 0
                episodic_r_pkt_loss_random = 0

                # joining or leaving flows after Nth episode:
                if ep == N:
                    if mode_flow_change != "NONE":
                        env.update_flows(mode_flow_change, new_flows, flows_original)
                        flows_tl = env.get_flow_tl(traffic_load)
                        flow_traffic, queue_length_select = mode_selection(mode_select, len_traffic_load, queue_length,
                                                                           flows_tl, ft)
                        # use action from last episode
                        action = np.array(action).reshape((nodes, nodes))
                        prev_state = env.reset(action, flow_traffic)
                        # update all paths
                        all_paths = env.get_all_paths()

                for step in range(total_steps):
                    tf_prev_state = tf.convert_to_tensor(prev_state.reshape(1, nodes ** 2))

                    # update exploration rate
                    exploration_rate -= 1.0 / (total_episodes * total_steps)

                    # shape: (nodes ** 2 * 2, )
                    # 1 for training
                    action = policy(actor_model, tf_prev_state, noise, weights_original, 1, exploration_rate)

                    # receive state and reward from environment
                    state, reward, r_delay, avg_delay, r_pkt_loss = env.step(action, flow_traffic, queue_length_select,
                                                                             a_delay, a_pkt_loss)

                    # add replay buffer
                    buffer.record(prev_state.reshape(1, nodes ** 2), action, reward, state.reshape(1, nodes ** 2))
                    episodic_reward += reward
                    episodic_r_delay += - r_delay
                    episodic_avg_delay += avg_delay
                    episodic_r_pkt_loss += - r_pkt_loss

                    buffer.learn(actor_model, critic_model, target_actor_model, target_critic_model, actor_optimizer,
                                 critic_optimizer, gamma)
                    update_target_weights(target_actor_model.variables, actor_model.variables, tau)
                    update_target_weights(target_critic_model.variables, critic_model.variables, tau)

                    # update state
                    prev_state = state

                    # random selection
                    paths_random = [all_paths[i][np.random.randint(0, len(all_paths[i]))] for i in range(0, flows)]
                    state_random = env.get_state(paths_random, flow_traffic)
                    reward_random, r_delay_random, avg_delay_random, r_pkt_loss_random = env.get_reward(
                        state_random, paths_random, queue_length_select, a_delay, a_pkt_loss)
                    episodic_reward_random += reward_random
                    episodic_r_delay_random += - r_delay_random
                    episodic_avg_delay_random += avg_delay_random
                    episodic_r_pkt_loss_random += - r_pkt_loss_random

                # mean reward of each episode
                ep_reward_list[ft].append(episodic_reward / total_steps)
                ep_r_delay_list[ft].append(episodic_r_delay / total_steps)
                ep_avg_delay_list[ft].append(episodic_avg_delay / total_steps)
                ep_r_pkt_loss_list[ft].append(episodic_r_pkt_loss / total_steps)

                # random case
                ep_reward_random_list[ft].append(episodic_reward_random / total_steps)
                ep_r_delay_random_list[ft].append(episodic_r_delay_random / total_steps)
                ep_avg_delay_random_list[ft].append(episodic_avg_delay_random / total_steps)
                ep_r_pkt_loss_random_list[ft].append(episodic_r_pkt_loss_random / total_steps)

                # mean reward of last 10 episodes
                avg_reward = np.mean(ep_reward_list[ft][-10:])
                avg_reward_list[ft].append(avg_reward)

                avg_r_delay = np.mean(ep_r_delay_list[ft][-10:])
                avg_r_delay_list[ft].append(avg_r_delay)

                avg_avg_delay = np.mean(ep_avg_delay_list[ft][-10:])
                avg_avg_delay_list[ft].append(avg_avg_delay)

                avg_r_pkt_loss = np.mean(ep_r_pkt_loss_list[ft][-10:])
                avg_r_pkt_loss_list[ft].append(avg_r_pkt_loss)

                # to see whether the total reward has converged, sliding window size = 10
                cost = 0
                for i in range(0, min(10, ep + 1)):
                    cost += (ep_reward_list[ft][-1 - i] - avg_reward) ** 2
                RMSD = math.sqrt(cost / min(10, ep + 1))
                # print("RMSD: ", RMSD)
                if RMSD < 0.001 and ep_converged[ft] == 0 and ep >= 10:
                    ep_converged[ft] = ep + 1
                    reward_converged[ft] = ep_reward_list[ft][-1]

            # test
            tf_prev_state_test = tf.convert_to_tensor(prev_state.reshape(1, nodes ** 2))
            action_test = policy(actor_model, tf_prev_state_test, noise, weights_original, 0)  # 0 for test
            opt_path_list[ft].append(env.get_opt_path_advance(np.array(action_test).reshape((nodes * 2, nodes)),
                                                              flow_traffic))
        print("opt_path_list", opt_path_list)

        # create csv
        # No.{ex}
        df = pd.DataFrame(ep_reward_list)
        df.to_csv("/home/tud/Github/Master-Thesis/simulation8.0/csv/{}/No.{}, {}, {}, {}, {}, {}.csv"
                  .format(noise_mode, ex, total_episodes, total_steps, noise_mode, mode, mode_flow_change),
                  header=False, index=False)

        df = pd.DataFrame(reward_converged)
        df.to_csv("/home/tud/Github/Master-Thesis/simulation8.0/csv/{}/CR: No.{}, {}, {}, {}, {}, {}.csv"
                  .format(noise_mode, ex, total_episodes, total_steps, noise_mode, mode, mode_flow_change),
                  header=False, index=False)

        end = time.time()
        runtime = end - start
        print("runtime: ", runtime)

    # plot graph
    # episodes versus Avg. Rewards
    colors = ['r', 'y', 'g', 'c', 'b', 'm', 'gray', 'orange', 'purple', 'pink']
    colors2 = ['gray', 'orange', 'purple', 'pink', 'r', 'y', 'g', 'c', 'b', 'm']
    labels_1 = ['TL=0.2', 'TL=0.3', 'TL=0.5', 'TL=0.7', 'TL=1.0', 'TL=1.2']
    labels_2 = ['QL=1', 'QL=2', 'QL=3']
    labels_1R = ['Random TL=0.2', 'Random TL=0.3', 'Random TL=0.5', 'Random TL=0.7', 'Random TL=1.0', 'Random TL=1.2']
    labels_2R = ['Random QL=1', 'Random QL=2', 'Random QL=3']
    if mode_select == len_traffic_load:
        labels = labels_1
        labelsR = labels_1R
    else:
        labels = labels_2
        labelsR = labels_2R

    plt.figure(1)
    for i in range(mode_select):
        plt.plot(ep_reward_list[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    # plt.savefig('Avg. Episodic Reward.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(2)
    for i in range(mode_select):
        plt.plot(ep_r_delay_list[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic r_delay")
    # plt.savefig('Avg. Episodic r_delay.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(3)
    for i in range(mode_select):
        plt.plot(ep_avg_delay_list[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic avg_delay")
    # plt.savefig('Avg. Episodic r_delay.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(4)
    for i in range(2, 3):
        plt.plot(ep_reward_list[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
        plt.plot(ep_reward_random_list[i], linewidth=0.5, linestyle='-', marker='^', markersize=2, label=labelsR[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic Reward")
    # plt.savefig('Avg. Episodic Reward.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(5)
    for i in range(2, 3):
        plt.plot(ep_r_delay_list[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
        plt.plot(ep_r_delay_random_list[i], linewidth=0.5, linestyle='-', marker='^', markersize=2, label=labelsR[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic r_delay")
    # plt.savefig('Avg. Episodic r_delay.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(6)
    for i in range(2, 3):
        plt.plot(ep_avg_delay_list[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
        plt.plot(ep_avg_delay_random_list[i], linewidth=0.5, linestyle='-', marker='^', markersize=2, label=labelsR[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic avg_delay")
    # plt.savefig('Avg. Episodic r_delay.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(7)
    for i in range(mode_select):
        plt.plot(ep_r_pkt_loss_list[i], linewidth=0.5, linestyle='--', marker='o', markersize=2, label=labels[i])
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Avg. Episodic r_pkt_loss")
    # plt.savefig('Avg. Episodic r_pkt_loss.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
