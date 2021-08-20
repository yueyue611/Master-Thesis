import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import tensorflow as tf
from tensorflow.keras import layers

from env_try3 import Env


flows = 10
nodes = 6  # consistent with graph
max_bw = 10
max_link_lt = 50
env = Env(flows, nodes, max_bw, max_link_lt)

state_dim, action_dim = env.observation_space()
print("Size of State Space ->  {}".format(state_dim))
print("Size of Action Space ->  {}".format(action_dim))

upper_bound = 1
lower_bound = 0


# Ornstein-Uhlenbeck noise
class OUNoise:
    def __init__(self, processes, mean=0, sigma=0.3, theta=0.15, dt=0.1, x_initial=None):
        self.processes = processes  # action_dim
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_initial = x_initial
        self.x_prev = self.reset()

    def __call__(self):
        dw = norm.rvs(scale=self.dt, size=self.processes)
        dx = self.theta * (self.mean - self.x_prev) * self.dt + self.sigma * dw
        x = self.x_prev + dx
        # store x into x_prev, make next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.ones(self.processes) * self.mean
        return self.x_prev


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
        # if buffer_capacity is exceeded, new experience will replace oldest ones
        index = self.experience_counter % self.buffer_capacity

        self.state_buffer[index] = state
        self.action_buffer[index] = action
        self.reward_buffer[index] = reward
        self.next_state_buffer[index] = next_state

        self.experience_counter += 1

    # decorate with tf.function
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
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
    def learn(self):
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

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


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


def policy(state, noise, weights_original, indicator, exploration_rate=1.0):
    weights = np.array(weights_original).reshape(1, nodes ** 2)
    sampled_actions = actor_model(state)
    noise = noise()
    # greedy exploration: add decreased noise
    if indicator == 1 and np.random.uniform(0, 1) < 0.8 and exploration_rate > 0:
        action_plus_n = sampled_actions.numpy() + exploration_rate * noise
        action_minus_n = sampled_actions.numpy() - exploration_rate * noise
        sampled_actions = np.where((action_plus_n < upper_bound) & (action_plus_n > lower_bound),
                                   action_plus_n, action_minus_n)
    # test or exploitation
    else:
        sampled_actions = sampled_actions.numpy()
    # make sure action is within bounds, set 0 to link weights in action when there is no link
    legal_actions = np.where((weights == 1), sampled_actions, 0).clip(lower_bound, upper_bound)
    action = np.squeeze(legal_actions)
    """
    print("sampled actions: ", sampled_actions)
    print("noise: ", noise)
    print("legal actions: ", legal_actions)
    """
    return action


"""
training
"""

ou_noise = OUNoise(action_dim)

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001
# optimizer implements the Adam algorithm
critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

# discount factor for future rewards
gamma = 0.95
# used to update target networks
tau = 0.005
# create buffer
buffer = Buffer(1000, 16)

total_episodes = 100
total_steps = 100

# set up different traffic load level
traffic_load = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
len_traffic_load = len(traffic_load)
flows_tl = env.get_flow_tl(traffic_load)

# set up different queue length
queue_length = [1, 2, 3, 4, 5]
# queue_length = [i for i in range(1, flows + 1)]
len_queue_length = len(queue_length)
print("queue_length: ", queue_length)

# original link weights matrix
weights_original = env.get_weights()

mode_select = len_queue_length   # len_traffic_load or len_queue_length

# To store reward history of each episode
ep_reward_list = [[] for i in range(mode_select)]
# To store average reward history of last few episodes
avg_reward_list = [[] for j in range(mode_select)]

opt_path_list = [[] for j in range(mode_select)]

for ft in range(mode_select):
    if mode_select == len_traffic_load:
        index1 = 2  # queue_length = 3
        queue_length_select = queue_length[-1]
        flow_traffic = [flows_tl[index1][i][2] for i in range(len(flows_tl[index1]))]
    else:
        index2 = -2  # traffic load = 1.0
        flow_traffic = [flows_tl[index2][i][2] for i in range(len(flows_tl[index2]))]
        queue_length_select = queue_length[ft]

    actor_model = ActorNetwork().create_actor_network()
    critic_model = CriticNetwork().create_critic_network()
    target_actor_model = ActorNetwork().create_actor_network()
    target_critic_model = CriticNetwork().create_critic_network()

    # make the weights equal initially
    target_actor_model.set_weights(actor_model.get_weights())
    target_critic_model.set_weights(critic_model.get_weights())

    # initial state of simulator
    prev_state = env.reset(weights_original, flow_traffic)

    exploration_rate = 1.0

    for ep in range(total_episodes):
        print("case {}: episode {} begins!".format(ft, ep))
        episodic_reward = 0

        for step in range(total_steps):
            tf_prev_state = tf.convert_to_tensor(prev_state.reshape(1, nodes ** 2))

            # update exploration rate
            exploration_rate -= 1.0 / (total_episodes * total_steps)

            # shape: (nodes ** 2, )
            action = policy(tf_prev_state, ou_noise, weights_original, 1, exploration_rate)  # 1 for training

            # receive state and reward from environment
            state, reward = env.step(action, flow_traffic, queue_length_select)

            # add replay buffer
            buffer.record(prev_state.reshape(1, nodes ** 2), action, reward, state.reshape(1, nodes ** 2))
            episodic_reward += reward

            buffer.learn()
            update_target_weights(target_actor_model.variables, actor_model.variables, tau)
            update_target_weights(target_critic_model.variables, critic_model.variables, tau)

            prev_state = state

        ep_reward_list[ft].append(episodic_reward)

        # mean of last episodes
        avg_reward = np.mean(ep_reward_list[ft][-10:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list[ft].append(avg_reward)

    # test
    prev_state_test = env.reset(weights_original, flow_traffic)
    tf_prev_state_test = tf.convert_to_tensor(prev_state_test.reshape(1, nodes ** 2))
    action_test = policy(tf_prev_state_test, ou_noise, weights_original, 0)  # 0 for test
    opt_path_list[ft].append(env.get_opt_path(np.array(action_test).reshape((nodes, nodes))))

print("opt_path_list", opt_path_list)

# plot graph
# episodes versus Avg. Rewards
colors = ['r', 'y', 'g', 'c', 'b', 'm', 'gray', 'orange', 'purple', 'pink']
labels_1 = ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2']
labels_2 = ['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K']
l = ['l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7', 'l8', 'l9', 'l10']
if mode_select == len_traffic_load:
    labels = labels_1
else:
    labels = labels_2
for i in range(mode_select):
    l[i] = plt.plot(avg_reward_list[i], color=colors[i], linewidth=1.0, linestyle='--', marker='o', label=labels[i])
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Avg. Episodic Reward")
plt.show()
