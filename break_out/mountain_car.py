import gym
import numpy as np
import pickle
import logging
import tensorflow as tf
from plot_3d_value_function import plot_value_function


env = gym.make('MountainCar-v0')
env.reset()

ACTION_SPACE = [0, 1, 2]
K = 4
SPEED_RANGE = [-0.07, 0.07]
POSITION_RANGE = [-1.21, 0.61]
RESOLUTION = 10
NUM_ACTIONS = 3


def select_random_action(action_space):
    action = action_space[np.random.randint(0,3)]
    return action


def one_hot_encoding_single_input(parameter_input, parameter_range, resolution):
    check_input(parameter_input, parameter_range)
    #logging.info('input: {}, range: {}'.format(parameter_input, parameter_range))
    index = int(np.floor(((parameter_input - parameter_range[0]) / (parameter_range[1] - parameter_range[0])) * resolution))
    features = np.zeros(resolution)
    features[index] = 1
    return np.matrix(features)


def check_input(parameter_input, parameter_range):
    if (parameter_input <= parameter_range[0]) | (parameter_input > parameter_range[1]):
        logging.warning('input out of range, input: {}, range: {}'.format(parameter_input, parameter_range))

# TEST
print(one_hot_encoding_single_input(0.5, POSITION_RANGE, 10))


def one_hot_encoding(state, position_range, speed_range, resolution):
    position = state[0]
    position_features = one_hot_encoding_single_input(position, position_range, resolution)
    speed = state[1]
    speed_features = one_hot_encoding_single_input(speed, speed_range, resolution)
    result = np.matmul(position_features.transpose(), speed_features)
    return result.flatten()


def state_action_one_hot(state, action, position_range, speed_range, resolution):
    state_one_hot = one_hot_encoding(state, position_range, speed_range, resolution)
    action_one_hot = np.matrix([0.0,0.0,0.0])
    action_one_hot[0,action] = 1.0
    result = np.matmul(state_one_hot.transpose(), action_one_hot)
    return np.array(result.flatten())[0]

# TEST
res = state_action_one_hot([-0.5, 0.0], 1, POSITION_RANGE, SPEED_RANGE, RESOLUTION)
print(res)

def calcualte_max_funtion(one_hot_action_weights, NUM_ACTIONS, RESOLUTION):
    # unpack actions
    action_mat = one_hot_action_weights.reshape(RESOLUTION * RESOLUTION, NUM_ACTIONS)
    speed_position_mats = []
    for action in range (NUM_ACTIONS):
        # unpack to 2d speed position mat
        speed_position_mats.append(action_mat[:, action].reshape(RESOLUTION, RESOLUTION))
    q_function = np.amax(np.array(speed_position_mats), 0)
    return q_function

res2 = calcualte_max_funtion(res, NUM_ACTIONS, RESOLUTION)
print(res2)
#plot_value_function(res2, POSITION_RANGE, SPEED_RANGE, RESOLUTION)



number_states = RESOLUTION * RESOLUTION * NUM_ACTIONS


session = tf.Session()
tf_state = tf.placeholder("float", [None, number_states])
targets = tf.placeholder("float", [None, 1])

weights = tf.Variable(tf.constant(0., shape=[number_states, 1]))
#freeze_weights = tf.placeholder("float", [number_states, NUM_ACTIONS])

value_function = tf.matmul(tf_state, weights)

loss = tf.reduce_mean(tf.square(value_function - targets))
train_operation = tf.train.AdamOptimizer(0.1).minimize(loss)

session.run(tf.initialize_all_variables())


def choose_action(state, number_actions, e):
    if np.random.rand() < e:
        action = np.random.randint(0, number_actions)
    else:

        new_state_action_one_hot0 = state_action_one_hot(state, 0, POSITION_RANGE, SPEED_RANGE, RESOLUTION)
        new_state_action_one_hot1 = state_action_one_hot(state, 1, POSITION_RANGE, SPEED_RANGE, RESOLUTION)
        new_state_action_one_hot2 = state_action_one_hot(state, 2, POSITION_RANGE, SPEED_RANGE, RESOLUTION)

        value_new_state_action0 = session.run(value_function, feed_dict={tf_state: [new_state_action_one_hot0]})
        value_new_state_action1 = session.run(value_function, feed_dict={tf_state: [new_state_action_one_hot1]})
        value_new_state_action2 = session.run(value_function, feed_dict={tf_state: [new_state_action_one_hot2]})

        action = np.argmax([value_new_state_action0, value_new_state_action1, value_new_state_action2])
    return action


# def sample_from_min_batch(state_batch, action_batch, reward_batch, new_state_batch, terminal_batch, batch_size):
#     memory_size = len(state_batch)
#     indexes = [np.random.randint(0, memory_size) for _ in range(batch_size)]
#     return np.array(state_batch)[indexes], np.array(action_batch)[indexes], np.array(reward_batch)[indexes], np.array(new_state_batch)[indexes], np.array(terminal_batch)[indexes]

e = 0.9
#choose_action(one_hot_encoding([0.0, 0.0], POSITION_RANGE, SPEED_RANGE, RESOLUTION), NUM_ACTIONS, e)


def create_batch(e):
    # 1000 in replay memory
    # N = 1000

    memory_size = 4000
    freeze_weights = session.run(weights)

    state_ation_batch = []
    action_batch = []
    new_state_batch = []
    reward_batch = []
    terminal_batch = []
    target_batch = []
    state = np.array([-0.5, 0.0])

    # initialise memory
    env.reset()
    for i in range(memory_size):
        env.render()
        # Need to save state, action, reward, state
        # Choose action
        action = choose_action(state, NUM_ACTIONS, e)
        # run env
        for j in range(4):
            result = env.step(action)
        finished = result[2]
        reward = result[1]
        new_state = result[0]
        state_ation_batch.append(state_action_one_hot(state, action, POSITION_RANGE, SPEED_RANGE, RESOLUTION))
        action_batch.append(action)
        new_state_batch.append(new_state)

        # r + value of next state
        new_state_action_one_hot0 = state_action_one_hot(new_state, 0, POSITION_RANGE, SPEED_RANGE, RESOLUTION)
        new_state_action_one_hot1 = state_action_one_hot(new_state, 1, POSITION_RANGE, SPEED_RANGE, RESOLUTION)
        new_state_action_one_hot2 = state_action_one_hot(new_state, 2, POSITION_RANGE, SPEED_RANGE, RESOLUTION)

        #print(new_state_action_one_hot0)

        # if we have finished we know that the target is -1.0 else estimate the target from previous parameter set.
        if finished:
            target = [[-1.0]]
        else:
            value_new_state_action0 = session.run(value_function, feed_dict={tf_state: [new_state_action_one_hot0]})
            value_new_state_action1 = session.run(value_function, feed_dict={tf_state: [new_state_action_one_hot1]})
            value_new_state_action2 = session.run(value_function, feed_dict={tf_state: [new_state_action_one_hot2]})
            max_value_new_state_action = max([value_new_state_action0, value_new_state_action1, value_new_state_action2])
            #logging.info('Value_new_state_action: {}'.format(max_value_new_state_action))
            target = max_value_new_state_action + reward
        #logging.info('target: {}'.format(target))

        reward_batch.append(reward)
        target_batch.append(target[0])
        terminal_batch.append(result[2])

        logging.info('state: {}, action: {}, reward: {}, new state: {}'.format(state, action, reward, new_state))
        state = new_state.copy()


        if finished:
            logging.info('Success.. ending run... score: {}'.format(i))
            break


        #print(result)
    # print(state_ation_batch)
    # print(action_batch)
    # print(reward_batch)
    # print(new_state_batch)
    # print(target_batch)

 #   print(sample_from_min_batch(state_batch,action_batch, reward_batch, new_state_batch, terminal_batch, 32))

    logging.info('Start Learning')

    session.run(train_operation, feed_dict={
        tf_state: state_ation_batch,
        targets: target_batch})
    np_weights = session.run(weights)
    print(np_weights)
    plotable_weights = calcualte_max_funtion(np_weights, NUM_ACTIONS, RESOLUTION)
    return plotable_weights, i


if __name__ == '__main__':
    identifier = 'test'
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    scores = []
    for i in range(20000):
        e = 2000 / (2000 + i)
        logging.info('e: {}'.format(e))
        logging.info('start {}'.format(i))
        plotable_weights, score = create_batch(e)
        scores.append(score)
        if i%100 == 0:
            pickle.dump(plotable_weights, open('results/weights_{}_{}.p'.format(i, identifier), 'wb'))
            pickle.dump(scores, open('results/scores_{}_{}.p'.format(i, identifier), 'wb'))
            #plot_value_function(plotable_weights, POSITION_RANGE, SPEED_RANGE, RESOLUTION)
