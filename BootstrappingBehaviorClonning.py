import tensorflow as tf
import pickle as pkl
import numpy as np
from Bootstrap_data_retriever import bootstrap_training_set
import os


class policy_maker:
    def __init__(self, PATH='behavior_cloning//'):
        self.PATH = PATH
        self.n_obs = 17
        self.n_hidden_1 = 64
        self.n_hidden_2 = 32
        self.n_action = 6
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_obs, self.n_hidden_1], dtype=tf.float64), dtype=tf.float64,
                              name='h1'),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], dtype=tf.float64), dtype=tf.float64,
                              name='h2'),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_action], dtype=tf.float64), dtype=tf.float64,
                               name='out')}

        self.biases = {'b1': tf.Variable(tf.random_normal([self.n_hidden_1], dtype=tf.float64), dtype=tf.float64,
                                         name='b1'),
                       'b2': tf.Variable(tf.random_normal([self.n_hidden_2], dtype=tf.float64), dtype=tf.float64,
                                         name='b2'),
                       'bout': tf.Variable(tf.random_normal([self.n_action], dtype=tf.float64), dtype=tf.float64,
                                           name='bout')}

    def multilayer_perceptron(self, state_input):
        layer_1 = tf.nn.tanh(tf.matmul(state_input, self.weights['h1']) + self.biases['b1'])
        layer_2 = tf.nn.tanh(tf.matmul(layer_1, self.weights['h2']) + self.biases['b2'])
        action = tf.matmul(layer_2, self.weights['out']) + self.biases['bout']

        return layer_1, layer_2, action

    def load_policy(self, weights, biases):
        self.weights['h1'] = tf.Variable(tf.constant(weights['h1'], dtype=tf.float64))
        self.weights['h2'] = tf.Variable(tf.constant(weights['h2'], dtype=tf.float64))
        self.weights['out'] = tf.Variable(tf.constant(weights['out'], dtype=tf.float64))
        self.biases['b1'] = tf.Variable(tf.constant(biases['b1'], dtype=tf.float64))
        self.biases['b2'] = tf.Variable(tf.constant(biases['b2'], dtype=tf.float64))
        self.biases['out'] = tf.Variable(tf.constant(biases['bout'], dtype=tf.float64))


def get_save_path(net_number):
    save_dir = 'checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir + 'network' + str(net_number)


def get_save_path_DAgger(net_number):
    save_dir = 'DAgger/checkpoints/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir + 'network' + str(net_number)


if __name__ == '__main__':
    alpha = 1e-3
    epochs = 10000000000000000000000001
    num_networks = 100
    policy = policy_maker()

    with open('expert_data//HalfCheetah-v2.pkl', 'rb') as f:
        input = pkl.load(f)
    input_size = input['actions'].shape[0]
    observations = input['observations'][:input_size // 10 * 9]
    target_observations = input['observations'][input_size // 10 * 9:]

    actions = np.reshape(input['actions'], (input['actions'].shape[0], input['actions'].shape[2]))[
              :input_size // 10 * 9]
    target_actions = np.reshape(input['actions'], (input['actions'].shape[0], input['actions'].shape[2]))[
                     input_size // 10 * 9:]

    obs = tf.placeholder(shape=[None, policy.n_obs], dtype=tf.float64)
    desired_action = tf.placeholder(shape=[None, policy.n_action], dtype=tf.float64)
    layer1, layer2, action = policy.multilayer_perceptron(obs)
    loss_op = tf.losses.mean_squared_error(predictions=action, labels=desired_action)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1e-5
    loss = reg_constant * tf.reduce_sum(regularization) + loss_op
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha, beta1=0.9, beta2=0.999)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=num_networks)
    c_old = 100
    c_old_2 = 100
    with tf.Session() as sess:
        for i in range(35,num_networks):
            print('Neural network: {0}'.format(i))
            x_train, y_train, _ = bootstrap_training_set(observations, actions)
            sess.run(init)
            for epoch in range(epochs):
                _, c = sess.run([train_op, loss_op],
                                feed_dict={obs: x_train, desired_action: y_train})
                if epoch % 100 == 0:
                    c_new = sess.run(loss_op, feed_dict={obs: target_observations, desired_action: target_actions})
                    print('Previous loss 2:{0}'.format(c_old_2))
                    print('Previous loss 1:{0}'.format(c_old))
                    print('New loss:{0}'.format(c_new))
                    if c_new < c_old or c_new < c_old_2:
                        c_old_2 = c_old
                        c_old = c_new
                        save_path = saver.save(sess, save_path=get_save_path(i))
                        print('Stored! Model saved in path: %s' % save_path)
                    else:
                        c_old = 100
                        c_old_2 = 100
                        break
