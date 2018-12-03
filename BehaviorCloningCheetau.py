import tensorflow as tf
import pickle as pkl
import numpy as np


class policy_maker:
    def __init__(self, PATH='behavior_cloning//'):
        self.PATH = PATH
        # self.Save_PATH = '//home//zichuliu//Desktop//hw1//homework-master//hw1//'
        self.n_obs = 17
        self.n_hidden_1 = 64
        self.n_hidden_2 = 32
        #self.n_hidden_3 = 256
        #self.n_hidden_4 = 128
        self.n_action = 6
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_obs, self.n_hidden_1], dtype=tf.float64), dtype=tf.float64,
                              name='h1'),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], dtype=tf.float64), dtype=tf.float64,
                              name='h2'),
            # 'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3], dtype=tf.float64), dtype=tf.float64,
            #                   name='h3'),
            # 'h4': tf.Variable(tf.random_normal([self.n_hidden_3, self.n_hidden_4], dtype=tf.float64), dtype=tf.float64,
            #                   name='h4'),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_action], dtype=tf.float64), dtype=tf.float64,
                               name='out')}

        self.biases = {'b1': tf.Variable(tf.random_normal([self.n_hidden_1], dtype=tf.float64), dtype=tf.float64,
                                         name='b1'),
                       'b2': tf.Variable(tf.random_normal([self.n_hidden_2], dtype=tf.float64), dtype=tf.float64,
                                         name='b2'),
                       # 'b3': tf.Variable(tf.random_normal([self.n_hidden_3], dtype=tf.float64), dtype=tf.float64,
                       #                   name='b3'),
                       # 'b4': tf.Variable(tf.random_normal([self.n_hidden_4], dtype=tf.float64), dtype=tf.float64,
                       #                   name='b4'),
                       'bout': tf.Variable(tf.random_normal([self.n_action], dtype=tf.float64), dtype=tf.float64,
                                           name='bout')}

    def multilayer_perceptron(self, state_input):
        layer_1 = tf.nn.tanh(tf.matmul(state_input, self.weights['h1'])+self.biases['b1'])
        layer_2 = tf.nn.tanh(tf.matmul(layer_1, self.weights['h2'])+self.biases['b2'])
        # layer_3 = tf.nn.tanh(tf.matmul(layer_2, self.weights['h3'])+self.biases['b3'])
        # layer_4 = tf.nn.tanh(tf.matmul(layer_3, self.weights['h4'])+self.biases['b4'])
        action = tf.matmul(layer_2, self.weights['out']) + self.biases['bout']

        return layer_1, layer_2, action

    def load_policy(self, weights, biases):
        self.weights['h1'] = tf.Variable(tf.constant(weights['h1'], dtype=tf.float64))
        self.weights['h2'] = tf.Variable(tf.constant(weights['h2'], dtype=tf.float64))
        # self.weights['h3'] = tf.Variable(tf.constant(weights['h3'], dtype=tf.float64))
        # self.weights['h4'] = tf.Variable(tf.constant(weights['h4'], dtype=tf.float64))
        self.weights['out'] = tf.Variable(tf.constant(weights['out'], dtype=tf.float64))
        self.biases['b1'] = tf.Variable(tf.constant(biases['b1'], dtype=tf.float64))
        self.biases['b2'] = tf.Variable(tf.constant(biases['b2'], dtype=tf.float64))
        # self.biases['b3'] = tf.Variable(tf.constant(biases['b3'], dtype=tf.float64))
        # self.biases['b4'] = tf.Variable(tf.constant(biases['b4'], dtype=tf.float64))
        self.biases['out'] = tf.Variable(tf.constant(biases['bout'], dtype=tf.float64))


if __name__ == '__main__':
    alpha = 1e-3
    epochs = 100000000001
    policy = policy_maker()
    # policy.initialize_policy()
    with open('expert_data//HalfCheetah-v2.pkl', 'rb') as f:
        input = pkl.load(f)

    input_size = input['actions'].shape[0]
    input['actions'] = np.reshape(input['actions'], (input['actions'].shape[0], input['actions'].shape[2]))
    obs = tf.placeholder(shape=[None, policy.n_obs], dtype=tf.float64)
    desired_action = tf.placeholder(shape=[None, policy.n_action], dtype=tf.float64)
    layer1, layer2, action = policy.multilayer_perceptron(obs)
    loss_op = tf.losses.mean_squared_error(predictions=action, labels=desired_action)
    regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 1e-5
    loss = reg_constant * tf.reduce_sum(regularization) + loss_op
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,beta1=0.9,beta2=0.999)
    train_op = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # sess.run(init)
        saver.restore(sess, tf.train.latest_checkpoint('HalfCheetah/tmp'))
        print('Model Restored')
        for epoch in range(epochs):
            _, c = sess.run([train_op, loss_op],
                            feed_dict={obs: input['observations'], desired_action: input['actions']})
            print(c)
            if c < 0.5:
                alpha = 0.001
            if epoch % 100 == 0:
                save_path = saver.save(sess, 'HalfCheetah/tmp/model.ckpt')
                print('Stored! Model saved in path: %s' % save_path)
