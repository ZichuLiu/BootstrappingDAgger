import tensorflow as tf
import tf_util
import load_policy
from BehaviorCloning import policy_maker
import numpy as np
import pickle
import gym

tf.set_random_seed = 0

class DAgger:

    def __init__(self,
                 expert_policy_file='//home//zichuliu//Desktop//hw1//homework-master//hw1//experts//Humanoid-v2.pkl',
                 gym_env='Humanoid-v2'):
        self.expert_policy_file = expert_policy_file
        self.gym_env = gym_env
        self.custom_policy = policy_maker()
        self.initialize_policy()

    def initialize_policy(self):
        PATH = self.custom_policy.PATH
        with open(PATH + 'num_1000_loss_33.71499375_weights.pickle', 'rb') as load_weights:
            weights = pickle.load(load_weights)
        with open(PATH + 'num_1000_loss_33.71499375_bias.pickle', 'rb') as load_biases:
            biases = pickle.load(load_biases)
        self.custom_policy.load_policy(weights, biases)
        return weights, biases

    def initialize_policy_with_params(self, weights, bias):
        self.custom_policy.load_policy(weights, bias)

    def run_DAgger(self, beta=0.5, iter=1000):
        # setup the expert policy
        expert_policy_fn = load_policy.load_policy(self.expert_policy_file)
        # setup the output arrays, one for observations and one for expert actions
        actions = list()
        observations = list()
        rewards = list()
        # setup the gym environment and reset the state.
        env = gym.make(self.gym_env)
        obs = env.reset()

        # run the mixed policies $iter iterations to collect training data.
        with tf.Session():
            tf_util.initialize()

            for i in range(0, iter):
                rand_b = np.random.random()

                # run the expert policy with a probability of beta and collect this data make it the new training data.
                if rand_b > beta:
                    action = expert_policy_fn(obs[None, :])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    rewards.append(r)
                    if done:
                        print('Terminate within ' + str(i) + ' iterations and reward:' + str(r))
                        break
                # run the customized policy to explore unfamiliar states but not collect these data.
                else:
                    inv_obs = obs.reshape(1, obs.shape[0])
                    Tensor_action = self.custom_policy.multilayer_perceptron(inv_obs)
                    action = Tensor_action.eval()
                    obs, r, done, _ = env.step(action)
                    if done:
                        print('Terminate within ' + str(i) + ' iterations and reward:' + str(r))
                        break
        print('Number of expert demonstrations generated: ' + str(len(observations)))
        return [observations, actions, rewards]


if __name__ == '__main__':
    with open('//home//zichuliu//Desktop//hw1//homework-master//hw1//expert_data//Humanoid-v2.pkl', 'rb') as f:
        Data_input = pickle.load(f)

    alpha = 0.001
    epochs = 2
    observations = Data_input['observations']
    actions = Data_input['actions']
    actions = np.reshape(actions, (actions.shape[0], actions.shape[2]))
    new_DAgger = DAgger()
    store_weights = list()
    store_bias = list()
    input_size = actions.shape[0]
    temp_layer = list()
    temp_layer_2 = list()
    for i in range(0, 100):
        # new_Data = new_DAgger.run_DAgger()
        # new_observatiosn = np.asarray(new_Data[0])
        # new_actions = np.asarray(new_Data[1])
        # new_actions = np.reshape(new_actions, (new_actions.shape[0], new_actions.shape[2]))
        # observations = np.concatenate((observations, new_observatiosn), axis=0)
        # actions = np.concatenate((actions, new_actions), axis=0)

        obs = tf.placeholder(shape=[None, new_DAgger.custom_policy.n_obs], dtype=tf.float64)
        desired_action = tf.placeholder(shape=[None, new_DAgger.custom_policy.n_action], dtype=tf.float64)
        layer1, layer2, action = new_DAgger.custom_policy.multilayer_perceptron(obs)
        loss_op = tf.losses.mean_squared_error(labels=desired_action, predictions=action)
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
        train_op = optimizer.minimize(loss_op)
        init = tf.global_variables_initializer()
        weights = list()
        bias = list()
        with tf.Session() as sess:
            sess.run(init)
            weights_watcher = new_DAgger.custom_policy.weights['h2'].eval()
            bias_watcher = new_DAgger.custom_policy.biases['b1'].eval()
            print('Watcher')
            print(bias_watcher[:10])
            print('watcher ends here')
            # print(bias_watcher)
            for epoch in range(epochs):
                c = sess.run(loss_op, feed_dict={obs: Data_input['observations'], desired_action: actions})
                sess.run(train_op, feed_dict={obs: Data_input['observations'], desired_action: actions})
                d = sess.run(loss_op, feed_dict={obs: Data_input['observations'], desired_action: actions})
                print('before')
                print(c)
                print('after')
                print(d)
                if epoch % 100 == 0:
                    weights = sess.run(new_DAgger.custom_policy.weights)
                    bias = sess.run(new_DAgger.custom_policy.biases)
                    layer_1 = layer1.eval(feed_dict={obs: Data_input['observations'], desired_action: actions})
                    print(layer_1)
                    print(temp_layer == layer_1)
                else:
                    temp_layer = layer1.eval(feed_dict={obs: Data_input['observations'], desired_action: actions})
                    print(temp_layer)
        new_DAgger.initialize_policy_with_params(weights, bias)

    if i % 20 == 0:
        store_weights = weights
        store_bias = bias
        with open(new_DAgger.custom_policy.PATH + 'DAgger//num_' + str(i) + '_loss_' + str(
                c / input_size) + '_weights.pickle',
                  'wb') as w:
            pickle.dump(store_weights, w)
        with open(new_DAgger.custom_policy.PATH + 'DAgger//num_' + str(i) + '_loss_' + str(
                c / input_size) + '_bias.pickle',
                  'wb') as b:
            pickle.dump(store_bias, b)
        print('Stored!')
print('terminate here')
