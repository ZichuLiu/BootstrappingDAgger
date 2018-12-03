import tensorflow as tf
import tf_util
import load_policy
from BootstrappingBehaviorClonning import policy_maker, get_save_path, get_save_path_DAgger
import numpy as np
import pickle
import gym
import env_wrapper
import os
from Bootstrap_data_retriever import bootstrap_training_set

tf.set_random_seed = 0


class DAgger:

    def __init__(self,
                 expert_policy_file='experts//HalfCheetah-v2.pkl',
                 gym_env='HalfCheetah-v2'):
        self.expert_policy_file = expert_policy_file
        self.gym_env = gym_env
        self.custom_policy = policy_maker()
        self.num_networks = 30

    def initialize_policy_with_params(self, weights, bias):
        self.custom_policy.load_policy(weights, bias)

    def run_DAgger(self, beta=1, iter=1000):
        # setup the expert policy
        expert_policy_fn = load_policy.load_policy(self.expert_policy_file)
        # setup the output arrays, one for observations and one for expert actions
        actions = list()
        observations = list()
        rewards = list()
        displacement = list()
        # setup the gym environment and reset the state.
        env = env_wrapper.env_wrapper(gym.make(self.gym_env))
        obs = env.reset()
        saver = tf.train.Saver()
        totalr = 0

        with tf.Session() as sess_DAgger:
            # run the mixed policies $iter iterations to collect training data.

            for i in range(0, iter):
                action = list()
                for network in range(self.num_networks):
                    saver.restore(sess_DAgger, save_path=get_save_path_DAgger(network))
                    # print('Restore Network: {0}'.format(network))
                    inv_obs = obs.reshape(1, obs.shape[0])
                    _, _, Tensor_action = self.custom_policy.multilayer_perceptron(inv_obs)
                    action.append(Tensor_action.eval())
                action = np.asarray(action)
                action = action.reshape((action.shape[0], action.shape[2]))
                stds_act = np.std(action, axis=0)
                mean_act = np.mean(action, axis=0)
                print('Sum of standard deviation of actions: {0}'.format(np.sum(stds_act)))

                if np.sum(stds_act) > beta:
                    # run the expert policy with a probability of beta and collect this data make it the new training data.
                    action = expert_policy_fn(obs[None, :])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, reward_dict = env.step(action)
                    rewards.append(r)
                    displacement.append(reward_dict['reward_run'])
                    totalr += r
                    if done:
                        print('Terminate within ' + str(i) + ' iterations and reward:' + str(totalr))
                        break
                # run the customized policy to explore unfamiliar states but not collect these data.
                else:
                    obs, r, done, reward_dict = env.step(mean_act)
                    print('Reward after taking custom policy: {0}'.format(r))
                    totalr += r
                    if done:
                        print('Terminate within ' + str(i) + ' iterations and reward:' + str(totalr))
                        break
                env.render()
            print(beta)
        env.close()
        print('Number of expert demonstrations generated: ' + str(len(observations)) + ' reward:' + str(totalr))
        return [observations, actions, rewards]


if __name__ == '__main__':
    with open('expert_data//HalfCheetah-v2.pkl', 'rb') as f:
        Data_input = pickle.load(f)
    alpha = 1e-4
    epochs = 10000000000000000000001
    input_size = Data_input['actions'].shape[0]
    observations = Data_input['observations'][:input_size // 10 * 9]
    target_observations = Data_input['observations'][input_size // 10 * 9:]

    actions = np.reshape(Data_input['actions'], (Data_input['actions'].shape[0], Data_input['actions'].shape[2]))[
              :input_size // 10 * 9]
    target_actions = np.reshape(Data_input['actions'], (Data_input['actions'].shape[0], Data_input['actions'].shape[2]))[
                     input_size // 10 * 9:]

    new_DAgger = DAgger()
    store_weights = list()
    store_bias = list()
    input_size = actions.shape[0]
    weights = list()
    bias = list()

    beta = 1
    c_old = 100
    c_old_2 = 100

    with tf.Session() as sess:
        obs = tf.placeholder(shape=[None, new_DAgger.custom_policy.n_obs], dtype=tf.float64)
        desired_action = tf.placeholder(shape=[None, new_DAgger.custom_policy.n_action], dtype=tf.float64)
        layer1, layer2, action = new_DAgger.custom_policy.multilayer_perceptron(obs)
        loss_op = tf.losses.mean_squared_error(labels=desired_action, predictions=action)
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
        train_op = optimizer.minimize(loss_op)
        saver = tf.train.Saver()
        for i in range(0, 20):
            new_Data = new_DAgger.run_DAgger(beta=beta)
            beta = beta * 0.89125
            new_observations = np.asarray(new_Data[0])
            new_actions = np.asarray(new_Data[1])
            new_actions = np.reshape(new_actions, (new_actions.shape[0], new_actions.shape[2]))
            observations = np.concatenate((observations, new_observations), axis=0)
            actions = np.concatenate((actions, new_actions), axis=0)
            print(observations.shape)
            for iter in range(new_DAgger.num_networks):
                print('Neural network: {0}'.format(iter))
                x_train, y_train, _ = bootstrap_training_set(observations, actions)
                saver.restore(sess, save_path=get_save_path_DAgger(iter))
                print('Restore Network: {0}'.format(iter))
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
                            save_path = saver.save(sess, save_path=get_save_path_DAgger(i))
                            print('Stored! Model saved in path: %s' % save_path)
                        else:
                            c_old = 100
                            c_old_2 = 100
                            break