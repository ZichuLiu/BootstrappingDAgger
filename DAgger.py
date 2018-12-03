import tensorflow as tf
import tf_util
import load_policy
from BehaviorCloningCheetau import policy_maker
import numpy as np
import pickle
import gym
import env_wrapper

tf.set_random_seed = 0


class DAgger:

    def __init__(self,
                 expert_policy_file='//home//zichuliu//Desktop//hw1//homework-master//hw1//experts//HalfCheetah-v2.pkl',
                 gym_env='HalfCheetah-v2'):
        self.expert_policy_file = expert_policy_file
        self.gym_env = gym_env
        self.custom_policy = policy_maker()

    def initialize_policy_with_params(self, weights, bias):
        self.custom_policy.load_policy(weights, bias)

    def run_DAgger(self, beta=0.5, iter=1000):
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
            saver.restore(sess_DAgger, tf.train.latest_checkpoint(
                '//home//zichuliu//Desktop//hw1//homework-master//hw1//HalfCheetah//tmp//DAgger'))
            # saver.restore(sess,
            #               '//home//zichuliu//Desktop//hw1//homework-master//hw1//behavior_cloning//Adamtmp/tmp/model134.085')
            print('DAgger Model Restored')
            for i in range(0, iter):
                # print(i)
                rand_b = np.random.random()
                tf_util.initialize()
                # run the expert policy with a probability of beta and collect this data make it the new training data.
                if rand_b < beta:
                    action = expert_policy_fn(obs[None, :])
                    observations.append(obs)
                    actions.append(action)
                    obs, r, done, reward_dict = env.step(action)
                    rewards.append(r)
                    displacement.append(reward_dict['reward_run'])
                    # print(r)
                    totalr += r
                    if done:
                        print('Terminate within ' + str(i) + ' iterations and reward:' + str(totalr))
                        break
                # run the customized policy to explore unfamiliar states but not collect these data.
                else:
                    inv_obs = obs.reshape(1, obs.shape[0])
                    _, _, Tensor_action = self.custom_policy.multilayer_perceptron(inv_obs)
                    action = Tensor_action.eval()
                    obs, r, done, reward_dict = env.step(action)
                    # print(obs)
                    totalr += r
                    # print(r)
                    if done:
                        print('Terminate within ' + str(i) + ' iterations and reward:' + str(totalr))
                        break
                env.render()

            print(beta)
        env.close()
        print('Number of expert demonstrations generated: ' + str(len(observations)) + ' reward:' + str(totalr))
        return [observations, actions, rewards]


if __name__ == '__main__':
    with open('//home//zichuliu//Desktop//hw1//homework-master//hw1//expert_data//HalfCheetah-v2.pkl', 'rb') as f:
        Data_input = pickle.load(f)

    alpha = 1e-4
    epochs = 2001
    observations = Data_input['observations']
    actions = Data_input['actions']
    actions = np.reshape(actions, (actions.shape[0], actions.shape[2]))
    new_DAgger = DAgger()
    store_weights = list()
    store_bias = list()
    input_size = actions.shape[0]
    weights = list()
    bias = list()
    beta = 0.25118547034616364
    with tf.Session() as sess:
        obs = tf.placeholder(shape=[None, new_DAgger.custom_policy.n_obs], dtype=tf.float64)
        desired_action = tf.placeholder(shape=[None, new_DAgger.custom_policy.n_action], dtype=tf.float64)
        layer1, layer2, action = new_DAgger.custom_policy.multilayer_perceptron(obs)
        loss_op = tf.losses.mean_squared_error(labels=desired_action, predictions=action)
        optimizer = tf.train.AdamOptimizer(learning_rate=alpha)
        train_op = optimizer.minimize(loss_op)
        saver = tf.train.Saver()
        for i in range(0, 20):
            new_Data = new_DAgger.run_DAgger(beta = beta)
            beta = beta * 0.89125
            new_observations = np.asarray(new_Data[0])
            new_actions = np.asarray(new_Data[1])
            new_actions = np.reshape(new_actions, (new_actions.shape[0], new_actions.shape[2]))
            observations = np.concatenate((observations, new_observations), axis=0)
            actions = np.concatenate((actions, new_actions), axis=0)
            print(observations.shape)

            saver.restore(sess, tf.train.latest_checkpoint(
                '//home//zichuliu//Desktop//hw1//homework-master//hw1//HalfCheetah//tmp//DAgger'))
            # saver.restore(sess,
            #               '//home//zichuliu//Desktop//hw1//homework-master//hw1//behavior_cloning//Adamtmp/tmp/model134.085')
            print('Model Restored')
            for epoch in range(epochs):
                c = sess.run(loss_op, feed_dict={obs: observations, desired_action: actions})
                sess.run(train_op, feed_dict={obs: observations, desired_action: actions})
                d = sess.run(loss_op, feed_dict={obs: observations, desired_action: actions})
                # print(c)
                if epoch % 100 == 0:
                    save_path = saver.save(sess,
                                           '/home/zichuliu/Desktop/hw1/homework-master/hw1/HalfCheetah/tmp/DAgger/model.ckpt')
                    print('Stored! Model saved in path: %s' % save_path)

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
