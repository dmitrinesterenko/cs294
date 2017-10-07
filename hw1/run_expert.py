

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
import pdb
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import seaborn as sns
import matplotlib.pyplot as plt
from util import epoch



def BatchGenerator():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        if(max_steps > 1000):
            print("the hopper will fall of the cliff after 1000 steps")
            max_steps = 1000
        print("max steps {} ".format(max_steps))
        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                #pdb.set_trace()
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("{}/{} total reward {}, reward sample {}"
                                        .format(steps, max_steps, totalr, r))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))
        fig = plt.figure()
        sns.tsplot(time=range(args.num_rollouts), data=returns, color='b', linestyle=':')
        plt.title("Reward over time")
        #plt.show()
        plt.savefig("rewards_plt_{}.png".format(epoch()))
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        yield np.array(observations), np.array(actions)

        #TODO
        # Now that we have the expert observations and actions we can train our
        # own agent using the actions and their rewards as the labeled y which
        # will be used for our own networks loss functions
        #PLAN
        # 1. Figure out the dimension of the outputs of my network (i.e. what
        # does the hopper do?)
        # Hopper has position, height and angle
        # 2. Build our own network
        # 3. Start with a random action
        # 4. Calculate loss using the expert policy
        # 5. Backpropagate the loss into the network
        # 6. Keep training

class Model():
    def __init__(self):
        self.lr = 0.01
        self.l2 = 0.02
        self.n_inputs = 11 ## 11 observations for the hopper
        self.n_hidden = 4 # start small
        self.n_outputs = 3 # thigh, leg, foot joints
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

    def build(self):
        X = tf.placeholder(tf.float32, shape=(20, self.n_inputs), name="X_marks_the_spot")
        hidden = tf.layers.dense(X, self.n_hidden, activation=tf.nn.elu,
kernel_initializer=self.initializer)
        logits = tf.layers.dense(hidden, self.n_outputs,
kernel_initializer=self.initializer)
        outputs = tf.nn.softmax(logits)
        self.actions = outputs

        #yp = tf.placeholder(tf.float32, shape=[None, 1, self.n_outputs], name="y_is_it")
        #self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=yp, logits=self.actions)
        #optimizer = tf.train.AdamOptimizer(self.lr)
        #self.grads_and_vars = optimizer.compute_gradients(self.loss)

        self.init = tf.global_variables_initializer()

    def fit(self, X, y, batch_size, epoch):
        # TODO should we take y[batch_size:] and self.actions[batch_size:] for
        # each execution
        # TODO these lines can be in build() as well and fit will just be the
        # with tf.Session() ...
        #import pdb; pdb.set_trace()

        #yp = tf.placeholder(tf.float32, shape=[None, 1, self.n_outputs], name="y_is_it")
        #loss = tf.nn.softmax_cross_entropy_with_logits(labels=yp, logits=self.actions)
        #optimizer = tf.train.AdamOptimizer(self.lr)
        #grads_and_vars = optimizer.compute_gradients(loss)

        with tf.Session() as sess:
            sess.run(self.init)
            import pdb; pdb.set_trace()
            loss, _ = sess.run(self.actions, feed_dict={X: X})
            print("Loss is {0}".format(loss))


if __name__ == '__main__':
    model = Model()
    model.build()
    for X, y in BatchGenerator():
        #model.train(X, y, batch_size=32, epoch=1)
        model.fit(X, y, batch_size=32, epoch=1)

