"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""
X_SERVER=True
import pdb
import pickle
import tensorflow as tf
import numpy as np
import gym
import IPython.display as display
import PIL.Image as Image
import seaborn as sns
import pandas as pd
try:
     import matplotlib.pyplot as plt
except ImportError:
     X_SERVER=False
     print("No X support")

import tf_util
import load_policy
from util import epoch, render_hopper, plot_environment, try_action, plot_animation


def BatchGenerator(args):
    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        # TRIAL
        #try_action(env, [0,0,1])
        #try_action(env, [0,1,0])
        #try_action(env, [1,0,0])
        #END TRIAL

        print("The action space is {0}".format(env.action_space))

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
                observations.append(obs)
                # This was actions.append(action) which an array of [?,1,3]
                # which then contradicted what is produced in my own model which
                # produced for the actions [?,3]
                actions.append(action[0])
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
        plt.savefig("output/rewards_plt_{}.png".format(epoch()))
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
    def __init__(self, args):
        self.lr = 0.01
        self.l2 = 0.02
        self.n_inputs = 11 ## 11 observations for the hopper
        self.n_hidden = 4 # start small
        self.n_outputs = 3 # thigh, leg, foot joints
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        self.env = gym.make(args.envname)
        self.render = args.render

    def build(self):
        self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name="X_marks_the_spot")
        self.y = tf.placeholder(tf.float32, shape=(None, self.n_outputs), name="y_is_it")

        hidden = tf.layers.dense(self.X, self.n_hidden, activation=tf.nn.elu,
kernel_initializer=self.initializer)
        logits = tf.layers.dense(hidden, self.n_outputs,
kernel_initializer=self.initializer)
        self.actions = tf.nn.softmax(logits)

        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.actions)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
        self.minimize = self.optimizer.minimize(self.loss)
        self.init = tf.global_variables_initializer()

    def plot(self, steps=0, losses=[], rewards=[]):
        fig = plt.figure()
        plt.title("Loss and reward over steps trained")
        plt.axis("off")

        #df = pd.DataFrame()
        #df['losses'] = losses
        #df['rewards'] = rewards
        fig.add_subplot(1,2,1)
        sns.tsplot(time=range(steps), data=losses, linestyle='-')
        fig.add_subplot(1,2,2)
        sns.tsplot(time=range(steps), data=rewards, linestyle="-")
        plt.savefig("output/nn_performance_{}.png".format(steps))


    def fit(self, X_train, y_train, batch_size, epoch, n_max_steps=1000,
verbose=100, sample=500):
        # TODO should we take y[batch_size:] and self.actions[batch_size:] for
        # each execution
        rewards = []
        losses = []
        with tf.Session() as sess:
            sess.run(self.init)
            obs = self.env.reset()
            for step in range(n_max_steps):
                reward_cum = 0
                frames = []
                #img = render_hopper(self.env, obs)
                #frames.append(img)
                # FUN FACT: if you try to create a feed_dict like {X: X_train} where
                # X is not actually in scope in this function you will get a strange
                # seeming error that refers to Not a hashable type 'np.ndarray'
                # thus it's important to actually refer to self.X (or however you
                # would access the placeholder variable in the current function
                # scope)
                feed_dict = {self.X: X_train, self.y: y_train}
                loss = sess.run(self.loss, feed_dict=feed_dict )
                losses.append(np.mean(loss))

                #grads_and_vars = sess.run(self.grads_and_vars, feed_dict=feed_dict)
                optimize = sess.run(self.minimize, feed_dict=feed_dict)

                # Run the optimizer to learn better loss
                #...
                # Note action here is an array of size equal to the roll-in size
                # of X so it's not a single generated action
                # Example: with a 1000 step roll-out of the expert the
                # action.shape if (1000,3) for the Hopper, 1000 actions are
                # suggested with each action being a 3 dimensional action
                actions = sess.run(self.actions, feed_dict=feed_dict)

                # Do a run through of the current environment with the predicted
                # actions (exciting and is the part that we will be optimizing
                # and learning how to construct actions given an experts
                # roll-out)
                for i in range(len(actions)):
                    obs, reward, done, info = self.env.step(actions)
                    reward_cum += reward
                    if self.render and step % sample == 0:
                        img = render_hopper(self.env, obs)
                        frames.append(img)
                    #if done:
                    #    print("We are done at step {0}".format(i))
                    #    break
                rewards.append(reward_cum)

                if step % verbose == 0:
                    print("Step {0}: Loss {1}, Reward {2}".format(step, np.mean(loss), reward_cum))
                if self.render and step % sample == 0:
                    animation = plot_animation(frames, repeat=True, step=step)
                    #plt.show()
            self.plot(steps=n_max_steps, rewards=rewards, losses=losses)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    model = Model(args)
    model.build()

    # The shape is (10*num_rollouts, 11) for X and (10*num_rollouts, 3) for y
    for X, y in BatchGenerator(args):
        model.fit(X, y, batch_size=32, epoch=20, n_max_steps=100, sample=50, verbose=10)
        #model.fit(X, y, batch_size=32, epoch=20, n_max_steps=1000, sample=200, verbose=100)


