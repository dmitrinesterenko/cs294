X_SERVER=True
import os
import pdb
import tensorflow as tf
import gym
import roboschool
import numpy as np
try:
     import PIL.Image as Image
     import seaborn as sns
     import matplotlib.pyplot as plt
except ImportError:
     X_SERVER=False
     print("No X support")
from util import render_hopper, plot_animation, epoch



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
    """Class that defines the model that learns from the expert behavior"""
    def __init__(self, args):
        self.lr = 0.01
        self.l2 = 0.05
        self.n_inputs = 15 ##15 obs on the roboschool hopper, 11 observations for the mujoco hopper
        self.n_hidden = 170 # start small
        self.n_outputs = 3 # thigh, leg, foot joints
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        print("Env name {0}".format(args.envname))	
        self.env = gym.make(args.envname)
        self.render = args.render
        self.nn_name = "nn_{0}_{1}_{2}_{3}".format(self.lr, self.l2, self.n_hidden, epoch())
        self.weights_path = "weights/{0}".format(self.nn_name)

    def initialize(self):
        with tf.Session() as sess:
            sess.run(self.init)

    def build(self):
        with tf.variable_scope("NN", reuse=tf.AUTO_REUSE):
            self.X = tf.placeholder(tf.float32, shape=(None, self.n_inputs), name="X_marks_the_spot")
            self.y = tf.placeholder(tf.float32, shape=(None, self.n_outputs), name="y_is_it")

            hidden = tf.layers.dense(self.X, self.n_hidden, activation=tf.nn.elu,
kernel_initializer=self.initializer)
            self.logits = tf.layers.dense(hidden, self.n_outputs,
kernel_initializer=self.initializer)
            self.actions = self.logits
            #action_mask = tf.one_hot(self.action, self.action_size, 1.0, 0.0)
            #self.actions = tf.reduce_sum(self.logits * action_mask, 1)

            #self.actions = tf.nn.softmax(self.logits)
            ## Sigmoid Cross Entropy
            #self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.actions))
            ## MSE
            #self.loss = tf.reduce_mean(tf.square(tf.subtract(self.y, self.actions)))
            ## Euclidean distance
            self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.y, self.actions))))
            #self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.actions)
            ## Softmax
            #self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.actions)
            self.optimizer = tf.train.AdamOptimizer(self.lr)
            #self.optimizer = tf.train.GradientDescentOptimizer(self.lr)
            #self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.minimize = self.optimizer.minimize(self.loss)
            self.init = tf.global_variables_initializer()



    def fit(self, X_train, y_train, batch_size, epoch, n_max_steps=1000,
verbose=100, sample=500):
        losses = []
        rewards = []
        reward_cum = 0
        print("Epoch {}".format(epoch))
        with tf.Session() as sess:
            #if(epoch==0):
            #    sess.run(self.init)
            #else:
            #    self.load_weights(sess)
            try:
                self.load_weights(sess)
            except Exception as e:
                sess.run(self.init)
                print('Weights not found {0}'.format(e))
            obs = self.env.reset()
            batch_size = 60 #X_train.shape[0] # if the num_steps == 300 then this batch is 2 rollouts
            steps = int(X_train.shape[0]/batch_size)
            for step in range(steps):
                # FUN FACT: if you try to create a feed_dict like {X: X_train} where
                # X is not actually in scope in this function you will get a strange
                # seeming error that refers to Not a hashable type 'np.ndarray'
                # thus it's important to actually refer to self.X (or however you
                # would access the placeholder variable in the current function
                # scope)
                feed_dict = {
                    self.X: X_train[step*batch_size:(step+1)*batch_size],
                    self.y: y_train[step*batch_size:(step+1)*batch_size]
                }
                #print(y_train[step])
                #print(sess.run(self.logits[step], feed_dict=feed_dict))
                #pdb.set_trace()
                loss = sess.run(self.loss, feed_dict=feed_dict )
                losses.append(np.mean(loss))

                #grads_and_vars = sess.run(self.grads_and_vars, feed_dict=feed_dict)
                # Run the optimizer to learn better loss
                optimize = sess.run(self.minimize, feed_dict=feed_dict)

                # Note action here is an array of size equal to the roll-in size
                # of X so it's not a single generated action
                # Example: with a 1000 step roll-out of the expert the
                # action.shape if (1000,3) for the Hopper, 1000 actions are
                # suggested with each action being a 3 dimensional action

                # actions = sess.run(self.actions, feed_dict=feed_dict)
                # rewards = self.run(actions, step=step, render=false)
                # reward_cum = np.sum(rewards)

                if step*batch_size % verbose == 0:
                    print("Step {0}/{2}: Loss {1}".format(step*batch_size, np.mean(loss), steps*batch_size))
                    self.save_weights(sess)

            #TODO
            #Should use x_validation and y_validation to get the actions here
            feed_dict = { self.X : X_train,
                          self.y : y_train}
            rewards_return, steps = self.run(sess.run(self.actions, feed_dict=feed_dict), step, render=False)
            rewards.append(rewards_return)

            # this works when we have rewards # self.plot(steps=len(rewards), rewards=rewards, losses=losses)
            #self.plot(steps=len(losses), rewards=rewards, losses=losses, epoch=epoch)
            return losses, rewards, steps

    def run(self, actions, step=0, render=False):
        """
        Run through of the current environment with the predicted
        actions (exciting and is the part that we will be optimizing
        and learning how to construct actions given an experts
        roll-out)

        actions : array of actions to take
        step: what step of training we are on, this is used primarily for noting
            which training step we are on when saving the rendered animation
        render: boolean of whether to render the animation

        """
        frames = []
        rewards = []
        reward_cum = 0
        steps_taken = 0
        for i in range(actions.shape[0]):
            obs, reward, done, info = self.env.step(actions[i])
            reward_cum += reward
            if render:
                img = render_hopper(self.env, obs)
                frames.append(img)
            if done:
                print("We are done at step {0}".format(i))
                break
            steps_taken = i
        rewards.append(reward_cum)
        if render:
            animation = plot_animation(frames, repeat=True, step=step)
        return rewards, i

    def save_weights(self, sess):
        saver = tf.train.Saver()
        if not os.path.exists(self.weights_path):
            os.makedirs(self.weights_path)
        saver.save(sess, self.weights_path)

    def load_weights(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, self.weights_path)

    def adjust_learning_rate(self, new_learning_rate):
        self.lr = new_learning_rate
        with tf.variable_scope("NN", reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr)

    def plot(self, steps=0, losses=[], rewards=[], epoch=0):
        if X_SERVER:
            fig = plt.figure()
            plt.title("Loss and reward over steps trained")
            plt.axis("off")

            #df = pd.DataFrame()
            #df['losses'] = losses
            #df['rewards'] = rewards
            fig.add_subplot(1,2,1)
            sns.tsplot(time=range(steps), data=losses, linestyle='-')
            fig.add_subplot(1,2,2)
            sns.tsplot(time=range(len(rewards)), data=np.reshape(rewards, -1), linestyle="-")
            plt.savefig("output/{0}_{1}.png".format(self.nn_name, epoch))
        else:
            print("The reward {0} and loss {1} means".format(np.mean(rewards),
np.mean(losses)))

