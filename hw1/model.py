X_SERVER=True
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
        self.l2 = 0.02
        self.n_inputs = 15 ##15 obs on the roboschool hopper 11 observations for the hopper
        self.n_hidden = 4 # start small
        self.n_outputs = 3 # thigh, leg, foot joints
        self.initializer = tf.contrib.layers.variance_scaling_initializer()
        print("Env name {0}".format(args.envname))	
        self.env = gym.make(args.envname)
        self.render = args.render
        self.weights_path = "weights/nn_{0}_{1}_{2}".format(self.lr, self.l2,
self.n_hidden)


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
            sns.tsplot(time=range(steps), data=rewards, linestyle="-")
            plt.savefig("output/nn_performance_{}.png".format(steps))
        else:
            print("The reward {0} and loss {1} means".format(np.mean(rewards),
np.mean(losses)))


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
                batch_size = 1000

                # Do a run through of the current environment with the predicted
                # actions (exciting and is the part that we will be optimizing
                # and learning how to construct actions given an experts
                # roll-out)

               #TODO: need to run the batches on X_train[batch_num-1*batch_size:batch_num*batch_size]
                for batch_num in range(actions.shape[0]/batch_size):
                    frames = []
                    reward_cum = 0
                    for i in range(len(batch_size)):
                        obs, reward, done, info = self.env.step(actions)
                        reward_cum += reward
                        if (self.render and step % sample == 0) or step == n_max_steps -1 :
                            img = render_hopper(self.env, obs)
                            frames.append(img)
                        #if done:
                        #    print("We are done at step {0}".format(i))
                        #    break
                rewards.append(reward_cum)

                if step % verbose == 0:
                    print("Step {0}: Loss {1}, Reward {2}".format(step, np.mean(loss), reward_cum))
                    saver = tf.train.Saver()
                    if not os.path.exists(self.weights_path):
                        os.makedirs(self.weights_path)
                    saver.save(sess, self.weights_path)

                if X_SERVER and self.render and step % sample == 0:
                    animation = plot_animation(frames, repeat=True, step=step)
                    #plt.show()
                if step == n_max_steps - 1:
                    animation = plot_animation(frames, repeat=True, step=step)
            self.plot(steps=n_max_steps, rewards=rewards, losses=losses)


