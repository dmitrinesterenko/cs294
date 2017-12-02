# CS294-112 HW 1: Imitation Learning

Dependencies: Python3 TensorFlow, OpenAI Gym, Roboschool v1.1
Apt install dependencies: python3-tk

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* RoboschoolAnt-v1.py
* RoboschoolHalfCheetah-v1.py
* RoboschoolHopper-v1.py
* RoboschoolHumanoid-v1.py
* RoboschoolReacher-v1.py
* RoboschoolWalker2d-v1.py

The name of the pickle file corresponds to the name of the gym environment.

## Installing
### Warning
I had a bear of a time to get Roboshool to work on on my laptop directly. I run Linux Ubuntu 16.04. Note: I use Anaconda 2 to run Python 2 and 3. I suspect that Anaconda is at the core of my issues as they all had to do with dependencies not being found. I have encountered both of the issues with [Boost](https://github.com/openai/roboschool/issues/23) and [OpenGL](https://github.com/openai/roboschool/issues/15). I have resolved OpenGL through the PATH tricks mentioned in the issue thread, I have never resolved the Boost issue.

### Working Version
What worked was getting [VirtualBox](https://www.virtualbox.org/) and starting with a brand new Ubuntu 16.04 (I matched the version that I run on the laptop to have better signals).

VirtualBox and following [Oleg's instructions](https://github.com/openai/roboschool) on Roboschool worked out and I was seeing hoppers and humanoids in under 20 minutes.

Note I did pursue a Docker container approach just to suss out if I could understand what the dependency interaction between OpenGL, Boost and Roboschool were but I did not pursue that very far because Docker wouldn't give me a X-windows which I wanted because I like to see my hoppers crawling before they are trained.

## Perfomance
Check out output/nn_performance_{steps}.png for the results on the training from
the expert

Example loss and reward after 9600 training steps
Step 9600: Loss 0.446856677532, Reward -1784.92529398
#

Training with data from 4 rollouts and 500 steps in each rollout gives results of
a Loss 0.446852952242, Reward -1785.4013682 (not that good!)

12/2/2017

With epoch training and data from 100 rollouts and 300 steps in each rollout
gives a loss mean of 0.33, reward of -840.
This gives about 28800 steps for each of the epoch executions. Ran this for 5-6
epochs however even with annealing the learning rate by 1.5 if the current
mean loss is not that much better than previous one (if no improvement over 0.01 of previous mean loss)
still get stuck in a rut of mean loss around 0.33.

Think that more steps may help (though it seems like we're just not training
the right things from the observations and actions given)

Also need: regularization (though overfitting is NOT currently the problem)
and drop out (though generalization is NOT currently the problem)

Next step: is to vary the optimizers and work more on annealing the learning
rate (seems to be the problem as the training is simply stuck in a rut as if the learning rate is too high)

Suspect a bug in my implementation that's of a more fundamental nature, for example
perhaps I am not really re-using weights between epochs.
