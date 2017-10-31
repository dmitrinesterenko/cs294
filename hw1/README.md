# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, OpenAI Gym, Roboschool v1.1, 
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

## Perfomance
Check out output/nn_performance_{steps}.png for the results on the training from
the expert

Example loss and reward after 9600 training steps
Step 9600: Loss 0.446856677532, Reward -1784.92529398
#

Training with data from 4 rollouts and 500 steps in each rollout gives results of
a Loss 0.446852952242, Reward -1785.4013682 (not that good!)

