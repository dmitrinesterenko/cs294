"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/RoboschoolHumanoid-v1.py --render \
            --num_rollouts 20
"""
X_SERVER=True
import sys
import os
import pdb
import argparse
import pickle
import tensorflow as tf
import numpy as np
import gym
import roboschool
import IPython.display as display
import pandas as pd
try:
     import PIL.Image as Image
     import seaborn as sns
     import matplotlib.pyplot as plt
except ImportError:
     X_SERVER=False
     print("No X support")

from util import epoch, render_hopper, plot_environment, try_action, plot_animation
from model import Model
import importlib

def BatchGenerator(args):
    print('loading expert policy')
    module_name = args.expert_policy_file.replace('/', '.')
    if module_name.endswith('.py'):
        module_name = module_name[:-3]
    policy_module = importlib.import_module(module_name)
    print('loaded')
	# TRIAL
        #try_action(env, [0,0,1])
        #try_action(env, [0,1,0])
        #try_action(env, [1,0,0])
        #END TRIAL
    
    env, policy = policy_module.get_env_and_policy()
    print("The action space is {0}".format(env.action_space))

    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    # if args.max_timesteps > 1000:
    #	print('you will possibly falls off the edge of the world')
    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.act(obs)
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
	
    if X_SERVER:
        fig = plt.figure()
        sns.tsplot(time=range(args.num_rollouts), data=returns, color='b', linestyle=':')
        plt.title("Reward over time")
        #plt.show()
        plt.savefig("output/rewards_plt_{}.png".format(epoch()))

    expert_data = {'observations': np.array(observations),
                    'actions': np.array(actions), 
		   'env': env}

    yield np.array(observations), np.array(actions)


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
        #model.fit(X, y, batch_size=32, epoch=20, n_max_steps=100, sample=50, verbose=10)
        model.fit(X, y, batch_size=32, epoch=20, n_max_steps=1000, sample=500, verbose=100)


