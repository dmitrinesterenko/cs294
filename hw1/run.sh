#!/bin/bash
cmd="python ./run_expert.py --max_timesteps 1000 --num_rollouts 20 ./experts/Hopper-v1.pkl Hopper-v1"
#nohup $cmd & #> output/output.out 2>&1 < /dev/null &
$cmd
