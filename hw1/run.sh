#!/bin/bash
cmd="python3 -m pdb ./run_expert.py --max_timesteps 1000 --num_rollouts 20 experts/RoboschoolHopper-v1.pkl RoboschoolHopper-v1"
#nohup $cmd & #> output/output.out 2>&1 < /dev/null &
$cmd
