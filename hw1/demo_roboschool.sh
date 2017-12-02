#!/bin/bash
set -eux
for e in Hopper-v1 Ant-v1 HalfCheetah-v1 Humanoid-v1 Reacher-v1 Walker2d-v1
do
    python run_expert.py experts/Roboschool$e.py $e --render --num_rollouts=1
done
