#!/bin/bash

# Define the environment names
env_names="CartPole-v0 CartPole-v1 Acrobot-v1 MountainCar-v0 Pendulum-v1"

# Loop over each environment name
for env_name in $env_names; do
    echo "Processing environment: $env_name"
    python test_dmd.py --env $env_name
done
