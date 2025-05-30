Temperature Control with Reinforcement Learning
This project implements a reinforcement learning agent to control the temperature in a simulated room using Gymnasium and Stable Baselines3.
Installation
Install the required dependencies:
pip install -r requirements.txt

Usage
To train the DQN agent, run the training script from the temperature_control directory:
python train_dqn.py

The trained model will be saved as dqn_temperature_control.zip. After training, the script will automatically test the agent by running a simulation for 100 steps, rendering the temperature at each step.
Configuration
Parameters for the environment and training process are stored in YAML files within the config/ directory:

config/config_env.yaml: Environment parameters (e.g., initial temperature, target temperature).
config/config_train.yaml: Training parameters (e.g., learning rate, buffer size).

To modify parameters, edit the corresponding YAML file before running the training script.
Example: Modifying Target Temperature
To change the target temperature to 25°C, update config/config_env.yaml:
environment:
  initial_temperature: 20.0
  target_temperature: 25.0  # Changed from 22.0
  max_steps: 100
  temperature_change: 1.0
  tolerance: 0.5

