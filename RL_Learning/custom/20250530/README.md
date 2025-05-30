# Temperature Control with Reinforcement Learning
This project implements a reinforcement learning agent to control the temperature in a simulated room using Gymnasium and Stable Baselines3.

## Installation
Install the required dependencies:
pip install -r requirements.txt

## Usage
To train the DQN agent:

```bash
python train_dqn.py
```

To evaluate the trained agent and visualize the temperature control:

```bash
python evaluate_dqn.py
```

The evaluation script will display a plot of the temperature over time after running the agent for 100 steps.


### How It Works
- **Data Collection**: During evaluation, the script collects the temperature at each step in the `temperatures` list and the corresponding step numbers in the `steps` list.
- **Plotting**: After 100 steps, the `plot_temperature` function creates a line plot of the temperature over time, with a dashed red line indicating the target temperature for reference.
- **Output**: The plot is displayed immediately after evaluation, showing how well the agent controls the temperature.
