import yaml
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.temperature_env import TemperatureControlEnv
from stable_baselines3 import DQN
from utils.plot_utils import plot_temperature

# Load environment and training configurations
with open("config/config_env.yaml", "r") as f:
    env_config = yaml.safe_load(f)["environment"]

with open("config/config_train.yaml", "r") as f:
    train_config = yaml.safe_load(f)["training"]

# Initialize the environment
env = TemperatureControlEnv(
    initial_temperature=env_config["initial_temperature"],
    target_temperature=env_config["target_temperature"],
    max_steps=env_config["max_steps"],
    temperature_change=env_config["temperature_change"],
    tolerance=env_config["tolerance"]
)

# Load the trained model
model_path = train_config["model_save_path"]
model = DQN.load(model_path)

# Test the trained model
print("\nTesting the trained agent...")
obs, _ = env.reset()
temperatures = []
steps = []
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

# Plot the temperature over time
plot_temperature(steps, temperatures, env.target_temperature)