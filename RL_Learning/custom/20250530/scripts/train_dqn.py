import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from envs.temperature_env import TemperatureControlEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Load environment configuration
with open("config/config_env.yaml", "r") as f:
    env_config = yaml.safe_load(f)["environment"]

# Load training configuration
with open("config/config_train.yaml", "r") as f:
    train_config = yaml.safe_load(f)["training"]

# Initialize environment with parameters from config
env = TemperatureControlEnv(
    initial_temperature=env_config["initial_temperature"],
    target_temperature=env_config["target_temperature"],
    max_steps=env_config["max_steps"],
    temperature_change=env_config["temperature_change"],
    tolerance=env_config["tolerance"]
)

# Check environment compatibility
check_env(env)

# Initialize DQN model with parameters from config
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=train_config["learning_rate"],
    buffer_size=train_config["buffer_size"],
    device=train_config["train_device"]
)

# Train the model
print("Training the DQN agent...")
model.learn(total_timesteps=train_config["total_timesteps"])

# Save the trained model
model.save(train_config["model_save_path"])
print(f"Model saved as '{train_config['model_save_path']}.zip'")

# Test the trained model
print("\nTesting the trained agent...")
obs, _ = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, _ = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()