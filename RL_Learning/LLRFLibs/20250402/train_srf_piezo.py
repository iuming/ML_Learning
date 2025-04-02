import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, ProgressBarCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rlo_piezo import SRFCavityEnv

def make_env(render_mode=None):
    """Create and wrap the environment"""
    env = SRFCavityEnv(render_mode=render_mode)
    env = Monitor(env)
    return env

def main():
    # Create output directory
    log_dir = "srf_piezo_training"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create and wrap the environment
    env = DummyVecEnv([lambda: make_env()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: make_env(render_mode="human")])
    eval_env = VecNormalize.load(f"{log_dir}/vec_normalize.pkl") if os.path.exists(f"{log_dir}/vec_normalize.pkl") else VecNormalize(eval_env)
    
    # Initialize the agent with a larger network
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 128],  # Policy network
                vf=[128, 128]   # Value network
            )
        ),
        verbose=1
    )
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/logs",
        eval_freq=1000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{log_dir}/checkpoints",
        name_prefix="ppo_srf_piezo"
    )
    
    # Train the agent
    total_timesteps = 1_000_000  # Adjust based on your needs
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, ProgressBarCallback()],
        progress_bar=True
    )
    
    # Save the final model and normalization stats
    model.save(f"{log_dir}/final_model")
    env.save(f"{log_dir}/vec_normalize.pkl")
    
    # Evaluate the final model
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=10,
        deterministic=True,
        render=True
    )
    
    print(f"Final evaluation results:")
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Close the environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    main() 