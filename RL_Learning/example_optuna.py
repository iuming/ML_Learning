"""
Module Name: example_optuna
Author: Ming Liu
Created Time: 2024-11-10

Description:
This module provides functionality to [describe what the module does].

Classes:
- ClassName: Brief description of the class.

Functions:
- function_name: Brief description of the function.

Usage:
Provide examples of how to use the module, classes, or functions if applicable.
"""

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.evaluation import evaluate_policy

def objective(trial):
    # 定义超参数的搜索空间
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512])
    n_epochs = trial.suggest_categorical('n_epochs', [10, 20, 30])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
    # n_layers = trial.suggest_int('n_layers', 1, 3)
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'leaky_relu'])
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
    ent_coef = trial.suggest_uniform('ent_coef', 0.0, 0.1)
    vf_coef = trial.suggest_uniform('vf_coef', 0.1, 1.0)
    max_grad_norm = trial.suggest_uniform('max_grad_norm', 0.1, 1.0)
    # policy_delay = trial.suggest_int('policy_delay', 1, 10)
    # target_update_interval = trial.suggest_int('target_update_interval', 1, 10)

    # 创建环境
    env = make_vec_env('CartPole-v1', n_envs=4)
    
    # 创建模型
    model = PPO('MlpPolicy', 
                env, 
                verbose=1, 
                device='cuda:0', 
                learning_rate=learning_rate, 
                n_steps=n_steps, 
                n_epochs=n_epochs,
                # n_layers=n_layers,
                # activation_fn=activation_fn,
                batch_size=batch_size,
                gamma=gamma,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm)
                # policy_delay=policy_delay,
                # target_update_interval=target_update_interval)
    
    # 训练模型
    model.learn(total_timesteps=10000)
    
    # 评估模型
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    
    # 清理环境
    del model
    env.close()
    
    # 返回评估结果作为目标值
    return mean_reward

# 创建一个 Optuna 研究对象
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=500)

# 打印最佳超参数
print('Best trial:')
trial = study.best_trial
print(f'  Value: {trial.value}')
print('  Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')