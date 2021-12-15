import os
from gym_roblox.envs.ContinuousActions_ContinuousStates import ContinuousActions_ContinuousStates as RobloxEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from ProgressBar import ProgressBarCallback, ProgressBarManager

import numpy as np

log_dir = "Pendulum/tensorboard/PPO"
os.makedirs(log_dir, exist_ok=True)

env=RobloxEnv(8844)
model = PPO(MlpPolicy, env, verbose=True)
mean_reward_before_train = evaluate_policy(model, env, n_eval_episodes=10)
checkpoint_callback = CheckpointCallback(save_freq=1, save_path=log_dir, name_prefix='rl_model')

with ProgressBarManager(2e2) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(total_timesteps=2e2, log_interval=1, callback=[ProgressBar, checkpoint_callback])

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
