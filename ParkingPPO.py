import os
import numpy as np
from gym_roblox.envs.ContinuousActions_ContinuousStates import ContinuousActions_ContinuousStates as RobloxEnv
from gym_roblox.envs.VecContinuousActions_ContinuousStates import VecContinuousActions_ContinuousStates as RobloxVecEnv

from stable_baselines3 import A2C, SAC, PPO, TD3, DQN, HerReplayBuffer, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from ProgressBar import ProgressBarCallback, ProgressBarManager

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor




# Create log dir
log_dir = "./Parking-PPO/"
os.makedirs(log_dir, exist_ok=True)
env=VecMonitor(RobloxVecEnv(8060), log_dir)
max_episode_length=env.info["timeout"]
num_envs=env.num_envs
n_steps = max_episode_length
rollout = num_envs * n_steps
batch_size = num_envs
n_epochs= rollout//batch_size

print("n_steps=%d, rollout=%d, batch_size=%d, n_epochs=%d"%(n_steps, rollout, batch_size, n_epochs))

model = PPO('MlpPolicy', env=env,
            verbose=True,
            tensorboard_log=log_dir,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=0.0001,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[128, 128, 128]),
            )


checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir,
                                         name_prefix='rl_model')

# model = PPO.load(log_dir + "/rl_model_50000_steps", verbose=True, tensorboard_log=log_dir, env=env)

with ProgressBarManager(2e6) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(2e6, log_interval=1,  callback=[ProgressBar, checkpoint_callback])

# model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
