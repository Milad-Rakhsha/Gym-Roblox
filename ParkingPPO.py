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
num_envs = env.num_envs
n_steps = max_episode_length//10
rollout = num_envs * n_steps
batch_size = rollout
n_epochs= rollout//batch_size
log_interval= 1#max_episode_length//n_steps
print("n_steps=%d, rollout=%d, batch_size=%d, n_epochs=%d"%(n_steps, rollout, batch_size, n_epochs))

model = PPO('MlpPolicy', env=env,
            verbose=True,
            tensorboard_log=log_dir,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=0.0002,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            policy_kwargs=dict(net_arch=[128, 128, 128]),
            )


checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir,
                                         name_prefix='rl_model')

#If we need to change any model parameter we should do it after the load process
# model = PPO.load(log_dir + "/Final", verbose=True, tensorboard_log=log_dir, env=env)
# model.n_steps = n_steps
# model.rollout = rollout
# model.batch_size = batch_size
# model.n_epochs= n_epochs
# model.log_interval= log_interval



with ProgressBarManager(5e7) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(5e7, log_interval=log_interval,  callback=[ProgressBar, checkpoint_callback])
model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
