
import os
import numpy as np
from gym_roblox.envs.ContinuousActions_ContinuousStates import ContinuousActions_ContinuousStates as RobloxEnv
from gym_roblox.envs.VecContinuousActions_ContinuousStates import VecContinuousActions_ContinuousStates as RobloxVecEnv

from stable_baselines3 import A2C, SAC, PPO, TD3, DQN, HerReplayBuffer, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from ProgressBar import ProgressBarCallback, ProgressBarManager

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.normalize import Normalize
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from shutil import copyfile

from stable_baselines3.common.env_checker import check_env

# Create log dir
log_dir = "./Parking-PPO/"
portNumber=8080
os.makedirs(log_dir, exist_ok=True)

# check_env(env)
filename="/rl_model_8080_478000_steps"
log_interval=1
env=VecMonitor(RobloxVecEnv(portNumber), log_dir)
env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=1.)
env = VecNormalize.load(log_dir + filename + "_vec_normalize.pkl", env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False
model = PPO.load(log_dir + filename, verbose=True, tensorboard_log=log_dir, env=env)
# check_env(env)


mean_reward, std_reward = evaluate_policy(model, env,  n_eval_episodes=1)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
