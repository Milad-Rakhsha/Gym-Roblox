import os
import numpy as np
from gym_roblox.envs.ContinuousActions_ContinuousStates import ContinuousActions_ContinuousStates as RobloxEnv
from gym_roblox.envs.VecContinuousActions_ContinuousStates import VecContinuousActions_ContinuousStates as RobloxVecEnv

from stable_baselines3 import A2C, SAC, PPO, TD3, DQN, HerReplayBuffer, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from ProgressBar import ProgressBarCallback, ProgressBarManager

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from shutil import copyfile

def copyCurrentSource(log_dir):
    import glob
    dirs_name=sorted(glob.glob(log_dir+"/DDPG_*"))
    max_run=0
    for d in dirs_name : max_run=max(int(d[len(log_dir)+4:]),max_run)
    copyfile("ParkingDDPG.py", log_dir + "/" + str(max_run+1) + ".py")



# Create log dir
log_dir = "./Parking-DDPG/"
os.makedirs(log_dir, exist_ok=True)
env=Monitor(RobloxEnv(8070), log_dir)
max_episode_length=env.info["timeout"]
num_envs = 1
n_steps = 50 # max_episode_length//10
rollout = num_envs * n_steps
batch_size = num_envs
n_epochs= 20 #rollout//batch_size * 5
log_interval= max_episode_length//n_steps//10


n_actions = env.action_space.shape[0]
noise_std = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
model = DDPG('MultiInputPolicy', env,
                    # replay_buffer_class=HerReplayBuffer,
                    # replay_buffer_kwargs=dict(
                    #         n_sampled_goal=4,
                    #         goal_selection_strategy= 'future',
                    #         online_sampling=True,
                    #         max_episode_length=max_episode_length),
                    verbose=1,
                    action_noise=action_noise,
                    learning_rate=1e-4,
                    gamma=0.99,
                    tau=0.005,
                    buffer_size=int(1e6),
                    learning_starts=max_episode_length*1,
                    batch_size=2048,
                    policy_kwargs=dict( net_arch=[dict(pi=[128, 128, 128], vf=[128, 128, 128])]),
                    tensorboard_log=log_dir,
                    train_freq=max_episode_length//50, gradient_steps=n_epochs,
                    )

checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir,
                                         name_prefix='rl_model')
print("max_episode_length=%d, n_steps=%d, rollout=%d, batch_size=%d, n_epochs=%d"%(max_episode_length, n_steps, rollout, batch_size, n_epochs))

#If we need to change any model parameter we should do it after the load process
# model = PPO.load(log_dir + "/rl_model_2000000_steps", verbose=True, tensorboard_log=log_dir, env=env)
# model.n_steps = n_steps
# model.rollout = rollout
# model.batch_size = batch_size
# model.n_epochs= n_epochs
# model.log_interval= log_interval
# model.learning_rate=0.0001
# model.gae_lambda=0.7

copyCurrentSource(log_dir)
with ProgressBarManager(5e7) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(5e7, log_interval=log_interval,  callback=[ProgressBar, checkpoint_callback])
model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
