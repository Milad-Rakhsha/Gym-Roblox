import os
import numpy as np
from gym_roblox.envs.DiscreteActions_ContinuousStates_Goal import DiscreteActions_ContinuousStates_Goal as RobloxEnv
from ProgressBar import ProgressBarCallback, ProgressBarManager
from stable_baselines3 import A2C, SAC, PPO, TD3, DQN, HerReplayBuffer, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tqdm.auto import tqdm



# Create log dir
log_dir = "./Parking-HER/"
os.makedirs(log_dir, exist_ok=True)
env=Monitor(RobloxEnv(), log_dir)

model_class = DQN
max_episode_length=env.info["timeout"]

model = DQN('MultiInputPolicy', env,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy= 'future',
                            online_sampling=True,
                            max_episode_length=max_episode_length),
                    verbose=1,
                    buffer_size=int(1e6),
                    learning_rate=1e-3,
                    batch_size=1024,
                    gamma=0.95,
                    # tau=0.05,
                    target_update_interval=1000,
                    learning_starts=0,
                    exploration_final_eps=0.05,
                    # policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
                    # policy_kwargs=dict(net_arch=[128, 128, 128]),
                    # normalize=True,
                    train_freq=max_episode_length,
                    gradient_steps=200,
                    tensorboard_log=log_dir,
                    )

# model = HER.load(log_dir + "/rl_model_105000_steps", verbose=True, tensorboard_log=log_dir, env=env)
# model.env=env

checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
                                         name_prefix='rl_model')

with ProgressBarManager(1e5) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(total_timesteps=1e5, log_interval=10, callback=[ProgressBar, checkpoint_callback])
model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
