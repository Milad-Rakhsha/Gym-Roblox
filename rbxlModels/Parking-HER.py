import os
import numpy as np
# from gym_roblox.envs.DiscreteActions_ContinuousStates import DiscreteActions_ContinuousStates as RobloxEnv
from gym_roblox.envs.ContinuousActions_ContinuousStates_Goal import ContinuousActions_ContinuousStates_Goal as RobloxEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3 import A2C, SAC, PPO, TD3, DQN, HerReplayBuffer, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from ProgressBar import ProgressBarCallback, ProgressBarManager
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from tqdm.auto import tqdm



# Create log dir
log_dir = "./Parking-HER/"
os.makedirs(log_dir, exist_ok=True)
env=Monitor(RobloxEnv(), log_dir)
max_episode_length=env.info["timeout"]

# env = DummyVecEnv([lambda: env])
# env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=150.)

# model = SAC('MultiInputPolicy', env,
#                     replay_buffer_class=HerReplayBuffer,
#                     replay_buffer_kwargs=dict(
#                             n_sampled_goal=4,
#                             goal_selection_strategy= 'future',
#                             online_sampling=True,
#                             max_episode_length=max_episode_length),
#                     verbose=1, buffer_size=int(1e6),
#                     learning_rate=1e-4,
#                     gamma=0.95, batch_size=1024,
#                     tau=0.05,
#                     # policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
#                     policy_kwargs=dict(net_arch=[128, 128, 128]),
#                     # normalize=True,
#                     tensorboard_log=log_dir,
#                     train_freq=max_episode_length, gradient_steps=100,
#                     )


n_actions = env.action_space.shape[0]
noise_std = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
model = DDPG('MultiInputPolicy', env,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy= 'future',
                            online_sampling=True,
                            max_episode_length=max_episode_length),
                    verbose=1,
                    action_noise=action_noise,
                    learning_rate=1e-3,
                    gamma=0.99,
                    tau=0.005,
                    buffer_size=int(1e5),
                    learning_starts=max_episode_length*10,
                    batch_size=1024,
                    policy_kwargs=dict(net_arch=[128, 128, 128]),
                    tensorboard_log=log_dir,
                    train_freq=max_episode_length, gradient_steps=max_episode_length+1,
                    )

# model = DDPG.load(log_dir + "/rl2_model_500000_steps", verbose=True, tensorboard_log=log_dir, env=env)
# model.replay_buffer_kwargs=dict(
#                                 n_sampled_goal=20,
#                                 goal_selection_strategy= 'future',
#                                 online_sampling=True,
#                                 max_episode_length=max_episode_length)


checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir,
                                         name_prefix='rl_model')

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=2)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

with ProgressBarManager(2e5) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(total_timesteps=2e5, log_interval=10, callback=[ProgressBar, checkpoint_callback])
# model.learn(1000)
model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
