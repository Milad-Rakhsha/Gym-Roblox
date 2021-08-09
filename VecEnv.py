from gym_roblox.envs.VecContinuousActions_ContinuousStates import VecContinuousActions_ContinuousStates as RobloxEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy
import numpy as np
import gym
import os
from stable_baselines3.common.callbacks import CheckpointCallback
from ProgressBar import ProgressBarCallback, ProgressBarManager
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

# Create log dir
log_dir = "./Vec-Pendulum/"
os.makedirs(log_dir, exist_ok=True)

# env = DummyVecEnv([lambda:gym.make('CartPole-v1') for i in range(3)])
# env=VecMonitor(env)


# env=RobloxEnv(8070)
# for i in range(2):
#     env.reset()
#     for i in range(3):
#         env.actions=np.asarray([[0],[0],[0]])
#         env.step_wait()
#         env.get_ob()
#
# # print(env.buf_infos)
# model = PPO(MlpPolicy, env, verbose=0)
#
# # Train the agent for 10000 steps
# model.learn(total_timesteps=10000)




env=VecMonitor(RobloxEnv(8070), log_dir)
max_episode_length=env.info["timeout"]

model = PPO('MlpPolicy', env=env,
            verbose=True,
            tensorboard_log=log_dir,
            n_steps=1,
            # learning_rate=0.0005,
            batch_size=400,
            # n_epochs=20,
            # gamma=0.995,
            # gae_lambda=0.2,
            # clip_range=0.05,
            # policy_kwargs=dict(net_arch=[128, 128, 128]),
            )


checkpoint_callback = CheckpointCallback(save_freq=100, save_path=log_dir,
                                         name_prefix='rl_model')

# model = PPO.load(log_dir + "/rl_model_235000_steps", verbose=True, tensorboard_log=log_dir, env=env)


with ProgressBarManager(1e5) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(1e5, log_interval=1,  callback=[ProgressBar, checkpoint_callback])

# model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
