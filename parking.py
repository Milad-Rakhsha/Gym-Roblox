import gym
import highway_env
import numpy as np
# Agent
from stable_baselines3 import HerReplayBuffer, DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("parking-v0")
# Create the action noise object that will be used for exploration
n_actions = env.action_space.shape[0]
max_episode_length=100

noise_std = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
model = DDPG('MultiInputPolicy', env,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(
                            n_sampled_goal=4,
                            goal_selection_strategy= 'future',
                            online_sampling=True,
                            max_episode_length=100),
                    verbose=1, buffer_size=int(1e5),
                    learning_rate=1e-3, action_noise=action_noise,
                    gamma=0.99, batch_size=1024,
                    policy_kwargs=dict(net_arch=[512, 512, 512]),
                    device='cuda',
                    tensorboard_log="DDPG",
                    train_freq=(1,"episode"),
                    gradient_steps=max_episode_length+1,
                    )

# model = SAC('MultiInputPolicy', env,
#                     replay_buffer_class=HerReplayBuffer,
#                     replay_buffer_kwargs=dict(
#                             n_sampled_goal=4,
#                             goal_selection_strategy= 'future',
#                             online_sampling=True,
#                             max_episode_length=max_episode_length),
#                     verbose=1, buffer_size=int(1e6),
#                     learning_rate=1e-3,
#                     gamma=0.95, batch_size=1024,
#                     tau=0.05,
#                     # policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
#                     policy_kwargs=dict(net_arch=[128, 128, 128]),
#                     # normalize=True,
#                     tensorboard_log="SAC",
#                     train_freq=max_episode_length, gradient_steps=max_episode_length,
#                     )

# Train for 2e5 steps
model.learn(total_timesteps=5e5, log_interval=10,)
# Save the trained agent
model.save('her_ddpg_highway')
obs = env.reset()

# Evaluate the agent
episode_reward = 0
rewards = []
for n in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_reward += reward
    print(obs, reward, done, info, action)
    rewards.append(episode_reward)
    if done or info.get('is_success', False) or True:
        # print("Reward:", episode_reward, "Success?", info.get('is_success', False))
        print('\rEpisode {}\tReward: {:.2f}'.format(n, episode_reward, end=""), "\tSuccess?", info.get('is_success', False))
        episode_reward = 0.0
        obs = env.reset()
