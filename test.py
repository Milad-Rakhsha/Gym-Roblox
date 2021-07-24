from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
model_class = DQN# DQN  # works also with SAC, DDPG and TD3
N_BITS = 15

env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = N_BITS

# Initialize the model
# model = model_class(
#     "MultiInputPolicy",
#     env,
#     replay_buffer_class=HerReplayBuffer,
#     # Parameters for HER
#     replay_buffer_kwargs=dict(
#         n_sampled_goal=4,
#         goal_selection_strategy=goal_selection_strategy,
#         online_sampling=online_sampling,
#         max_episode_length=max_episode_length,
#     ),
#     verbose=1,
#     learning_rate=1e-3,
#     train_freq=1,
#     learning_starts=100,
#     exploration_final_eps=0.02,
#     target_update_interval=500,
#     # seed=0,
#     batch_size=32,
#     buffer_size=int(1e5),
# )

model = model_class('MultiInputPolicy', env,
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
                    target_update_interval=500,
                    learning_starts=0,
                    exploration_final_eps=0.02,
                    # policy_kwargs=dict(n_critics=2, net_arch=[512, 512, 512]),
                    # policy_kwargs=dict(net_arch=[128, 128, 128]),
                    # normalize=True,
                    train_freq=max_episode_length,
                    gradient_steps=10,
                    tensorboard_log="test",
                    )

# Train the model
model.learn(total_timesteps=20000, log_interval=100)

model.save("./her_bit_env")
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load('./her_bit_env', env=env)


# 90% training success
assert np.mean(model.ep_success_buffer) > 0.90
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)

    if done:
        print(obs)
        obs = env.reset()
