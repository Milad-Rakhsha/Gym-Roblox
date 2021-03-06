import os
from ProgressBar import ProgressBarCallback, ProgressBarManager
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.envs import BitFlippingEnv
from stable_baselines3.common.vec_env import DummyVecEnv


model_class = TD3  # works also with SAC, DDPG and TD3
N_BITS = 15

env = BitFlippingEnv(n_bits=N_BITS, continuous=model_class in [DDPG, SAC, TD3], max_steps=N_BITS)

# Available strategies (cf paper): future, final, episode
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# If True the HER transitions will get sampled online
online_sampling = True
# Time limit for the episodes
max_episode_length = N_BITS

# Initialize the model
model = model_class(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    # Parameters for HER
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy=goal_selection_strategy,
        online_sampling=online_sampling,
        max_episode_length=max_episode_length,
    ),
    verbose=1,
)

# Train the model
model.learn(10000)
log_dir = "./HER-BitFlip/"
# checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
#                                          name_prefix='rl_model')
#
# with ProgressBarManager(10000) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
#     model.learn(10000,callback=[ProgressBar, checkpoint_callback])

# Create log dir
os.makedirs(log_dir, exist_ok=True)
# model.save("./her_bit_env")
model.save(log_dir + "/her_bit_env")

# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
# model = model_class.load('./her_bit_env', env=env)

obs = env.reset()
print(obs)
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = env.step(action)
    if done:
        print(obs, reward)
        obs = env.reset()
