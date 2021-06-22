import os
import numpy as np
from gym_roblox.envs.RobloxPendulum import RobloxPendulum as Agent
from stable_baselines3 import A2C, SAC, PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps

    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)

        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()

# Create log dir
log_dir = "./PPO-Pendulum/"
os.makedirs(log_dir, exist_ok=True)
env=Agent()
if(False):
    model = PPO.load(log_dir + "/rl_model_3000_steps", verbose=True)
    model.set_env(env)
else:
    # Create and wrap the environment
    env = Monitor(env, log_dir)
    model = PPO('MlpPolicy', env, verbose=1)


checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=log_dir,
                                         name_prefix='rl_model')


mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


with ProgressBarManager(5000) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(5000,callback=[ProgressBar, checkpoint_callback])

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

model.save(log_dir + "/Final")
