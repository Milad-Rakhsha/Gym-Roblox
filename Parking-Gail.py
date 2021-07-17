import os
import numpy as np
# from gym_roblox.envs.DiscreteActions_ContinuousStates import DiscreteActions_ContinuousStates as RobloxEnv
from gym_roblox.envs.ContinuousActions_ContinuousStates import ContinuousActions_ContinuousStates as RobloxEnv
from gym_roblox.envs.GenerateExpertTraj import generate_expert_traj

from stable_baselines3 import A2C, SAC, PPO, TD3, HER, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.gail import generate_expert_traj

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
log_dir = "./Parking-GAIL/"
os.makedirs(log_dir, exist_ok=True)
env=Monitor(RobloxEnv(), log_dir)

print("begin recording expert")
# dataset = generate_expert_traj(env, log_dir+"/expertData",  n_episodes=5)
# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path=log_dir+'expertData.npz',
                        traj_limitation=5, batch_size=128)

model = PPO('MlpPolicy', env, verbose=1)
model.pretrain(dataset, n_epochs=1000)


checkpoint_callback = CheckpointCallback(save_freq=20000, save_path=log_dir,
                                         name_prefix='rl_model')

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

with ProgressBarManager(1e6) as ProgressBar: # this the garanties that the tqdm progress bar closes correctly
    model.learn(1e6,callback=[ProgressBar, checkpoint_callback])

model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
