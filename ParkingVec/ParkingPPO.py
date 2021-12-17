import os
import numpy as np
from gym_roblox.envs.ContinuousActions_ContinuousStates import ContinuousActions_ContinuousStates as RobloxEnv
# from gym_roblox.envs.VecContinuousActions_ContinuousStates import VecContinuousActions_ContinuousStates as RobloxVecEnv
from gym_roblox.envs.VecContinuousActions_ContinuousStates_Old import VecContinuousActions_ContinuousStates_Old as RobloxVecEnv

from stable_baselines3 import A2C, SAC, PPO, TD3, DQN, HerReplayBuffer, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from ProgressBar import ProgressBarCallback, ProgressBarManager
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from shutil import copyfile

env_dir = "ParkingVec/"

def copyCurrentSource(env_dir, log_dir):
    import glob
    dirs_name=sorted(glob.glob(log_dir+"/PPO_*"))
    max_run=0
    for d in dirs_name : max_run=max(int(d[len(log_dir)+4:]),max_run)
    copyfile(env_dir + "ParkingPPO.py", log_dir + "/" + str(max_run+1) + ".py")

class CustomCheckpointCallback(CheckpointCallback):
    def __init__(self, env, save_freq: int, save_path: str, name_prefix: str = "rl_model", verbose: int = 0):
        super(CustomCheckpointCallback,self).__init__(save_freq, save_path, name_prefix, verbose)
        self.env=env
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            stats_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps_vec_normalize.pkl")
            self.env.save(stats_path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True


portNumber=8070
name_prefix='rl_model_%s'%(portNumber)
# Create log dir
log_dir = env_dir + "PPO/"
os.makedirs(log_dir, exist_ok=True)

if(True):
    env=VecMonitor(RobloxVecEnv(portNumber), log_dir)
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                       clip_obs=1.)
    max_episode_length=env.info["timeout"]
    num_envs = env.num_envs
    n_steps = 20 # max_episode_length//10
    rollout = num_envs * n_steps
    batch_size = rollout
    n_epochs= 20 #rollout//batch_size * 5
    log_interval= max(max_episode_length//n_steps//2,1)
    print("max_episode_length=%d, n_steps=%d, rollout=%d, batch_size=%d, n_epochs=%d"%(max_episode_length, n_steps, rollout, batch_size, n_epochs))

    model = PPO('MlpPolicy', env=env,
                verbose=True,
                tensorboard_log=log_dir,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                learning_rate=0.0001,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.001,
                target_kl=0.02,
                policy_kwargs=dict( net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])])
                )
else:
    #If we need to change any model parameter we should do it after the load process
    filename="/rl_model_8080_4500000_steps"
    log_interval=5
    env= RobloxVecEnv(portNumber)
    # env=VecMonitor(RobloxVecEnv(portNumber), log_dir)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True,
    #                    clip_obs=1.)
    # env= VecNormalize.set_venv(venv=env)
    #the version that works
    env = VecNormalize.load(load_path=log_dir + filename + "_vec_normalize.pkl", venv=env)
    model = PPO.load(log_dir + filename, verbose=True, tensorboard_log=log_dir, env=env, print_system_info=True)
    print("PPO num_env: ", model.env.num_envs )


copyCurrentSource(env_dir, log_dir)
checkpoint_callback = CustomCheckpointCallback(env, save_freq=5000, save_path=log_dir,
                                                name_prefix=name_prefix)

with ProgressBarManager(5e7) as ProgressBar:
    model.learn(5e7, log_interval=log_interval,  callback=[ProgressBar, checkpoint_callback])
model.save(log_dir + "/Final")

# Evaluate the trained agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
