import os,warnings
from typing import Dict
import numpy as np
from gym import spaces

def generate_expert_traj(env=None, save_path=None, n_episodes=20):
    """
    Train expert controller (if needed) and record expert trajectories.
    .. note::
        only Box and Discrete spaces are supported for now.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env:  The environment
    :param n_episodes: (int) Number of trajectories (episodes) to record
    :return: (dict) the generated expert trajectories.
    """
    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    idx = 0

    while ep_idx < n_episodes:
        # get action, obs, reward, done
        obs,action,reward,done= env.getStepFromExpert()
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward
        if env.is_done():
            obs = env.reset()
            episode_returns[ep_idx] = reward_sum
            ep_idx += 1
            print("Stored trajectory for episode: ", ep_idx, " expert reward: ", reward_sum)

    if isinstance(env.observation_space, spaces.Box):
        print(np.concatenate(observations))
        print(env.observation_space.shape)
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    # pytype: disable=attribute-error
    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }

    for key, val in numpy_dict.items():
        print(key, val.shape)
    # pytype: enable=attribute-error

    if save_path is not None:
        np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict
