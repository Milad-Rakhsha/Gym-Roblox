import os, sys, queue, threading
import numpy as np
from gym import Env, spaces
from collections import OrderedDict
from http.server import HTTPServer
from gym_roblox.envs.Server import MakeHandlerClassFromArgv

from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvStepReturn,
    VecEnvWrapper,
)
import gym
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)


import zmq
from tinyrpc import RPCClient
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from gym_roblox.envs.TcpTransport import TcpTransport


class RobloxVecBaseEnv(VecEnv):
    """
    An abstract asynchronous, vectorized environment.
    Running via json-rpc 2.0 with Roblox Studio

    :param num_envs: the number of environments
    :param num_envs: the number of environments
    :param observation_space: the observation space
    :param action_space: the action space
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, server_port):

        ctx = zmq.Context()
        self.rpc_client = RPCClient(JSONRPCProtocol(), TcpTransport(server_port))
        self.rpc_client_proxy = self.rpc_client.get_proxy()

        self.initialize()
        VecEnv.__init__(self, self.num_envs, self.observation_space, self.action_space)
        obs_space = self.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )

        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)

        self.buf_infos = [OrderedDict([]) for _ in range(self.num_envs)]
        self.actions = None

    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        self.data = self.rpc_client_proxy.reset()

        self.states = np.asarray(self.data["observations"])
        for env_idx in range(self.num_envs):
            self._save_obs(env_idx, self.states[env_idx])
            self.buf_dones[env_idx] = False

        return self._obs_from_buf()

    def step_async(self, actions: np.ndarray) -> None:
        self.step_data = self.rpc_client_proxy.step(actions.tolist())

    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information

        When using vectorized environments, the environments are automatically reset at the end of each episode.
        Thus, the observation returned for the i-th environment when done[i] is true will in fact be the first observation
        of the next episode, not the last observation of the episode that has just terminated. You can access the “real”
        final observation of the terminated episode—that is, the one that accompanied the done event provided by the underlying
        environment—using the terminal_observation keys in the info dicts returned by the vecenv.
        """


        self.buf_rews = np.asarray(self.step_data["rewards"])
        self.buf_dones = np.asarray(self.step_data["is_done"])
        self.buf_infos = np.asarray(self.step_data["info"])


        for env_idx in range(self.num_envs):
            obs = np.asarray(self.step_data["observations"][env_idx])
            self._save_obs(env_idx, obs)

        # print(self.buf_infos)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        return self.rpc_client_proxy.close()

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """
        Check if environments are wrapped with a given wrapper.
        """
        return [False for i in indices] if indices else [False]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        return self.rpc_client_proxy.seed(seed)
