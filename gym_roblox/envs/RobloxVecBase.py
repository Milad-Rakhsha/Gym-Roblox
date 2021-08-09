import os,sys,queue,threading
import numpy as np
from gym import Env, spaces
from collections import OrderedDict
from http.server import HTTPServer
from gym_roblox.envs.Server import MakeHandlerClassFromArgv
hostName = 'localhost'
serverPort = 8080

# from gym.vector.vector_env import VectorEnv as VecEnv
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper
import gym
from collections import OrderedDict
from typing import Any, Dict, List, Tuple
# from stable_baselines3.common.vec_env.base_vec_env import VecEnv
# from stable_baselines3.common.vec_env.util import DummyVecEnv
# from stable_baselines3.common.vec_env.util import  obs_space_info
# from stable_baselines3.common.preprocessing import  check_for_nested_spaces

from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import copy_obs_dict, dict_to_obs, obs_space_info

class RobloxBaseVecEnv(VecEnv):
    def __init__(self, serverPort):
        self.server=HTTPServer((hostName, serverPort), MakeHandlerClassFromArgv(self))
        self.serverThread = threading.Thread(target = self.server.serve_forever)
        self.serverThread.daemon = True
        self.serverThread.start()

        self.render_setup=False
        self.lock = threading.Lock()
        self.cv = threading.Condition()
        self.agentRequests=[]
        self.data=[]
        self.states=[]

        self.initialize()
        VecEnv.__init__(self, self.num_envs, self.observation_space, self.action_space )
        obs_space = self.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])

        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_steps= np.zeros((self.num_envs,), dtype=np.int)
        self.buf_infos = [OrderedDict([]) for _ in range(self.num_envs)]
        self.actions = None
        return

    def reset(self):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"reset"})
            self.cv.wait()
            self.states=np.asarray(self.data["observations"])
        for env_idx in range(self.num_envs):
            self._save_obs(env_idx, self.states[env_idx])
            self.buf_dones[env_idx]= False
            self.buf_steps[env_idx]= 0
        return self._obs_from_buf()

    def resetIdx(self, env_idx):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"resetSingleEnv", "Idx": env_idx})
            self.cv.wait()
        return np.asarray(self.data["observations"])

    def __step(self, ac):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"step", "actions": ac.tolist()})
            self.cv.wait()
        return  self.data

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        tmp_data=self.__step(self.actions)
                                # tmp_data["is_done"][env_idx] or self.steps[env_idx] > self.info["timeout"],\

        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =  \
                        np.asarray(tmp_data["observations"][env_idx]), \
                        tmp_data["rewards"][env_idx], \
                        tmp_data["is_done"][env_idx], \
                        tmp_data["info"][env_idx]
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.resetIdx(env_idx)

            self._save_obs(env_idx, obs)

        # print(self.buf_infos)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def get_ob(self):
        with self.cv:
            self.agentRequests.append({"command":"get_ob"})
            self.cv.wait()
        return np.asarray(self.data["observations"])

    def Render(self):
       pass

    def step_async(self, actions):
        self.actions = actions

    def close(self):
        return

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

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

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
