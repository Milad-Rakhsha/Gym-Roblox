import logging,threading
import numpy as np
from gym import spaces
from gym_roblox.envs.RobloxVecBase import RobloxBaseVecEnv

class VecContinuousActions_ContinuousStates(RobloxBaseVecEnv):
    def __init__(self, serverPort):
        RobloxBaseVecEnv.__init__(self, serverPort)
        self.reset()

    def initialize(self):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"initialize"})
            self.cv.wait()


        self.num_envs = self.data["num_envs"]

        assert len(self.data["obs_info"]["high"]) == len(self.data["obs_info"]["low"])
        num_states=len(self.data["obs_info"]["low"])
        low =np.asarray(self.data["obs_info"]["low"])
        high =np.asarray(self.data["obs_info"]["high"])
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.states=np.zeros(num_states)
        # print(self.observation_space,self.states)

        assert len(self.data["ac_info"]["high"]) == len(self.data["ac_info"]["low"])
        num_actions=len(self.data["ac_info"]["low"])
        low =np.asarray(self.data["ac_info"]["low"])
        high =np.asarray(self.data["ac_info"]["high"])
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        self.action=np.zeros(num_actions)

        self.info =  {"timeout": self.data["maxSteps"]}
