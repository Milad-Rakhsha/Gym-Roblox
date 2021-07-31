import logging,threading
import numpy as np
from gym import spaces
from gym_roblox.envs.RobloxBase import RobloxBaseEnv

class ContinuousActions_ContinuousStates(RobloxBaseEnv):
    def __init__(self, serverPort):
        RobloxBaseEnv.__init__(self, serverPort)
        self.initialize()
        self.reset()

    def initialize(self):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"initialize"})
            self.cv.wait()


        assert len(self.data["obs_info"]["states"]["high"]) == len(self.data["obs_info"]["states"]["low"])
        num_states=len(self.data["obs_info"]["states"]["low"])
        low =np.asarray(self.data["obs_info"]["states"]["low"])
        high =np.asarray(self.data["obs_info"]["states"]["high"])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.states=np.zeros(num_states)

        assert len(self.data["ac_info"]["high"]) == len(self.data["ac_info"]["low"])
        num_actions=len(self.data["ac_info"]["low"])
        low =np.asarray(self.data["ac_info"]["low"])
        high =np.asarray(self.data["ac_info"]["high"])
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        self.action=np.zeros(num_actions)

        self.info =  {"timeout": self.data["maxSteps"]}
