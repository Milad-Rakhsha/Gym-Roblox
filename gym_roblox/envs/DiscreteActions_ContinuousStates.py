import logging,threading
import numpy as np
from gym import spaces
from gym_roblox.envs.RobloxBase import RobloxBaseEnv

class DiscreteActions_ContinuousStates(RobloxBaseEnv):
    def __init__(self, serverPort):
        RobloxBaseEnv.__init__(self, serverPort)
        self.initialize()
        self.reset()

    def initialize(self):
        self.step_data = self.rpc_client_proxy.initialize()
        assert len(self.data["obs_info"]["high"]) == len(self.data["obs_info"]["low"])
        num_states=len(self.data["obs_info"]["low"])
        low =np.asarray(self.data["obs_info"]["low"])
        high =np.asarray(self.data["obs_info"]["high"])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.states=np.zeros(num_states)

        self.action_space =spaces.MultiDiscrete([ 3,3 ])
        self.action=np.zeros(2)

        self.info =  {"timeout": self.data["maxSteps"]}
