import logging,threading
import numpy as np
from gym import spaces
from gym_roblox.envs.RobloxVecBase import RobloxVecBaseEnv

class VecContinuousActions_ContinuousStates(RobloxVecBaseEnv):
    def __init__(self, serverPort):
        RobloxVecBaseEnv.__init__(self, serverPort)

    def initialize(self):
        init_result = self.rpc_client_proxy.initialize()

        self.num_envs = init_result["num_envs"]

        assert len(init_result["obs_info"]["high"]) == len(init_result["obs_info"]["low"])
        num_states=len(init_result["obs_info"]["low"])
        low =np.asarray(init_result["obs_info"]["low"])
        high =np.asarray(init_result["obs_info"]["high"])
        self.observation_space = spaces.Box(low=np.array(low), high=np.array(high), dtype=np.float32)
        self.states=np.zeros(num_states)
        # print(self.observation_space,self.states)

        assert len(init_result["ac_info"]["high"]) == len(init_result["ac_info"]["low"])
        num_actions=len(init_result["ac_info"]["low"])
        low =np.asarray(init_result["ac_info"]["low"])
        high =np.asarray(init_result["ac_info"]["high"])
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        self.action=np.zeros(num_actions)

        self.info =  {"timeout": init_result["maxSteps"]}
