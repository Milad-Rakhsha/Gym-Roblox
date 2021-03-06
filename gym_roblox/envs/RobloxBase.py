import numpy as np
from gym import Env, spaces
from collections import OrderedDict
import zmq
from tinyrpc import RPCClient
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from gym_roblox.envs.TcpTransport import TcpTransport

class RobloxBaseEnv(Env):
    def __init__(self, serverPort, delay):
        ctx = zmq.Context()
        self.rpc_client = RPCClient(JSONRPCProtocol(), TcpTransport(serverPort))
        self.rpc_client_proxy = self.rpc_client.get_proxy()
        self.data=[]
        self.delay= delay
        self.rew = 0
        self.steps= 0
        self.states=[]
        self.done=False
        return

    def initialize(self):
       raise NotImplementedError

    def reset(self):
        self.data = self.rpc_client_proxy.reset()
        self.states=np.asarray(self.data["observations"])
        self.done= False
        self.steps= 0
        return self.states

    def step(self, ac):
        self.data = self.rpc_client_proxy.step(ac.tolist())
        self.states=np.asarray(self.data["observations"])
        self.done=self.data["is_done"] or self.steps > self.info["timeout"]
        self.rew= self.data["reward"]
        self.steps+=1
        return self.states, self.rew, self.done, self.info

    def get_ob(self):
        self.step_data = self.rpc_client_proxy.get_ob()
        self.states=np.asarray(self.data["observations"])
        return self.states

    def is_done(self):
         return self.done

    def Render(self):
       pass

    def getStepFromExpert(self):
        self.step_data = self.rpc_client_proxy.getStepFromExpert()
        self.rew= self.data["reward"]
        self.states=np.asarray(self.data["observations"])
        self.action=np.asarray(self.data["actions"])
        self.done= self.data["is_done"]
        self.steps+= 1
        return self.states,self.action,self.rew,self.done

    def convert_observation_to_space(self, observation):
        if isinstance(observation, dict):
            space = spaces.Dict(OrderedDict([
                (key, self.convert_observation_to_space(value))
                for key, value in observation.items()
            ]))
        elif isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -float('inf'))
            high = np.full(observation.shape, float('inf'))
            space = spaces.Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)

        return space

    def _set_observation_space(self, observation):
        self.observation_space = self.convert_observation_to_space(observation)
        return self.observation_space

    def __del__(self):
        pass

    def __setstate__(self, state):
        self.__init__()
        return {self}

    def __getstate__(self):
        return {}
