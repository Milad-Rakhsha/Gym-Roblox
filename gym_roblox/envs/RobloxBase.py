import os,sys,queue,threading
import numpy as np
from gym import Env, spaces
from collections import OrderedDict

from http.server import HTTPServer
from gym_roblox.envs.Server import MakeHandlerClassFromArgv
hostName = 'localhost'
serverPort = 8080

class RobloxBaseEnv(Env):
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
        self.rew = 0
        self.steps= 0
        self.states=[]
        self.done=False
        return

    def initialize(self):
       raise NotImplementedError

    def reset(self):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"reset"})
            self.cv.wait()
        self.done= False
        self.steps= 0
        return self.states

    def step(self, ac):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"step", "actions": ac.tolist()})
            self.cv.wait()
            self.states=np.asarray(self.data["observations"])
            self.done=self.data["is_done"] or self.steps > self.info["timeout"]

        self.rew= self.data["reward"]
        self.steps+=1

        return self.states, self.rew, self.done, self.info

    def get_ob(self):
        with self.cv:
            self.agentRequests.append({"command":"get_ob"})
            self.cv.wait()
            self.states=np.asarray(self.data["observations"])

        return self.states

    def is_done(self):
         return self.done

    def Render(self):
       pass

    def getStepFromExpert(self):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"getStepFromExpert"})
            self.cv.wait()

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
        if self.render_setup:
            print('Destructor called, Device deleted.')
        else:
            print('Destructor called, No device to delete.')

    def __setstate__(self, state):
        self.__init__()
        return {self}

    def __getstate__(self):
        return {}
