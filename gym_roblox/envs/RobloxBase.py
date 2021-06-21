import os,sys,queue,threading
import numpy as np
from gym import Env, spaces
from collections import OrderedDict

class RobloxBaseEnv(Env):
   def __init__(self):
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

   def step(self, ac):
       raise NotImplementedError

   def reset(self):
       raise NotImplementedError

   def get_ob(self):
       raise NotImplementedError

   def is_done(self):
       raise NotImplementedError

   def ScreenCapture(self, interval):
       raise NotImplementedError

   def Render(self):
       raise NotImplementedError

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
