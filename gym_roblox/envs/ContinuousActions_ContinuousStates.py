import logging,threading
import numpy as np
from gym import spaces
from gym_roblox.envs.RobloxBase import RobloxBaseEnv

from http.server import HTTPServer
from gym_roblox.envs.Server import MakeHandlerClassFromArgv
hostName = 'localhost'
serverPort = 8080


class ContinuousActions_ContinuousStates(RobloxBaseEnv):
    def __init__(self):
        RobloxBaseEnv.__init__(self)
        self.server=HTTPServer((hostName, serverPort), MakeHandlerClassFromArgv(self))
        self.serverThread = threading.Thread(target = self.server.serve_forever)
        self.serverThread.daemon = True
        self.serverThread.start()
        self.initialize()
        self.reset()

    def initialize(self):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"initialize"})
            self.cv.wait()

        assert len(self.data["obs_info"]["high"]) == len(self.data["obs_info"]["low"])
        num_states=len(self.data["obs_info"]["low"])
        low =np.asarray(self.data["obs_info"]["low"])
        high =np.asarray(self.data["obs_info"]["high"])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.states=np.zeros(num_states)

        assert len(self.data["ac_info"]["high"]) == len(self.data["ac_info"]["low"])
        num_actions=len(self.data["ac_info"]["low"])
        low =np.asarray(self.data["ac_info"]["low"])
        high =np.asarray(self.data["ac_info"]["high"])
        self.action_space = spaces.Box(low, high, dtype=np.float32)
        self.action=np.zeros(num_actions)

        self.info =  {"timeout": self.data["maxSteps"]}

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
            self.done=self.data["is_done"] or self.steps>self.info["timeout"]

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

    def render(self):
         pass

if __name__ == '__main__':
    env=RobloxPendulum()
    env.reset()
