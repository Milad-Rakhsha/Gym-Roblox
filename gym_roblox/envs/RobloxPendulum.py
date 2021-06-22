import logging,threading
import numpy as np
from gym import spaces
from gym_roblox.envs.RobloxBase import RobloxBaseEnv

from http.server import HTTPServer
from gym_roblox.envs.Server import MakeHandlerClassFromArgv
hostName = 'localhost'
serverPort = 8080


class RobloxPendulum(RobloxBaseEnv):
    def __init__(self):
        RobloxBaseEnv.__init__(self)
        low = np.full(4, -10)
        high = np.full(4, 10)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-20.0, high=20.0, shape=(1,), dtype=np.float32)
        self.info =  {"timeout": 500}
        self.states=np.zeros(4)
        self.action=np.zeros(1)

        self.server=HTTPServer((hostName, serverPort), MakeHandlerClassFromArgv(self))
        self.serverThread = threading.Thread(target = self.server.serve_forever)
        self.serverThread.daemon = True
        self.serverThread.start()
        self.reset()

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

        self.rew= 1.0
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
