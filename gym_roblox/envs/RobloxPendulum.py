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
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(1,), dtype=np.float32)
        self.info =  {"timeout": 100000}

        self.server=HTTPServer((hostName, serverPort), MakeHandlerClassFromArgv(self))
        self.serverThread = threading.Thread(target = self.server.serve_forever)
        self.serverThread.daemon = True
        self.serverThread.start()

    def reset(self):
        with self.cv:
            logging.debug('waiting for roblox to reset the env')
            with self.lock:
                self.agentRequests.append({"command":"reset"})
            self.cv.wait()
            logging.debug('reset done')
        self.done= False
        self.steps= 0
        return self.states

    def step(self, ac):
        # action=float(ac[0])
        # self.ac = chrono.ChFunction_Const(action)
        # self.rev_pend_sys.DoStepDynamics(self.timestep)
        with self.cv:
            logging.debug('waiting for roblox to make a step')
            with self.lock:
                self.agentRequests.append({"command":"step", "actions": ac})
            self.cv.wait()
            logging.debug('Step Done')
            self.states=self.data["observations"]
            self.done=self.data["is_done"]

        self.rew= 1.0
        self.steps+=1

        return self.states, self.rew, self.done

    def get_ob(self):
        with self.cv:
            logging.debug('waiting for roblox to provid env obs')
            self.agentRequests.append({"command":"get_ob"})
            self.cv.wait()
            self.states=self.data["observations"]
            logging.debug('get_ob done')

        return np.asarray(self.states)

    def is_done(self):
         return done

    def render(self):
         pass

if __name__ == '__main__':
    env=RobloxPendulum()
    env.reset()
