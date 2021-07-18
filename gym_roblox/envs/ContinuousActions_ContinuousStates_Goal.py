import logging,threading
import numpy as np
from gym import spaces
from gym import GoalEnv, spaces
from collections import OrderedDict
from http.server import HTTPServer
from gym_roblox.envs.Server import MakeHandlerClassFromArgv
hostName = 'localhost'
serverPort = 8080


class ContinuousActions_ContinuousStates_Goal(GoalEnv):
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

        # print(self.data["obs_info"])
        num_states=len(self.data["obs_info"]["states"]["low"])
        num_goal=len(self.data["obs_info"]["desired_goal"]["low"])

        low_dg =np.asarray(self.data["obs_info"]["desired_goal"]["low"])
        high_dg =np.asarray(self.data["obs_info"]["desired_goal"]["high"])

        low_ag =np.asarray(self.data["obs_info"]["achieved_goal"]["low"])
        high_ag =np.asarray(self.data["obs_info"]["achieved_goal"]["high"])

        low_obs =np.asarray(self.data["obs_info"]["states"]["low"])
        high_obs =np.asarray(self.data["obs_info"]["states"]["high"])

        # print(num_states)
        # self.states=np.zeros(num_states)
        # print(self.observation_space.spaces)
        self.observation_space = spaces.Dict(dict(
            observation= spaces.Box(low=np.full(num_states,-np.inf), high=np.full(num_states, np.inf), dtype=np.float32),
            achieved_goal=spaces.Box(low=np.full(num_goal,-np.inf), high=np.full(num_goal, np.inf), dtype=np.float32),
            desired_goal=spaces.Box(low=np.full(num_goal,-np.inf), high=np.full(num_goal, np.inf), dtype=np.float32)
        ))
        self.states=OrderedDict([
                    ('observation', self.data["obs_info"]["states"] ),
                    ('achieved_goal',  self.data["obs_info"]["achieved_goal"] ),
                    ('desired_goal', self.data["obs_info"]["desired_goal"] )
                ])

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
        self.states=OrderedDict([
                    ('observation', np.asarray(self.data["observations"]["states"]) ),
                    ('achieved_goal', np.asarray(self.data["observations"]["achieved_goal"])),
                    ('desired_goal',np.asarray(self.data["observations"]["desired_goal"]))
                ])
        self.done= False
        self.steps= 0
        return self.states

    def step(self, ac):
        with self.cv:
            with self.lock:
                self.agentRequests.append({"command":"step", "actions": ac.tolist()})
            self.cv.wait()
            self.states=OrderedDict([
                        ('observation', np.asarray(self.data["observations"]["states"]) ),
                        ('achieved_goal', np.asarray(self.data["observations"]["achieved_goal"])),
                        ('desired_goal',np.asarray(self.data["observations"]["desired_goal"]))
                    ])

            self.done=self.data["is_done"] or self.steps>self.info["timeout"]

        self.rew= self.data["reward"]
        self.steps+=1
        return self.states, self.rew, self.done, self.info

    # for now the reward computation is performed here not on the Lua side
    def compute_reward(self, achieved_goal, desired_goal, info):
        # with self.cv:
        #     with self.lock:
        #         self.agentRequests.append({
        #         "command":"compute_reward",
        #         "achieved_goal": achieved_goal[None,0,:].tolist(),
        #         "desired_goal":  desired_goal[None,0,:].tolist()
        #         })
        #     self.cv.wait()
        # test= np.asarray(self.data["rewards"])
        d_eps=10
        o_eps=10
        v_eps=10
        w_eps=10
        d= np.sqrt(
        	np.power(achieved_goal[:,0]-desired_goal[:,0],2 )+
        	np.power(achieved_goal[:,1]-desired_goal[:,1],2 ))
        o=np.abs(achieved_goal[:,2]-desired_goal[:,2])

        v=np.sqrt(
        	np.power(achieved_goal[:,3]-desired_goal[:,3],2 )+
        	np.power(achieved_goal[:,4]-desired_goal[:,4],2 ))
        w=np.abs(achieved_goal[:,5]-desired_goal[:,5])

        # print("achieved_goal[0]: ",achieved_goal[0,:] )
        # print("desired_goal[0]: ",desired_goal[0,:] )

        reward= np.where(d>d_eps, -1, 10/np.maximum(d, 0.01) + 10/np.maximum(v,0.01) +10/np.maximum(o, 0.01) + 10/np.maximum(w,0.01))
                # +np.where(o>o_eps, -1, 0)+np.where(w>w_eps, -1, 0)
        # print("roblox: ", test, " python: ", reward[0])
        return reward

    def get_ob(self):
        with self.cv:
            self.agentRequests.append({"command":"get_ob"})
            self.cv.wait()
            self.states=OrderedDict([
                        ('observation', np.asarray(self.data["observations"]) ),
                        ('achieved_goal', np.asarray(self.data["achieved_goal"])),
                        ('desired_goal',np.asarray(self.data["desired_goal"]))
                    ])

        print(self.states)

        return self.states

    def is_done(self):
         return self.done

    def render(self):
         pass
