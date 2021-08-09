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
        self.reward = 0
        self.eps_reward = 0
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
            # observation= spaces.Box(low=np.full(num_states,-np.inf), high=np.full(num_states, np.inf), dtype=np.float32),
            # achieved_goal=spaces.Box(low=np.full(num_goal,-np.inf), high=np.full(num_goal, np.inf), dtype=np.float32),
            # desired_goal=spaces.Box(low=np.full(num_goal,-np.inf), high=np.full(num_goal, np.inf), dtype=np.float32)
            observation= spaces.Box(low=low_obs, high=high_obs, dtype=np.float32),
            achieved_goal=spaces.Box(low=low_ag, high=high_ag, dtype=np.float32),
            desired_goal=spaces.Box(low=low_dg, high=high_dg, dtype=np.float32)
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

        self.info =  {"timeout": self.data["maxSteps"],
                      "success_goal_reward": self.data["success_goal_reward"],
                      "reward_weights": np.asarray(self.data["reward_weights"]),
                      "scale": np.asarray(self.data["scale"]),
                      "p_norm":  self.data["p_norm"] ,
                       }

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
        self.reward = 0
        self.eps_reward = 0
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
        self.done=self.data["is_done"] or self.steps>=self.info["timeout"]
        # print(self.data)
        self.reward= self.compute_reward(self.states["achieved_goal"], self.states["desired_goal"], self.info)
        # self.reward=self.data["reward"]
        self.eps_reward += self.reward
        self.info['is_success'] = self._is_success(self.states['achieved_goal'], self.states['desired_goal'], self.info["success_goal_reward"])
        self.steps+=1
        return self.states, self.reward, self.done, self.info

    def _is_success(self, achieved_goal, desired_goal, cutoff):
        return self.compute_reward(np.atleast_2d(achieved_goal),np.atleast_2d(desired_goal),{}) > cutoff
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
        # test_rew=  self.data["rewards"]

        # d_eps=10/self.info["scale"][0]
        # o_eps=10
        # v_eps=10
        # w_eps=10
        achieved_goal=np.atleast_2d(achieved_goal)
        desired_goal= np.atleast_2d(desired_goal)
        d= np.sqrt(
        	np.power(achieved_goal[:,0]-desired_goal[:,0],2 )+
        	np.power(achieved_goal[:,1]-desired_goal[:,1],2 ))
        # o=np.sqrt(
        # 	np.power(achieved_goal[:,2]-desired_goal[:,2],2 )
        #     # +np.power(achieved_goal[:,3]-desired_goal[:,3],2 )
        #     )
        # v=np.sqrt(
        # 	np.power(achieved_goal[:,3]-desired_goal[:,3],2 )+
        # 	np.power(achieved_goal[:,4]-desired_goal[:,4],2 ))
        #
        # # w=np.sqrt(
        # # 	np.power(achieved_goal[:,4]-desired_goal[:,4],2 ))* self.info["reward_weights"][6]
        #
        #
        # reward= np.where(d/d_eps > 1, -1,
        #      self.info["reward_weights"][0] / np.maximum(d,0.001)
        #     +self.info["reward_weights"][2] / np.maximum(o,0.001)
        #     +self.info["reward_weights"][4] / np.maximum(v,0.001)
        # # +self.info["reward_weights"][6] / np.maximum(w,0.01)
        # )
        reward=-np.power(
                        np.dot(np.abs(achieved_goal - desired_goal), self.info["reward_weights"]),
                        self.info["p_norm"])
        # reward= np.where(d>0.1, -1, -1.0/np.minimum(reward,-0.1))
        # print("reward test: ",  reward[0] )
        return reward.squeeze()

    def get_ob(self):
        with self.cv:
            self.agentRequests.append({"command":"get_ob"})
            self.cv.wait()
            self.states=OrderedDict([
                    ('observation', np.asarray(self.data["observations"]["states"]) ),
                    ('achieved_goal', np.asarray(self.data["observations"]["achieved_goal"])),
                    ('desired_goal',np.asarray(self.data["observations"]["desired_goal"]))
                    ])

        return self.states

    def is_done(self):
         return self.done

    def render(self):
         pass
