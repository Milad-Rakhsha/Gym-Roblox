import time
import numpy as np
import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from multiagent.multi_discrete import MultiDiscrete
from gym_roblox.envs.RobloxBase import RobloxBaseEnv
from gym_roblox.envs.RobloxObjects import *

class MultiAgent(RobloxBaseEnv):
    def __init__(self, serverPort, delay=0.0):
        RobloxBaseEnv.__init__(self, serverPort, delay)
        self.initialize()
        self.reset()

    def initialize(self):
        data = self.rpc_client_proxy.initialize()
        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.agents=[]
        self.num_adversaries=0
        self.dim_p= data["info"]["dim_p"]
        self.dim_c= data["info"]["dim_c"]
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = data["info"]["discrete_action"]
        self.initialize_agents(data)

        # if true, every agent has the same reward
        self.shared_reward = data["info"]["collaborative"]
        self.time = 0

        self.info =  {"timeout": data["maxSteps"]}

    def initialize_agents(self,data):
        for i in range(data["policy_agents"]["num_agents"]):
            agent = Agent()
            agent.blind = data["policy_agents"]["blind"][i]
            # agent.movable = data["policy_agents"]["movable"][i]
            agent.silent = data["policy_agents"]["silent"][i]
            # agent.u_noise = data["policy_agents"]["u_noise"][i]
            agent.u_range = data["policy_agents"]["u_range"][i]
            # agent.state.c = data["policy_agents"]["comm"][i]
            self.num_adversaries+=data["policy_agents"]["adversary"][i]
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)

            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(self.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(self.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = data["policy_agents"]["obs_dim"][i]
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.dim_c)
            self.agents.append(agent)

        # set required vectorized gym env property
        self.n = len(self.agents)

    def reset(self):
        obs_n = []
        time.sleep(self.delay * 2)
        self.data = self.rpc_client_proxy.reset()
        for i in range(self.n):
            obs_n.append(np.asarray(self.data["observations"][0][i]))
        return obs_n


    def step(self, action_n):
        self.steps+=1
        time.sleep(self.delay)
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        # set action for each agent
        action_dic= {
                    "u":[],
                    "c":[]
                    }
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
            action_dic["u"].append(agent.action.u.tolist())
            action_dic["c"].append(agent.action.c.tolist())


        self.data = self.rpc_client_proxy.step(action_dic)

        # record observation for each agent
        for i in range(self.n):
            obs_n.append(np.asarray(self.data["observations"][0][i]))
            reward_n.append(np.asarray(self.data["rewards"][0][i]))
            done_n.append(np.asarray(self.data["is_done"][0][i]))
            info_n['n'].append(np.asarray(self.data["info"][0][i]))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)

        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n,info_n


        # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.dim_p)
        agent.action.c = np.zeros(self.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]

            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n
