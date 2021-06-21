from gym_roblox.envs.RobloxPendulum import RobloxPendulum as Agent

if __name__ == '__main__':
    agent=Agent()
    for i in range(5):
        agent.reset()
        while(not agent.done):
            agent.step([0])
