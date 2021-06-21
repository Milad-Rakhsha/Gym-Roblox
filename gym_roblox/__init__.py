import logging
from gym.envs.registration import register

register(
    id='roblox_pendulum-v0',
    entry_point='gym_roblox.envs:RobloxPendulum') # NAme of the CLASS after the colon
