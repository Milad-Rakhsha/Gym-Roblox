import logging
from gym.envs.registration import register

register(
    id='ContinuousActions_ContinuousStates-v0',
    entry_point='gym_roblox.envs:ContinuousActions_ContinuousStates') # NAme of the CLASS after the colon
