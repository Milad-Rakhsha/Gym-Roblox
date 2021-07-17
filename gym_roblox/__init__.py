import logging
from gym.envs.registration import register

register(
    id='ContinuousActions_ContinuousStates-v0',
    entry_point='gym_roblox.envs:ContinuousActions_ContinuousStates') # NAme of the CLASS after the colon

register(
    id='DiscreteActions_ContinuousStates-v0',
    entry_point='gym_roblox.envs:DiscreteActions_ContinuousStates') # NAme of the CLASS after the colon

register(
    id='ContinuousActions_ContinuousStates_Goal-v0',
    entry_point='gym_roblox.envs:ContinuousActions_ContinuousStates_Goal') # NAme of the CLASS after the colon
