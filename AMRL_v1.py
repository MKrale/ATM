'''Implementation of AMRL algorithm '''


import numpy as np
import gym
import AMRL_Agent as amrl

        ### SETTING UP ENVIRONMENT ###

# Lake env:
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
env.reset()
Actions = np.arange(0,4)
States = np.arange(0,64)
s_init = 1

ActionSize = np.size(Actions)

StateSize = np.size(States)

### DEFINING AGENT  ###

# Agent Variables:
Measurements= np.array([0,1])
MeasureSize = 2
eta = 0.1

agent = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize)
agent.train_N(20, 1000)
#print(agent.QTable)
