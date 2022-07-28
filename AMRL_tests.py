# Python file to run small tests:

# Imports from run-file:
import numpy as np
import gym
import matplotlib.pyplot as plt
import bottleneck as bn
import time as t
from scipy.signal import savgol_filter

import AMRL_Agent as amrl
from AM_Env_wrapper import AM_ENV as wrapper
from AM_Env_wrapper import AM_Visualiser as visualiser
from AMRL_variant_v2 import AMRL_v2
from AMRL_variant_v3 import AMRL_v3

# Test code:

# Lake Envs
s_init = 0
MeasureCost = 0.01

# Small Lake env, deterministic:
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
StateSize, ActionSize = 16,4

keep_going = True
r_tot = 0
for i in range(1):
    ENV = wrapper(env,StateSize,ActionSize,MeasureCost,s_init, True)
    ENV.reset()

    agent = AMRL_v3(ENV, nmbr_particles = 10)

    (r_avg, rewards,steps,ms) = agent.run(2500, True) 
    print(np.sum(ms))
    print(np.sum(steps) -np.sum(ms))  
    print(r_avg)
    r_tot += r_avg



    print ("failed to find optimal strategy")

    vis = visualiser(ENV, agent)
    vis.plot_choice_certainty()
    vis.plot_choice_density()
    vis.plot_choice_maxQ()
    vis.plot_choice_state_accuracy()
    print("Density, Most Common Choices & Accuracy")
    print (np.reshape(vis.density,  (4,4)))
    print (np.reshape(np.argmax(vis.choice ,axis=1) ,  (4,4)))
    print (np.reshape(vis.accuracy, (4,4)))
    keep_going = False