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

# Test code:

# Lake Envs
s_init = 0
MeasureCost = 0.01

# Small Lake env, deterministic:
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
StateSize, ActionSize = 64,4

keep_going = True
r_tot = 0
for i in range(10):
    ENV = wrapper(env,StateSize,ActionSize,MeasureCost,s_init, True)
    ENV.reset()

    agent_var2 = AMRL_v2(ENV, nmbr_particles = 10)

    (r_avg, rewards,steps,ms) = agent_var2.run(2000, True) 
    print(np.sum(ms))
    print(np.sum(steps) -np.sum(ms))  
    print(r_avg)
    r_tot += r_avg
print (r_tot/(i+1))



    # if r_avg < 200:
    #     print ("failed to find optimal strategy")

    #     vis = visualiser(ENV, agent_var2)
    #     vis.plot_choice_certainty()
    #     vis.plot_choice_density()
    #     vis.plot_choice_maxQ()
    #     vis.plot_choice_state_accuracy()
    #     print("Density, Most Common Choices & Accuracy")
    #     print (np.reshape(vis.density,  (4,4)))
    #     print (np.reshape(np.argmax(vis.choice ,axis=1) ,  (4,4)))
    #     print (np.reshape(vis.accuracy, (4,4)))
    #     keep_going = False