'''Implementation of AMRL algorithm '''

######################################################
        ###     Intitialising stuff        ###
######################################################


import numpy as np
import gym
import matplotlib.pyplot as plt
import bottleneck as bn
import time as t

import AMRL_Agent as amrl
import AMRL_variant as amrlv
        ### SETTING UP ENVIRONMENT ###

# Lake env:
# TODO: try this for different environments, too!
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
env.reset()
Actions = np.arange(0,4)
States = np.arange(0,16)
s_init = 1

ActionSize = np.size(Actions)

StateSize = np.size(States)

######################################################
        ###     Defining Agents        ###
######################################################

# Agent Variables:
Measurements= np.array([0,1])
MeasureSize = 2
eta = 0.1

#Normal AMRL-agents:
#agent_1 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize)
#agent_2 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize, m_bias = 0.05)
#agent_3 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize,measureCost = 0.2)
#agent_4 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize,measureCost = 0.2, m_bias = 0.05)

#AMRL-variants mk1
# agent_1 = amrlv.AMRL_Variant_1(env, StateSize, MeasureSize, ActionSize)
# agent_2 = amrlv.AMRL_Variant_1(env, StateSize, MeasureSize, ActionSize, m_bias = 0.05)
# agent_3 = amrlv.AMRL_Variant_1(env, StateSize, MeasureSize, ActionSize,measureCost = 0.2)
# agent_4 = amrlv.AMRL_Variant_1(env, StateSize, MeasureSize, ActionSize,measureCost = 0.2, m_bias = 0.05)

agent_var = amrlv.AMRL_Variant_1(env, StateSize, MeasureSize, ActionSize)
agent_nor = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize)
agents = [agent_var, agent_nor]
#agents = [agent_1, agent_2, agent_3, agent_4]

#legend = ["Bias = 0.1, Cost = 0.01 (Default)", "Bias = 0.05, Cost = 0.01", "Bias = 0.1, Cost = 0.2", "Bias = 0.05, Cost = 0.2"]
legend = ["AMRL-Agent Variant", "Original AMRL-Agent"]

######################################################
        ###     Running Simulations       ###
######################################################

# Defining runs
nmbr_episodes = 5000
nmbr_runs = 25
all_results = np.zeros((len(agents),nmbr_runs, nmbr_episodes, 3))
all_avgs = np.zeros((len(agents), nmbr_episodes, 3))

# Run-loop:
for a in range(len(agents)):
        t_start = t.perf_counter()
        print("Running agent {}...".format(a+1))
        thisAgent = agents[a]
        for i in range(nmbr_runs):
                (r_avg, all_results[a,i]) = thisAgent.train_run(nmbr_episodes, True)
                print("Run {0} done!".format(i))
        all_avgs[a] = np.average(all_results[a], axis=0)
        print("Agent Done! ({0} runs, total of {1} ms)\n\n".format(nmbr_runs, t.perf_counter()-t_start))


######################################################
        ###     Plotting Results        ###
######################################################

# Create Rolling Averages for plotting
print("Smooting out averages...")
window = 10
for a in range(len(agents)):
        all_avgs[a] = bn.move_mean(all_avgs[a], window, axis=0)

plt.xlabel("Episode number")
print("Plotting Graphs...")
# Plotting Rewards
plt.title("Average reward per episode for AMRL-var in Lake Environment")
plt.ylabel("Reward")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,0])

plt.legend(legend)
plt.savefig("AMRL-var_results_redone_reward.png")
plt.clf()

# Plotting # Steps
plt.title("Average nmbr Steps per episode for AMRL var in Lake Environment")
plt.ylabel("# Steps taken")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,1])

plt.legend(legend)
plt.savefig("AMRL-var_results_redone_steps.png")

# Plotting # Steps
plt.title("Average nmbr measurements per episode for AMRL var in Lake Environment")
plt.ylabel("# measurements taken")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,2])

plt.legend(legend)
plt.savefig("AMRL-var_results_redone_measurements.png")

print("Done!")