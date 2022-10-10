'''Implementation of AMRL algorithm '''

######################################################
        ###     Intitialising stuff        ###
######################################################


import numpy as np
import gym
import matplotlib.pyplot as plt
import bottleneck as bn
import time as t
from scipy.signal import savgol_filter

import Baselines.AMRL_Agent as amrl
from AM_Gyms.AM_Env_wrapper import AM_ENV as wrapper
from AM_Gyms.AM_Env_wrapper import AM_Visualiser as visualiser
from AMRL_variant import AMRL_Variant_1
from AMRL_variant_redone import AMRL_Variant_2

        ### SETTING UP ENVIRONMENT ###
# Lake Envs
s_init = 0
MeasureCost = 0.01

# # Small Lake env, deterministic:
# env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
# StateSize, ActionSize = 16,4
# ENV_SmallLake_det = wrapper(env,StateSize,ActionSize,MeasureCost,s_init, True)

# # Small Lake env, non-deterministic
# env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
# StateSize, ActionSize = 16,4
# ENV_SmallLake_nondet = wrapper(env,StateSize,ActionSize,MeasureCost,s_init)

# Big Lake env, deterministic:
env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)
StateSize, ActionSize = 64,4
ENV_BigLake_det = wrapper(env,StateSize,ActionSize,MeasureCost,s_init, True)

# # Big Lake env, non-deterministic
# env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
# StateSize, ActionSize = 64,4
# ENV_BigLake_nondet = wrapper(env,StateSize,ActionSize,MeasureCost,s_init)

# # Taxi env
# env = gym.make('Taxi', )
# StateSize, ActionSize = 500, 4
# ENV_Taxi = wrapper(env, StateSize, ActionSize, MeasureCost, s_init = 0)

######################################################
        ###     Defining Agents        ###
######################################################

# Agent Variables:
Measurements= np.array([0,1])
MeasureSize = 2
eta = 0.1
nmbr_vars_recorded = 3

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

agent_var1 = AMRL_Variant_1(ENV_BigLake_det)
agent_var2 = AMRL_Variant_2(ENV_BigLake_det)
agent_nor = amrl.AMRL_Agent(ENV_BigLake_det)
agents = [agent_var2,agent_var1, agent_nor]
#agents = [agent_1, agent_2, agent_3, agent_4]

#legend = ["Bias = 0.1, Cost = 0.01 (Default)", "Bias = 0.05, Cost = 0.01", "Bias = 0.1, Cost = 0.2", "Bias = 0.05, Cost = 0.2"]
legend = ["AMRL-Agent Variant v2", "AMRL-Agent Variant v1", "Original AMRL-Agent"]

######################################################
        ###     Running Simulations       ###
######################################################

# Defining runs
nmbr_episodes = 500
nmbr_runs = 5
all_results = np.zeros((len(agents),nmbr_runs, nmbr_episodes, nmbr_vars_recorded))
all_avgs = np.zeros((len(agents), nmbr_episodes, nmbr_vars_recorded))

# Run-loop:
for a in range(len(agents)):
        t_start = t.perf_counter()
        print("Running agent {}...".format(a+1))
        thisAgent = agents[a]
        for i in range(nmbr_runs):
                t_this_start = t.perf_counter()
                (r_avg, all_results[a,i]) = thisAgent.run(nmbr_episodes, True)
                t_this_end = t.perf_counter()
                print("Run {0} done! (in {1} s)".format(i, t_this_end-t_this_start))
        all_avgs[a] = np.average(all_results[a], axis=0)
        print("Agent Done! ({0} runs, total of {1} s)\n\n".format(nmbr_runs, t.perf_counter()-t_start))


######################################################
        ###     Plotting Results        ###
######################################################
# Create Rolling Averages for plotting (this could be more efficiently done by setting axis correctly...)
print("Smooting out averages...")
window = 25
for a in range(len(agents)):
        for v in range(nmbr_vars_recorded):
                all_avgs[a,:,v] = savgol_filter(all_avgs[a,:,v], window, 5)
                

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

# Plotting Cumulative Rewards
plt.title("Cumulative episodic reward for AMRL-var in Lake Environment")
plt.ylabel("Cumulative Reward")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, np.cumsum(all_avgs[a,:,0]))

plt.legend(legend)
plt.savefig("AMRL-var_cumresults_redone_reward.png")
plt.clf()


# Plotting # Steps
plt.title("Average nmbr Steps per episode for AMRL var in Lake Environment")
plt.ylabel("# Steps taken")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,1])

plt.legend(legend)
plt.savefig("AMRL-var_results_redone_steps.png")
plt.clf()

# Plotting # measurements
plt.title("Average nmbr measurements per episode for AMRL var in Lake Environment")
plt.ylabel("# measurements taken")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,2])

plt.legend(legend)
plt.savefig("AMRL-var_results_redone_measurements.png")
plt.clf()

print("Done!")

# Plotting Reward + measureCosts
plt.title("Average reward (excluding costs) per episode for AMRL-var in Lake Environment")
plt.ylabel("Reward")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,0])
        plt.plot(x, all_avgs[a,:,0]+all_avgs[a,:,2]*MeasureCost)
        
legend = ["AMRL-Agent Variant v1, cost included", "AMRL-Agent Variant v1, cost excluded","AMRL-Agent Variant v2, cost included", "AMRL-Agent Variant v2, no excluded", "Original AMRL-Agent, cost included", "Original AMRL-Agent, cost excluded"]
plt.legend(legend)
plt.savefig("AMRL-var_results_noCost_redone_reward.png")
plt.clf()

# Plotting Reward + measureCosts
plt.title("Average reward (excluding costs) per episode for AMRL-var in Lake Environment")
plt.ylabel("Reward")

# Plotting cum. Reward + measurecosts
x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, np.cumsum(all_avgs[a,:,0]))
        plt.plot(x, np.cumsum(all_avgs[a,:,0]+all_avgs[a,:,2]*MeasureCost))
        
legend = ["AMRL-Agent Variant v1, cost included", "AMRL-Agent Variant v1, cost excluded","AMRL-Agent Variant v2, cost included", "AMRL-Agent Variant v2, no excluded", "Original AMRL-Agent, cost included", "Original AMRL-Agent, cost excluded"]
plt.legend(legend)
plt.savefig("AMRL-var_results_noCost_redone_reward.png")
plt.clf()