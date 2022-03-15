'''Implementation of AMRL algorithm '''


import numpy as np
import gym
import AMRL_Agent as amrl
import matplotlib.pyplot as plt
import bottleneck as bn

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


# Defining runs
nmbr_episodes = 2500
nmbr_runs = 10
agent_1 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize)
agent_2 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize, m_bias = 0.05)
agent_3 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize,measureCost = 0.2)
agent_4 = amrl.AMRL_Agent(env, StateSize, MeasureSize, ActionSize,measureCost = 0.2, m_bias = 0.05)

agents = [agent_1, agent_2, agent_3, agent_4]
all_results = np.zeros((len(agents),nmbr_runs, nmbr_episodes, 2))
all_avgs = np.zeros((len(agents), nmbr_episodes, 2))

legend = ["Bias = 0.1, Cost = 0.01 (Default)", "Bias = 0.05, Cost = 0.01", "Bias = 0.1, Cost = 0.05", "Bias = 0.05, Cost = 0.1"]

# Run simulations
for a in range(len(agents)):
        print("Running agent {}...".format(a))
        thisAgent = agents[a]
        for i in range(nmbr_runs):s
                (r_avg, all_results[a,i]) = thisAgent.train_run(nmbr_episodes, True)
        all_avgs[a] = np.average(all_results[a], axis=0)
        print("Done!")

# Create Rolling Averages for plotting
print("Smooting out averages")
window = 10
for a in range(len(agents)):
        all_avgs[a] = bn.move_mean(all_avgs[a], window, axis=0)

# Plotting Rewards
plt.title("Average reward per episode for AMRL in Lake Environment")
plt.xlabel("Episode number")
plt.ylabel("Reward")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,0])

plt.legend(legend)
plt.savefig("AMRL_results_redone_reward.png")
plt.clf()
# Plotting # Steps
plt.title("Average nmbr Steps per episode for AMRL in Lake Environment")
plt.ylabel("# Steps taken")

x = np.arange(nmbr_episodes)
for a in range(len(agents)):
        plt.plot(x, all_avgs[a,:,1])

plt.legend(legend)
plt.savefig("AMRL_results_redone_steps.png")