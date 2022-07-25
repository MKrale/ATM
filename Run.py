'''

File for running & gathering data on Active-Measuring algorithms.


'''

######################################################
        ###             Imports                 ###
######################################################

import numpy as np
import gym
import matplotlib.pyplot as plt
import bottleneck as bn
import time as t
import datetime
import json
import argparse
from scipy.signal import savgol_filter

import AMRL_Agent as amrl
from AM_Env_wrapper import AM_ENV as wrapper
from AM_Env_wrapper import AM_Visualiser as visualiser
from AMRL_variant_v2 import AMRL_v2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

######################################################
        ###       Parsing Arguments           ###
######################################################

parser = argparse.ArgumentParser("Run tests on Active Measuring Algorithms")

parser.add_argument('-algo'             , default = 'AMRL',             help='Algorithm to be tested')
parser.add_argument('-env'              , default = 'Lake_small_det',   help='Environment on which to perform the testing')
parser.add_argument('-m_cost'           , default = -1,                 help='Cost of measuring (default: use as specified by environment)')
parser.add_argument('-nmbr_eps'         , default = 500,                help='nmbr of episodes per run')
parser.add_argument('-nmbr_runs'        , default = 1,                  help='nmbr of runs to perform')
parser.add_argument('-plot'             , default = False,              help='Automatically plot data using ... (default: False)')
parser.add_argument('-f'                , default = None,               help='File name (default: generated automatically)')
parser.add_argument('-rep'              , default = './Data/',          help='Repository to store data (default: ./Data')

args            = parser.parse_args()
algo_name       = args.algo
env_name        = args.env
MeasureCost     = int(args.m_cost)
nmbr_eps        = int(args.nmbr_eps)
nmbr_runs       = int(args.nmbr_runs)
plot            = args.plot
file_name       = args.f
rep_name        = args.rep


######################################################
        ###     Intitialise Environment        ###
######################################################

# Lake Envs
s_init = 0
MeasureCost = args.m_cost
MeasureCost_Lake_default = 0.01
MeasureCost_Taxi_default = 0.01

match env_name:
        case "Lake_small_det":
                env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False)
                StateSize, ActionSize, s_init = 16,4,0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Lake_default
                ENV = wrapper(env,StateSize,ActionSize,MeasureCost,s_init, True)
        case "Lake_small_nondet":
                env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
                StateSize, ActionSize, s_init = 16,4,0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Lake_default
                ENV = wrapper(env,StateSize,ActionSize,MeasureCost,s_init)

        case "Lake_big_det":
                env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)
                StateSize, ActionSize, s_init = 64,4,0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Lake_default
                ENV = wrapper(env,StateSize,ActionSize,MeasureCost,s_init, True)
        case "Lake_big_nondet":
                env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True)
                StateSize, ActionSize, s_init = 64,4,0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Lake_default
                ENV = wrapper(env,StateSize,ActionSize,MeasureCost,s_init)


        case "Taxi":
                env = gym.make('Taxi', )
                StateSize, ActionSize, s_init = 500, 4, 0
                if MeasureCost == -1:
                        MeasureCost = MeasureCost_Taxi_default
                ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)

"""" 
To be added: 
        * Basic loss-env described in report
        * Chain-environment AMRL-paper
        * ACNO-settings?
"""

######################################################
        ###     Defining Agents        ###
######################################################

match algo_name:
        case "AMRL":
                agent = amrl.AMRL_Agent(ENV)
        case "AMRL_v2":
                agent = AMRL_v2(ENV)

"""" 
To be added: 
        * ACNO's
        * AMRL_v2 with different settings?
"""

######################################################
        ###     Exporting Results       ###
######################################################

if file_name == None:
        file_name = 'AMData_{}_{}_eps={}_runs={}_t={}'.format(algo_name, env_name, nmbr_eps, nmbr_runs, datetime.datetime.now().strftime("%d%m%Y%H%M%S"))

def PR_to_data(pr_time):
        return (datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=pr_time)).strftime("%d%m%Y%H%M%S")

def export_data(rewards, steps, measures,  t_start):
        with open(rep_name+file_name, 'w') as outfile:
                json.dump({
                        'parameters'            :vars(args),
                        'reward_per_eps'        :rewards,
                        'steps_per_eps'         :steps,
                        'measurements_per_eps'  :measures,
                        'all_avgs'              :all_avgs,
                        'start_time'            :PR_to_data(t_start),
                        'current_time'          :PR_to_data(t.perf_counter())
                }, outfile, cls=NumpyEncoder) #cls=NumpyEncoder?


######################################################
        ###     Running Simulations       ###
######################################################

rewards, steps, measures = np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps))
t_start = t.perf_counter()
print("""
Start running agent with following settings:
Algorithm: {}
Environment: {}
nmbr runs: {}
nmbr episodes per run: {}.
""".format(algo_name, env_name, nmbr_runs, nmbr_eps))

for i in range(nmbr_runs):
        t_this_start = t.perf_counter()
        (r_avg, rewards[i], steps[i], measures[i]) = agent.run(nmbr_eps, True)
        t_this_end = t.perf_counter()
        export_data(rewards[:i+1],steps[:i],measures[:i],t_start)
        print("Run {0} done with average reward {2}! (in {1} s)\n".format(i, t_this_end-t_this_start, r_avg))
print("Agent Done! ({0} runs, total of {1} s)\n\n".format(nmbr_runs, t.perf_counter()-t_start))

######################################################
        ###     Plotting Results (To be moved!)    ###
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