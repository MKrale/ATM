"""
File for reading & plotting from given data files.

(Only works if nmbr Steps is equal in all files!)
"""

# To be filled in!

Data_path       = "Data/Run1/"

alg_names = ["AMRL", "BAM-QMDP", "BAM-QMDP+"]
env_name        = "semi-slippery frozen lake environment"
Env_fileName    = "LakeBigNonDetv2"
ending_filename = "Run1.txt"
Files_legend    = ["AMRL-Q","BAM-QMDP","BAM-QMDP+" ]

max_eps = 1000 #10_000

#########################################################
#               Where the magic happens!
#########################################################

# creating names:
Files_to_read = []
nmbr_files = len(alg_names)

for i in range(nmbr_files):
    Files_to_read.append("{}_{}_{}".format(alg_names[i], Env_fileName, ending_filename))
    

nmbr_files      = len(Files_to_read)

# Imports
import json
import numpy as np
import math as m
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
w1, w2 = 250,2 #vars for smooting out plots
timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")

# Plot Naming

Plot_names_title    = ["Average return", "Average steps", "Average measurements", "Average cumulative reward"]
Plot_Y_Label        = ["Return per Episode", "Steps", "Measurements", "Reward (Cumulative)"]
Plot_names_file     = ["Reward", "Steps", "Measures", "cumReward"]

nmbr_plots          = len(Plot_names_title)

# Data to obtain:

nmbr_steps      = []
measure_cost    = []

avg_reward          = []
avg_cum_reward      = []
avg_steps           = []
avg_measures        = []
avg_reward_noCost   = []

nmbr_eps = []
nmbr_runs = []
measure_cost = []
# Read data:

for file_name in Files_to_read:
    with open(Data_path + file_name) as file:
        contentDict = json.load(file)
        
        avg_reward.append       (savgol_filter( np.average (contentDict["reward_per_eps"]      , axis=0), w1, w2 ))

        avg_steps.append        (savgol_filter(np.average (contentDict["steps_per_eps"]       , axis=0), w1, w2 ) )
        avg_measures.append     (savgol_filter(np.average (contentDict["measurements_per_eps"], axis=0), w1, w2 ) )

        nmbr_eps.append     (int(contentDict["parameters"]["nmbr_eps"]))
        nmbr_runs.append    (int(contentDict["parameters"]["nmbr_runs"]))
        measure_cost.append (int(contentDict["parameters"]["m_cost"]))


for i in range(nmbr_files):
    avg_cum_reward.append       (np.cumsum(avg_reward[i]))


all_data = [avg_reward, avg_steps, avg_measures,avg_cum_reward]

# Basic Plot Loop:

eps_to_plot = min(np.max(nmbr_eps), max_eps)
for i in range (nmbr_plots):

    plt.title("{} in {}".format(Plot_names_title[i], env_name))
    plt.ylabel(Plot_Y_Label[i])
    plt.xlabel("Episode")

    x = np.arange(eps_to_plot)
    for j in range(nmbr_files):
        plt.plot(x,all_data[i][j][:eps_to_plot])

    plt.legend(Files_legend)
    plt.savefig("Plot_{}_{}".format(Env_fileName, Plot_names_file[i]))
    plt.clf()


# Save relevant vars in json file:

file_text = """
Data collected from running in {} , using the following algorithms:
""".format(env_name)

for i in range(nmbr_files):
    this_rew, this_steps, this_measures= np.average(avg_reward[i]), np.average(avg_steps[i]),np.average(avg_measures[i])
    this_costed = this_rew + this_measures*measure_cost[i]

    max = np.size(avg_reward[i])
    min = m.ceil(max*0.9)
    last_rew, last_steps, last_measures = np.average(avg_reward[i][min:max]), np.average(avg_steps[i][min:max]),np.average(avg_measures[i][min:max])
    last_costed = last_rew + this_measures*measure_cost[i]


    file_text +="""
{}:
nmbr_eps                    = {}
nmbr_runs                   = {}
measure_cost                = {}

avererage reward            = {}
average nmbr steps          = {}
average nmbr measurements   = {}
average non-costed reward       = {}

In last 1/10th of episodes:
avererage reward            = {}
average nmbr steps          = {}
average nmbr measurements   = {}
average non-costed reward       = {}

""".format(
        alg_names[i],
        nmbr_eps[i], nmbr_runs[i], measure_cost[i],
        this_rew, this_steps, this_measures, this_costed,
        last_rew, last_steps, last_measures, last_costed
    )

data_file_name = "Data_{}_{}".format(Env_fileName, timestamp)
with open(data_file_name, 'w') as f:
    f.write(file_text)