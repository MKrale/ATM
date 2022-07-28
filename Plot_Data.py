"""
File for reading & plotting from given data files.

(Only works if nmbr Steps is equal in all files!)
"""

# To be filled in!

Data_path       = "Data/"
Files_to_read   = ["AMRL_BigLakeDet_Run1.txt", "AMRLV2_BigLakeDet_Run1.txt", "AMRLV3_BigLakeDet_Run1.txt" ]
Files_legend    = ["AMRL","AMRL v2","AMRL v3" ]
nmbr_files      = len(Files_to_read)

env_name        = "Big Deterministic Lake-Environment"

#########################################################
#               Where the magic happens!
#########################################################

# Imports
import json
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy.signal import savgol_filter
w1, w2 = 250,2
timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")

# Plot Naming

Plot_names_title    = ["Average reward", "Average Steps", "Average Measurements", "Average Cumulative Reward"]
Plot_Y_Label        = ["Reward", "Steps", "Measurements", "Reward (Cumulative)"]
Plot_names_file     = ["avg_reward", "avg_steps", "avg_ms", "avg_cumrew"]

nmbr_plots          = len(Plot_names_title)

# Data to obtain:

nmbr_steps      = []
measure_cost    = []

avg_reward          = []
avg_cum_reward      = []
avg_steps           = []
avg_measures        = []
avg_reward_noCost   = []

# Read data:

for file_name in Files_to_read:
    with open(Data_path + file_name) as file:
        contentDict = json.load(file)
        
        avg_reward.append       (savgol_filter( np.average (contentDict["reward_per_eps"]      , axis=0), w1, w2 ))

        avg_steps.append        (savgol_filter(np.average (contentDict["steps_per_eps"]       , axis=0), w1, w2 ) )
        avg_measures.append     (savgol_filter(np.average (contentDict["measurements_per_eps"], axis=0), w1, w2 ) )

for i in range(nmbr_files):
    avg_cum_reward.append       (np.cumsum(avg_reward[i]))

nmbr_eps = avg_reward[0].size

all_data = [avg_reward, avg_steps, avg_measures,avg_cum_reward]

# Basic Plot Loop:

for i in range (nmbr_plots):

    plt.title("{} per episode in {}".format(Plot_names_title[i], env_name))
    plt.ylabel(Plot_Y_Label[i])
    plt.xlabel("Episode")

    x = np.arange(nmbr_eps)
    for j in range(nmbr_files):
        plt.plot(x,all_data[i][j])

    plt.legend(Files_legend)
    plt.savefig("Plot_{}_{}".format(Plot_names_file[i], timestamp))
    plt.clf()
