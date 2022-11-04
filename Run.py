'''
File for running & gathering data on Active-Measuring algorithms.
For a brief description of how to use it, see the Readme-file in this repo.

'''

######################################################
        ###             Imports                 ###
######################################################

# File structure stuff
import sys
sys.path.append("C:/Users/merli/OneDrive/Documents/6_Thesis_and_Internship/BAM-QMDP/Baselines")
sys.path.append("C:/Users/merli/OneDrive/Documents/6_Thesis_and_Internship/BAM-QMDP/Baselines/ACNO_generalised")

# External modules
import numpy as np
import gym
import matplotlib.pyplot as plt
import bottleneck as bn
import time as t
import datetime
import json
import argparse
from scipy.signal import savgol_filter
from typing import List, Optional
import os

# Agents
import Baselines.AMRL_Agent as amrl
from BAM_QMDP import BAM_QMDP
from Baselines.ACNO_generalised.Observe_then_plan_agent import ACNO_Agent_OTP
from Baselines.ACNO_generalised.Observe_while_plan_agent import ACNO_Agent_OWP
from Baselines.DRQN import DRQN_Agent
from Baselines.DynaQ import QBasic, QOptimistic, QDyna

# Environments
from AM_Gyms.NchainEnv import NChainEnv
from AM_Gyms.Loss_Env import Measure_Loss_Env
from AM_Gyms.frozen_lake_v2 import FrozenLakeEnv_v2
from AM_Gyms.Sepsis.SepsisEnv import SepsisEnv
from AM_Gyms.Blackjack import BlackjackEnv
from AM_Gyms.frozen_lake import FrozenLakeEnv, generate_random_map, is_valid

# Environment wrappers
from AM_Gyms.AM_Env_wrapper import AM_ENV as wrapper
from AM_Gyms.AM_Env_wrapper import AM_Visualiser as visualiser
from Baselines.ACNO_generalised.ACNO_ENV import ACNO_ENV

# JSON encoder
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

######################################################
        ###       Parsing Arguments           ###
######################################################

parser = argparse.ArgumentParser(description="Run tests on Active Measuring Algorithms")

parser.add_argument('-algo'             , default = 'AMRL',             help='Algorithm to be tested.')
parser.add_argument('-env'              , default = 'Lake_small_det',   help='Environment on which to perform the testing')
parser.add_argument('-env_var'          , default = 'None',             help='Variant of the environment to use (if applicable)')
parser.add_argument('-env_map'          , default = 'None',             help='Size of the environment to use (if applicable)')
parser.add_argument('-m_cost'           , default = -1.0,               help='Cost of measuring (default: use as specified by environment)')
parser.add_argument('-nmbr_eps'         , default = 500,                help='nmbr of episodes per run')
parser.add_argument('-nmbr_runs'        , default = 1,                  help='nmbr of runs to perform')
parser.add_argument('-f'                , default = None,               help='File name (default: generated automatically)')
parser.add_argument('-rep'              , default = './Data/',          help='Repository to store data (default: ./Data')
parser.add_argument('-plot'             , default = "False",              help='Automatically plot data using Plot_Data.py (default: False)')
parser.add_argument('-plot_rep'         , default = './Final_Plots/',   help='Repository to store plots (if plotting is turend on)')
parser.add_argument('-save'             , default = True,               help='Option to save or not save data.')

args            = parser.parse_args()
algo_name       = args.algo
env_name        = args.env
env_variant     = args.env_var
env_map         = args.env_map
MeasureCost     = float(args.m_cost)
nmbr_eps        = int(args.nmbr_eps)
nmbr_runs       = int(args.nmbr_runs)
plotRepo        = args.plot_rep
file_name       = args.f
rep_name        = args.rep

if args.save == "False" or args.save == "false":
        doSave = False
else:
        doSave = True

if args.plot == "False" or args.plot == "false":
        makePlot = False
else:
        makePlot = True

# Create name for Data file
envFullName = env_name
if env_map != 'None':
        envFullName += "_"+env_map
if env_variant != 'None':
        envFullName += "_"+env_variant

######################################################
        ###     Intitialise Environment        ###
######################################################

# Lake Envs
s_init                          = 0
MeasureCost_Lake_default        = 0.05
MeasureCost_Taxi_default        = 0.01 / 20
MeasureCost_Chain_default       = 0.05
remake_env                      = False

def get_env():
        global MeasureCost
        global remake_env
        match env_name:
                
                case "Lake":
                        ActionSize, s_init = 4,0
                        if MeasureCost == -1:
                                MeasureCost = MeasureCost_Lake_default
                        match env_map:
                                case 'None':
                                        StateSize = 4**2
                                        map_name = "4x4"
                                        desc = None
                                case "standard4":
                                        StateSize = 4**2
                                        map_name = "4x4"
                                        desc = None
                                case "standard8":
                                        StateSize = 8**2
                                        map_name = "8x8"
                                        desc = None
                                case "random4":
                                        StateSize = 4**2
                                        map_name = None
                                        desc = generate_random_map(size=4)
                                case "random8":
                                        StateSize = 8**2
                                        map_name = None
                                        desc = generate_random_map(size=8)
                                case "random12":
                                        StateSize = 12**2
                                        map_name = None
                                        desc = generate_random_map(size=12)
                                case "random16":
                                        StateSize = 16**2
                                        map_name = None
                                        desc = generate_random_map(size=16)
                                case "random20":
                                        StateSize = 20**2
                                        map_name = None
                                        desc = generate_random_map(size=20)
                                case "random 24":
                                        StateSize = 24**2
                                        map_name = None
                                        desc = generate_random_map(size=24)
                                case "random 32":
                                        StateSize = 32**2
                                        map_name = None
                                        desc = generate_random_map(size=32)
                                case other:
                                        print("Environment map not recognized for Lake environments!")
                                        exit()
                        if map_name != None:
                                remake_env = True
                        match env_variant:
                                case "det":
                                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                                case "slippery":
                                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=True)
                                case "semi-slippery":
                                        env = FrozenLakeEnv_v2(desc=desc, map_name=map_name)
                                case None:
                                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                                case other: #default = deterministic
                                        print("Environment var not recognised!")
                                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                        
                                        
                case "Taxi":
                        env = gym.make('Taxi-v3')
                        StateSize, ActionSize, s_init = 500, 6, -1
                        if MeasureCost == -1:
                                MeasureCost = MeasureCost_Taxi_default

                case "Chain":
                        match env_map:
                                case '10':
                                        StateSize = 10
                                case '20':
                                        StateSize = 20
                                case '30':
                                        StateSize = 30
                                case '50':
                                        StateSize = 50
                                case other: # default
                                        print("env_map not recognised!")
                                        StateSize = 20
                                
                        env = NChainEnv(StateSize)
                        ActionSize, s_init = 2, 0
                        if MeasureCost == -1:
                                MeasureCost = MeasureCost_Chain_default

                case "Loss":
                        env = Measure_Loss_Env()
                        StateSize, ActionSize, s_init = 4, 2, 0
                        if MeasureCost == -1:
                                MeasureCost = 0.1
                
                case 'Sepsis':
                        env = SepsisEnv()
                        StateSize, ActionSize, s_init = 720, 8, -1
                        if MeasureCost == -1:
                                MeasureCost = 0.05

                case 'Blackjack':
                        env = BlackjackEnv()
                        StateSize, ActionSize, s_init = 704, 2, -1
                        if MeasureCost ==-1:
                                MeasureCost = 0.05
                
                case other:
                        print("Environment not recognised, please try again!")
                        return
                        
        
        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)
        args.m_cost = MeasureCost
        return ENV
        """" 
        Possible extentions: 
                * Bigger settings
                * Eventually: partial measurement envs
        """

######################################################
        ###     Defining Agents        ###
######################################################

# Both final names and previous/working names are implemented here
def get_agent():
        ENV = get_env()
        match algo_name:
                case "AMRL":
                        agent = amrl.AMRL_Agent(ENV, turn_greedy=False)
                case "AMRL_greedy":
                        agent = amrl.AMRL_Agent(ENV, turn_greedy=True)
                case "BAM_QMDP":
                        agent = BAM_QMDP(ENV, offline_training_steps=0)
                case "BAM_QMDP+":
                        agent = BAM_QMDP(ENV, offline_training_steps=25)
                case "ACNO_OWP":
                        ENV_ACNO = ACNO_ENV(ENV)
                        agent = ACNO_Agent_OWP(ENV_ACNO)
                case "ACNO_OTP":
                        ENV_ACNO = ACNO_ENV(ENV)
                        agent = ACNO_Agent_OTP(ENV_ACNO)
                case "DRQN":
                        agent = DRQN_Agent(ENV)
                case "QBasic":
                        agent = QBasic(ENV)
                case "QOptimistic":
                        agent = QOptimistic(ENV)
                case "QDyna":
                        agent = QDyna(ENV)
                case other:
                        print("Agent not recognised, please try again!")
        return agent

######################################################
        ###     Exporting Results       ###
######################################################

if file_name == None:
        file_name = 'AMData_{}_{}_eps={}_runs={}_t={}.json'.format(algo_name, envFullName, nmbr_eps, nmbr_runs, datetime.datetime.now().strftime("%d%m%Y%H%M%S"))

if args.m_cost == -1:
        args.m_cost == MeasureCost
def PR_to_data(pr_time):
        return (datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=pr_time)).strftime("%d%m%Y%H%M%S")

def export_data(rewards, steps, measures,  t_start):
        with open(rep_name+file_name, 'w') as outfile:
                json.dump({
                        'parameters'            :vars(args),
                        'reward_per_eps'        :rewards,
                        'steps_per_eps'         :steps,
                        'measurements_per_eps'  :measures,
                        'start_time'            :t_start,
                        'current_time'          :t.perf_counter()
                }, outfile, cls=NumpyEncoder)


######################################################
        ###     Running Simulations       ###
######################################################

rewards, steps, measures = np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps)), np.zeros((nmbr_runs, nmbr_eps))
t_start = 0 + t.perf_counter()
print("""
Start running agent with following settings:
Algorithm: {}
Environment: {}
nmbr runs: {}
nmbr episodes per run: {}.
""".format(algo_name, envFullName, nmbr_runs, nmbr_eps))

agent = get_agent()

for i in range(nmbr_runs):
        t_this_start = t.perf_counter()
        (r_avg, rewards[i], steps[i], measures[i]) = agent.run(nmbr_eps, True)
        t_this_end = t.perf_counter()
        if doSave:
                export_data(rewards[:i+1],steps[:i+1],measures[:i+1],t_start)
        print("Run {0} done with average reward {2}! (in {1} s, with {3} steps and {4} measurements avg.)\n".format(i, t_this_end-t_this_start, r_avg, np.average(steps[i]),np.average(measures[i])))
        if remake_env:
                agent = get_agent()
print("Agent Done! ({0} runs, total of {1} s)\n\n".format(nmbr_runs, t.perf_counter()-t_start))

if makePlot:
        if not doSave:
                print("Cannot plot data if not saving!")
        else:
                command = 'python ./Plot_Data.py -folderData {0} -folderPlots {1} -file {2}'.format( rep_name, plotRepo, file_name)
                print (command)
                os.system(command)
