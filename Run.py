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
from typing import List, Optional

import AMRL_Agent as amrl
from AM_Env_wrapper import AM_ENV as wrapper
from AM_Env_wrapper import AM_Visualiser as visualiser
from BAM_QMDP import BAM_QMDP

from AM_Gyms.NchainEnv import NChainEnv
from AM_Gyms.Loss_Env import Measure_Loss_Env
from AM_Gyms.frozen_lake_v2 import FrozenLakeEnv_v2
from AM_Gyms.Sepsis.SepsisEnv import SepsisEnv
from AM_Gyms.Blackjack import BlackjackEnv
from AM_Gyms.frozen_lake import FrozenLakeEnv, generate_random_map, is_valid

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

######################################################
        ###       Parsing Arguments           ###
######################################################

parser = argparse.ArgumentParser("Run tests on Active Measuring Algorithms")

parser.add_argument('-algo'             , default = 'AMRL',             help='Algorithm to be tested.')
parser.add_argument('-env'              , default = 'Lake_small_det',   help='Environment on which to perform the testing')
parser.add_argument('-env_var'          , default = 'None',             help='Variant of the environment to use (if applicable')
parser.add_argument('-env_map'          , default = 'None',             help='Size of the environment to use (if applicable)')
parser.add_argument('-m_cost'           , default = -1.0,               help='Cost of measuring (default: use as specified by environment)')
parser.add_argument('-nmbr_eps'         , default = 500,                help='nmbr of episodes per run')
parser.add_argument('-nmbr_runs'        , default = 1,                  help='nmbr of runs to perform')
parser.add_argument('-plot'             , default = False,              help='Automatically plot data using ... (default: False)')
parser.add_argument('-f'                , default = None,               help='File name (default: generated automatically)')
parser.add_argument('-rep'              , default = './Data/',          help='Repository to store data (default: ./Data')
parser.add_argument('-save'             , default = True,               help='Option to save or not save data.')

args            = parser.parse_args()
algo_name       = args.algo
env_name        = args.env
env_variant     = args.env_var
env_map         = args.env_map
MeasureCost     = float(args.m_cost)
nmbr_eps        = int(args.nmbr_eps)
nmbr_runs       = int(args.nmbr_runs)
plot            = args.plot
file_name       = args.f
rep_name        = args.rep

if args.save == "False" or args.save == "false":
        doSave = False
else:
        doSave = True
        
envFullName = env_name
if env_map != 'None':
        envFullName += env_map
if env_variant != 'None':
        envFullName += env_variant

######################################################
        ###     Intitialise Environment        ###
######################################################

# Lake Envs
s_init                          = 0
MeasureCost_Lake_default        = 0.01
MeasureCost_Taxi_default        = 0.01 / 20
MeasureCost_Chain_default       = 0.05
remake_env                      = False

all_env_names = [
"Lake_small_det", "Lake_small_nondet", "Lake_small_nondet_v2",
"Lake_big_det", "Lake_big_nondet", "Lake_big_nondet_v2",
 "Chain_small", "Chain_big", "Chain_huge",
 "Loss", "Taxi", "Sepsis", "Blackjack"]
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
                                        stateSize = 4**2
                                        map_name = "4x4"
                                        desc = None
                                case "standard4":
                                        stateSize = 4**2
                                        map_name = "4x4"
                                        desc = None
                                case "standard8":
                                        stateSize = 8**2
                                        map_name = "8x8"
                                        desc = None
                                case "random4":
                                        stateSize = 4**2
                                        map_name = None
                                        desc = generate_random_map(size=4)
                                case "random8":
                                        stateSize = 8**2
                                        map_name = None
                                        desc = generate_random_map(size=8)
                                case "random12":
                                        stateSize = 12**2
                                        map_name = None
                                        desc = generate_random_map(size=12)
                                case "random16":
                                        stateSize = 16**2
                                        map_name = None
                                        desc = generate_random_map(size=16)
                                case "random 24":
                                        stateSize = 24**2
                                        map_name = None
                                        desc = generate_random_map(size=24)
                                case "random 32":
                                        stateSize = 32**2
                                        map_name = None
                                        desc = generate_random_map(size=32)
                                case other:
                                        print("Environment map not recognized for Lake environments!")
                                        exit()
                        if map_name != None:
                                remake_env = True
                        match env_variant:
                                case 'None': #default = deterministic
                                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                                case "det":
                                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=False)
                                case "slippery":
                                        env = FrozenLakeEnv(desc=desc, map_name=map_name, is_slippery=True)
                                case "semi-slippery":
                                        env = FrozenLakeEnv_v2(desc=desc, map_name=map_name)
                        print (stateSize, ActionSize, s_init)
                        ENV = wrapper(env, stateSize, ActionSize, MeasureCost, s_init)
                                        
                case "Taxi":
                        env = gym.make('Taxi-v3')
                        StateSize, ActionSize, s_init = 500, 6, -1
                        if MeasureCost == -1:
                                MeasureCost = MeasureCost_Taxi_default
                        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init, max_steps=500, max_reward = 20)

                case "Chain":
                        match env_map:
                                case 'None':
                                        StateSize = 20
                                case '10':
                                        StateSize = 10
                                case '20':
                                        StateSize = 20
                                case '30':
                                        StateSize = 30
                        env = NChainEnv(StateSize)
                        ActionSize, s_init = 2, 0
                        if MeasureCost == -1:
                                MeasureCost = MeasureCost_Chain_default
                        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)

                case "Loss":
                        env = Measure_Loss_Env()
                        StateSize, ActionSize, s_init = 4, 2, 0
                        if MeasureCost == -1:
                                MeasureCost = 0.1
                        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)
                
                case 'Sepsis':
                        env = SepsisEnv()
                        StateSize, ActionSize, s_init = 720, 8, -1
                        if MeasureCost == -1:
                                MeasureCost = 0.05
                        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)

                case 'Blackjack':
                        env = BlackjackEnv()
                        StateSize, ActionSize, s_init = 704, 2, -1
                        if MeasureCost ==-1:
                                MeasureCost = 0.05
                        ENV = wrapper(env, StateSize, ActionSize, MeasureCost, s_init)
                
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
                        agent = BAM_QMDP(ENV, update_globally=False)
                case "BAM_QMDP+":
                        agent = BAM_QMDP(ENV)
        return agent

"""" 
Possible Extentions: 
        * ACNO's
        * BAM_QMDP with different settings
        * AMRL which switches to completely greedy after initialisation
"""

######################################################
        ###     Exporting Results       ###
######################################################

if file_name == None:
        file_name = 'AMData_{}_{}_eps={}_runs={}_t={}'.format(algo_name, env_name, nmbr_eps, nmbr_runs, datetime.datetime.now().strftime("%d%m%Y%H%M%S"))

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
                        'start_time'            :PR_to_data(t_start),
                        'current_time'          :PR_to_data(t.perf_counter())
                }, outfile, cls=NumpyEncoder)


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
