# BAM-QMDP Repository

Repository containing code for BAM-QMDP, and gathered data, as used in the paper "The Value of Measuring in Q-learning for Markov Decision Processes"

## Contents

This repository contains the following files:

Code:
  - AMRL_Agent.py         : The AMRL-Q agent as a python-class(as defined by Bellinger et al, 2020)
  - AM_Env_wrapper.py     : Wrapper to turn generic RL-environments into  AM environments
  - AMRL_variant_v3.py    : The BAM-QMDP agent as a python class
  - Plot_Data.py          : Code for plotting data
  - Run.py                : Code for automatically running agents on environments & recording their data

Folders:

  - AM_Gyms             : Contains Gym environments used for testing
  - Data                : Contains data gathered on BAM-QMDP and AMRL-Q
  - Final_Plots         : Contains plots of data in Data folder
  - Unused_Code         : Contains code used in testing phase of BAM-QMPD, or previous version of code.
  - ACNO_generalised    : Contains code from (https://github.com/nam630/acno_mdp), edited to work for any openAI environment. Used as baseline.
  
## How to run

All algorithms can be run using the Run.py file from command line. Running 'python Run.py -h' gives an overview of the functionality.

As an example, starting a run looks something like

'python .\Run.py -algo BAM_QMDP -env Lake -env_map standard8 -env_var semi-slippery -nmbr_eps 2500 -nmbr_runs 1 -save true -plot true'

This command runs the BAM-QMDP algorithm on the 8x8 semi-slippery lake environment for 2500 episodes (1 run), then saves the date in ./Data using standard formatting, and automatically plots it using the ./Plot_Data.py file.