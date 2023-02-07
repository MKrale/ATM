# ATM Repository

Repository containing code for ATM-Q (referred to here as BAM-QMDP), and gathered data, as used in the paper "Act-Then-Measure: Reinforcement Learning for Partially Observable Environments with Active Measuring".

## Contents

This repository contains the following files:

Code:
  - **BAM_QMDP.py**           : The BAM-QMDP (a.k.a. (Dyna-)ATMQ) agent as a python class.
  - **Plot_Data.ipynb**       : Code for plotting data.
  - **Run.py**                : Code for automatically running agents on environments & recording their data.

Folders:

  - **AM_Gyms**             : Contains Gym environments used for testing, and wrapper class to make generic OpenAI envs into ACNO-MDP envs.
  - **Data**                : Contains gahtered date for BNAIC and ICAPS-paper (including analysed data & standard plots).
  - **Final_Plots**         : Contains compiled plots.
  - **Baselines**           : Contains code for all baseline algorithms used in the paper or in the testing phase.
  
## How to run

All algorithms can be run using the Run.py file from command line. Running 'python Run.py -h' gives an overview of the functionaliality.

As an example, starting a run looks something like:

'python .\Run.py -algo BAM_QMDP -env Lake -env_map standard8 -env_var semi-slippery -nmbr_eps 2500 -nmbr_runs 1'

This command runs the Dyna_ATMQ (BAM-QMDP) algorithm on the 8x8 semi-slippery lake environment for 2500 episodes (1 run), then saves the date in ./Data using standard formatting.
