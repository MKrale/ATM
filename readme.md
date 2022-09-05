Repository containing code for BAM-QMDP, and gathered data, as used in the paper "The Value of Measuring in Q-learning for Markov Decision Processes"

It contains the following files:

Code:
  AMRL_Agent.py         : The AMRL-Q agent as a python-class(as defined by Bellinger et al, 2020)
  AM_Env_wrapper.py     : Wrapper to turn generic RL-environments into AM environments
  AMRL_variant_v3.py    : The BAM-QMDP agent as a python class
  Plot_Data.py          : Code for plotting data
  Run.py                : Code for automatically running agents on environments & recording their data

Folders:
    AM_Gyms             : Contains Gym environments used for testing
    Data                : Contains data gathered on BAM-QMDP and AMRL-Q
    Final_Plots         : Contains plots of data in Data folder
    Unused_Code         : Contains code used in testing phase of BAM-QMPD, or previous version of code.
  