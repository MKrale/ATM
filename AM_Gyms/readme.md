# AM_Gyms (BAM-QMDP Repository)

This folder contains the following RL-environments used in the paper:

- uMV.py and uMV2.py  : the custom toy environments Lucky-Unlucky and A-B;
- SnakeMaze.py        : the SnakeMaze environment;
- DroneInCorridor.py  : the custom drone environmnet;

Some precomputed version of these environments are stored in the 'Learned_Models' folder.
Furthermore, this folder contains the following additional files:

- AM_Env_wrapper .py       : a wrapper class to add measuring functionality to openAI gyms;
- ModelLearnerV2.py     : a class to learn the dynamics of an environment, i.e. transition function, rewards, done-states & Q-values (as well as ModelLearner.py, a previous version);
- AM_Tables.py          : a class to represent and import/export model;
- ModelLearner_Robust   : a class to compute RMDP dynamics;
- Learned Models folder : contains pre-computed (robust) models.
- generic_gym.py        : a class to create openAI environment from P and R tables.

Lastly, it contains the following environments used only for testing:

- MachineMaintenace.py: an OpenAi implementation of a maintenace problem proposed by Delage and Mannor (2010);
- Frozen_lake.py      : a copy of the standard openAI Gym environment with the same name;
- Frozen_lake_v2.py   : a 'less random' version of OpenAI's frozen lake environment;
- Loss_Env.py         : an environment from Krale et al (2023) designed to test Measurement Regret;
- NChainEnv.py        : an implementation of the (discontinued) OpenAI chain environment;
- Blackjack.py        : a version of OpenAI's Blackjack environment, which returns non-factorised states;
- Sepsis folder       : an RL-environment introduced by Nam et al (2021) for Active Measure reinforced learning.
- Maze folder         : an openAI Maze environment.

