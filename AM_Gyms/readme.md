# AM_Gyms (BAM-QMDP Repository)

This folder contains the following RL-environments:

- Frozen_lake.py      : a copy of the standard openAI Gym environment with the same name.
- Frozen_lake_v2.py   : a 'less random' version of OpenAI's frozen lake environment
- Loss_Env.py         : a custom environment designed to test Measurement Regret
- NChainEnv.py        : an implementation of the (discontinued) OpenAI chain environment
- Blackjack.py        : a version of OpenAI's Blackjack environment, which returns non-factorised states.
- Sepsis folder       : a RL-environment introduced by Nam et al (2021) for Active Measure reinforced learning

Furthermore, it contains a AM_Env which is used to add ACNO-MDP functionality to the openAI framework.

For the first three environments mentioned above, data can be found in the data folder. The others were only used for testing.
