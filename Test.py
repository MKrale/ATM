import sys
sys.path.append("C:/Users/merli/OneDrive/Documents/6_Thesis_and_Internship/BAM-QMDP/ACNO_generalised")


from ACNO_generalised.ModelLearner import ModelLearner
from AM_Env_wrapper import AM_ENV
from AM_Gyms.NchainEnv import NChainEnv
from AM_Gyms.frozen_lake import FrozenLakeEnv


# StateSize, ActionSize,MeasureCost, s_init = 64, 4, 0.1 ,0
# env = FrozenLakeEnv(map_name="8x8", is_slippery=True)

StateSize, ActionSize, MeasureCost, s_init = 10, 2, 0.1, 0
env = NChainEnv(StateSize)

ENV = AM_ENV(env, StateSize, ActionSize, MeasureCost, s_init)
AENV = ModelLearner(ENV)

AENV.sample(1000)
AENV.filter_T()
print(AENV.T)


