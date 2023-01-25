import numpy as np
import math as m

from ACNO_Planning import ACNO_Planner
from AM_Gyms.AM_Env_wrapper import AM_ENV
from AM_Gyms.ModelLearner_Robust import ModelLearner_Robust
from AM_Gyms.generic_gym import GenericGym

class ACNO_Planner_Robust(ACNO_Planner):
    
    
    def __init__(self, Env:AM_ENV, alpha:float = 0.3, use_perturbed_MDP = True):
        
        # Regular planning vars
        super().__init__(Env=Env)
        
        # Robust-specific vars
        self.DeltaP     = np.zeros ( (self.StateSize, self.ActionSize, self.StateSize) )
        self.ICVaR      = np.zeros ( (self.StateSize, self.ActionSize) )
        self.ICVaR_max  = np.zeros ( (self.StateSize, self.ActionSize) )
        self.alpha      = alpha
        
        self.use_perturbed_MDP = use_perturbed_MDP
        
    
    def learn_model(self, eps = 10_000, logging = False):
        model = ModelLearner_Robust(self.env, self.alpha, self.df)
        model.run(m.ceil(eps/5), eps, logging=logging)
        (self.P, self.R, self.Q, self.DeltaP, self.ICVaR ) = model.get_model()
        self.Q_max, self.ICVaR_max = np.max(self.Q, axis=1), np.max(self.ICVaR, axis = 1)

        if self.use_perturbed_MDP:
            done_states = np.zeros(self.StateSize)
            done_states[self.doneState] = 1
            robustEnv = GenericGym(self.DeltaP, self.R, 0, terminal_prob = 0.02)
            self.env = AM_ENV(robustEnv, self.StateSize, self.ActionSize, self.cost, self.s_init)

    # Loop is all the same, so we can start with:
    
    #######################################################
    ###             LOOP STEPS FUNCTIONS:               ###
    #######################################################    
    
    def initialise_episode(self):
        
        # Regular planning vars
        super().initialise_episode()
    
        # We use regular b's as the 'robust' b's, since we use those most often.
        # We may also want to track the actual belief states, which we do with these:
        self.b_real_dict, self.b_real_next_dict = {}, {}
        self.b_real_dict[self.s_init], self.b_real_next_dict[self.s_init] = 1,1
        
    def update_step_vars(self):
        super().update_step_vars()
        self.b_real_dict = self.check_validity_belief(self.b_real_next_dict)
        
    def determine_action(self):
        self.a = self.determine_action_general(self.b_dict, self.ICVaR)
        
    def guess_next_state(self):
        _, self.b_real_next_dict        = self.guess_next_state_general(self.b_dict, self.a, self.P)
        self.b_next, self.b_next_dict   = self.guess_next_state_general(self.b_dict, self.a, self.DeltaP)
        
    def take_measurement(self):
        super().take_measurement()
        if self.m:
            self.b_real_next_dict = self.b_next_dict.copy()
    
    def determine_measurement(self):
        self.a_next = self.determine_action_general(self.b_next_dict, self.ICVaR)
        MV = self.get_MV_general(self.b_next_dict, self.a_next, self.ICVaR)
        self.m = MV >= self.cost


class ACNO_Planner_Semi_Robust(ACNO_Planner_Robust):
    """Normal Robust ACNO-Planner, except using the semi-robust Measuring Value"""

    def determine_measurement(self):
        self.a_next = self.determine_action_general(self.b_next_dict, self.ICVaR)
        MV = self.get_MV_general_semirobust(self.b_real_next_dict, self.a_next, self.Q, self.ICVaR)
        self.m = MV >= self.cost
    
    def get_MV_general_semirobust(self,  b_real:dict, a, Q, icvar):
        MV = 0
        for (s,p) in b_real.items():
            a_max_icvar = np.argmax(icvar[s])
            MV += Q[s, a_max_icvar] * p
        for (s,p) in b_real.items():
            MV -= p*Q[s,a]
        return MV
    
        
        