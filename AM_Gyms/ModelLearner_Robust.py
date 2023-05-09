import numpy as np
import math as m
#from AM_Gyms.AM_Tables import RAM_Environment_Explicit

def deep_copy(dict, S ,A):
    copy = {}
    for s in range(S):
        copy[s] = {}
        for a in range(A):
            copy[s][a] = dict[s][a]
    return copy
        

class ModelLearner_Robust():
    """Class to find the worst-case transition function and Q-values for a uMDP."""
    
    def __init__(self, model, df = 0.90, optimistic = False):
        
        # Unpacking variables from environment:
        self.model        = model # NOTE: must be of class RAM_Environment_Explicit!
        self.StateSize, self.ActionSize, self.cost, self.s_init = self.model.get_vars()
        self.doneState  = self.StateSize -1
        
        self.Pavg, self.R, self.Qavg = self.model.get_avg_tables()
        self.Pmin, self.Pmax, _R = self.model.get_uncertain_tables()
        
        self.Qavg_max = np.max(self.Qavg, axis=1)
        self.Qr, self.Qr_max = np.copy(self.Qavg), np.copy(self.Qavg_max)
        self.Pr = deep_copy(self.Pavg, self.StateSize, self.ActionSize)
                
        # Other variables:
        self.df         = df
        self.epsilon    = 0.25
        self.optimistic = optimistic
    
    def update_Qavg(self, s, a):
        """Updates Q-table according to (known) model dynamics (currently unused)"""
        self.Qavg[s,a] = self.df * "TBW"
        self.Qavg_max[s] = np.max(self.Qavg[s])
    
    def update_Qr(self, s, a):
        """Updates ICVaR according to (known) model dynamics"""
        
        # 1) Read relevant vars into lists 
        # NOTE: there must be a more efficienty way of doing this...
        pmin, pmax, pguess, qr = [], [], [], []
        states = []
        for state in self.Pavg[s][a].keys():
            states.append(state)
            pmin.append(self.Pmin[s][a][state])
            pmax.append(self.Pmax[s][a][state])
            pguess.append(self.Pr[s][a][state])
            qr.append(self.Qr_max[state])
        
        
        # 2) Get ICVaR values according to custom procedure
        pr = ModelLearner_Robust.custom_delta_minimize(pmin, pmax, pguess, qr, optimistic = self.optimistic)
        
        # 3) Update deltaP's and ICVaR
        thisQ = 0
        for (i,snext) in enumerate(states):
            self.Pr[s][a][snext] = pr[i]
            thisQ += self.df * pr[i] * qr[i]
            if snext in self.R[s][a]:
                thisQ += pr[i] * self.R[s][a][snext]
        self.Qr[s][a] = thisQ
        self.Qr_max[s] = np.max(self.Qr[s])
    
    @staticmethod
    def sort_arrays_to_indexes(arrays, indexes):
        for (i,array) in enumerate(arrays):
            arrays[i] = [x for _, x in sorted(zip(indexes, array))]
        return arrays
    
    @staticmethod    
    def custom_delta_minimize(Pmin:np.ndarray, Pmax:np.ndarray, Pguess:np.ndarray, Qr:np.ndarray, optimistic=False):
        """Calculates the worst-case disturbance delta of transition probabilities probs,
        according to next state icvar's and perturbation budget 1/alpha.
        
        The general idea of this method is to repeatedly maximize the probability for
        the worst-case scenerio (for which delta<1/alpha), while simultaiously lowering 
        probabilities for the best-case scenario. By alternating these, we aim to keep the 
        total transition probability equal to 1.
        """
        # if optimistic:
        #     Qr = np.negative(Qr)
        
        # 1) Sort according to Qr:
        sorted_indices = np.argsort(Qr)
        # Pmin = [p for i, p in sorted(zip(sorted_indices, Pmin))]
        
        Pmin, Pmax, Pguess, Qr = ModelLearner_Robust.sort_arrays_to_indexes( [Pmin, Pmax, Pguess, Qr], sorted_indices )
        
        # 2) Repeatedly higher/lower probability of lowest/highest icvar elements
        sum_delta_p = np.sum(Pguess)
        changable_probs = list(range(len(Pguess)))   # list of probabilities we have not yet changed
        while changable_probs:
            # Higher probability of worst outcomes
            if sum_delta_p <= 1:
                this_i = changable_probs[0]
                sum_delta_p +=  Pmax[this_i] - Pguess[this_i]
                Pguess[this_i] = Pmax[this_i]
                lowest_i_highered = this_i
                changable_probs.pop(0)
            # lower probability of best outcomes
            elif sum_delta_p > 1:
                this_i = changable_probs[-1]
                sum_delta_p -= Pguess[this_i] - Pmin[this_i]
                Pguess[this_i] = Pmin[this_i]
                highest_i_lowered = this_i
                changable_probs.pop(-1)
                
        # 3) Fix probability problems by editing last 'bad' change:
        if m.isclose(sum_delta_p, 1, rel_tol=1e-5):
            pass
        # If our total probability is too low, we must compensate by upping the last probability we lowered.
        elif sum_delta_p < 1:
            gap = np.max([1-sum_delta_p, 0]) 
            Pguess[highest_i_lowered] += gap
        # Similarly, if our total probability is too high, we compensate by lowering the last probability upped.
        elif sum_delta_p > 1:
            gap = sum_delta_p - 1
            Pguess[lowest_i_highered] = np.max([Pguess[lowest_i_highered] - gap, 0])

        # 4) Restore original order
        original_indices = np.argsort(sorted_indices)
        return ModelLearner_Robust.sort_arrays_to_indexes([Pguess], original_indices)[0]
        
        
    def pick_action(self,s):
        """Pick higheste icvar-action epsilon-greedily"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.ActionSize)
        return np.argmax(self.Qr[s])
    
    def run(self, updates = 1_000, logging = True):
        """Calculates model dynamics using eps_modelearning episodes, then ICVaR using
        updates updates per state."""

        if logging:
            print("Learning robust model started:")
        
        for i in range(updates):
            S = np.arange(self.StateSize-1)
            np.random.shuffle(S)
            for s in S:
                a = self.pick_action(s)
                self.update_Qr(s, a)
                
            if (i%(np.min([updates/10, 1000])) == 0 and logging):
                print("Episode {} completed!".format(i+1))

        if logging:
            print("Learning completed after {} updates per state!\n\n".format(updates))
    def get_model(self):
        """Return (Pr, Qr)"""
        return (self.Pr, self.Qr)

    
    
    # def calculate_model_dicts(self):
        
    #     self.P_dict, self.DeltaP_dict = {}, {}
    #     for s in range(self.StateSize):
    #         self.P_dict[s] = {}; self.DeltaP_dict[s] = {}
    #         for a in range(self.ActionSize):
    #             self.P_dict[s][a] = {}; self.DeltaP_dict[s][a] = {}
    #             for (snext,p) in enumerate(self.P[s,a]):
    #                 if p != 0:
    #                     self.P_dict[s][a][snext] = p
    #                     self.DeltaP_dict[s][a][snext] = self.DeltaP[s,a,snext]
                        

# Code for testing:

# from AM_Gyms.frozen_lake import FrozenLakeEnv

# Semi-slippery, larger   
# Env = AM_ENV( FrozenLakeEnv_v2(map_name = "8x8", is_slippery=True), StateSize=64, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# semi-slippery, small
# Env = AM_ENV( FrozenLakeEnv_v2(map_name = "4x4", is_slippery=True), StateSize=16, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# slippery, small
# Env = AM_ENV( FrozenLakeEnv(map_name = "4x4", is_slippery=True), StateSize=16, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# icvar = ICVaR(Env, alpha = 0.3)

# icvar.run(logging=True)
# print(icvar.ICVaR)