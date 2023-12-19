import numpy as np
import math as m

def deep_copy(dict, S ,A):
    copy = {}
    for s in range(S):
        copy[s] = {}
        for a in range(A):
            copy[s][a] = {}
            for (s2, p) in dict[s][a].items():
                copy[s][a][s2] = p
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
        """Updates Qr according to (known) model dynamics"""
        
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
        
        
        # 2) Get Pr values according to custom procedure
        pr = ModelLearner_Robust.custom_delta_minimize(np.array(pmin), np.array(pmax), np.array(pguess), np.array(qr), optimistic = self.optimistic)
        
        # 3) Update deltaP's and Qr's
        thisQ = 0
        for (i,snext) in enumerate(states):
            self.Pr[s][a][snext] = pr[i]
            thisQ += self.df * pr[i] * qr[i]
            if snext in self.R[s][a]:
                thisQ += pr[i] * self.R[s][a][snext]
        self.Qr[s][a] = thisQ
        self.Qr_max[s] = np.max(self.Qr[s])
    
    @staticmethod    
    def custom_delta_minimize(Pmin:np.ndarray, Pmax:np.ndarray, Pguess:np.ndarray, Qr:np.ndarray, optimistic = False):
        """Calculates the worst-case disturbance delta of transition probabilities probs,
        according to next state Qr's and uncertainty set Pmin/Pmax.
        
        The general idea of this method is to repeatedly maximize the probability for
        the worst-case scenerio (for which p < pmax), while simultaiously lowering 
        probabilities for the best-case scenario. By alternating these, we aim to keep the 
        total transition probability equal to 1.
        """
        if optimistic:
            Qr = -Qr
        # 1) Sort according to Qr:
        sorted_indices = np.argsort(Qr)
        Pmin, Pmax, Pguess, Qr = Pmin[sorted_indices], Pmax[sorted_indices], Pguess[sorted_indices], Qr[sorted_indices]
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
        return Pguess[original_indices]
        
        
    def pick_action(self,s):
        """Pick higheste icvar-action epsilon-greedily"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.ActionSize)
        return np.argmax(self.Qr[s])
    
    def run(self, updates = 1_000, logging = True):
        """Calculates model dynamics using eps_modelearning episodes, then Qr using
        'updates' updates per state."""

        if logging:
            print("Learning robust model started:")
        for i in range(updates):
            S = np.argsort(self.Qr_max) # heuristic ordering
            for s in S:
                for a in range(self.ActionSize):
                    self.update_Qr(s, a)
                
            # if ((i+1)%(min([round(updates/10), 1])) == 0 and logging):
            #     print("Episode {} completed!".format(i+1))

        if logging:
            print("Learning completed after {} updates per state!\n\n".format(updates))
    def get_model(self):
        """Return (Pr, Qr)"""
        return (self.Pr, self.Qr)