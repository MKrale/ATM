
import numpy as np
from AM_Gyms.AM_Env_wrapper import AM_ENV
from AM_Gyms.AM_Tables import AM_Environment_tables,  RAM_Environment_tables


class Gaussian_AM_ENV(AM_ENV):
    
    def __init__(self, table:AM_Environment_tables, cost, alpha):
        self.table = table
        self.alpha = alpha
        
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = table.get_vars()
        self.P, self.R, self.Q = table.get_tables()
        
        for s in self.StateSize:
            P_s = {}
            for a in self.ActionSize:
                P_sa = {}
                for (s,p) in enumerate(self.P[s,a]):
                    if p != 0:
                        
                this
        self.P_dict = 
        
        self.P_this_run = dict
        self.P_is_chosen= np.zeros((self.StateSize, self.ActionSize), dtype=bool)
        
        self.state = 0
        self.max_steps = 10_000 # Just guessed...
        
    def reset(self):
        self.P_is_chosen = np.zeros((self.StateSize, self.ActionSize), dtype=bool)
        
    def step(self, action):
        
        if self.state not in self.P_this_run:
            self.P_this_run[self.state] = {}
        P_s = self.P_this_run[self.state]
        
        if action not in P_s:
            P_s[action] = self.choose_P(self.state,action)
        
        state_prob_pairs = np.fromiter((P_s[action].items()))
        states, probs = state_prob_pairs[:,0], state_prob_pairs[:,1]
        next_state = np.random.choice(a=states, p=probs)
        
        reward = self.R[self.state, action]
        done = (self.state == self.StateSize-1)
        self.state = next_state
        
        self.steps_taken += 1
        if self.steps_taken > self.max_steps:
            done = True
        
        return (reward, done)
    
    def reset(self):
        self.state = self.s_init
        self.P_this_run = {}
        self.steps_taken = 0
        
    def choose_P(self, state, action):
        for 
        
        