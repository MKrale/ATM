### Variants of AMRL-agent

from math import comb
import numpy as np
import AMRL_Agent as amrl_template

class AMRL_Variant_1(amrl_template.AMRL_Agent):
    '''Variant of AMRL-agent using more complex estimations of '''

    def __init__(self,env,StateSize,MeasureSize,ActionSize, eta=0.05, s_init = 1, m_bias = 0.1, measureCost=0.01):
        #load all environment-specific variables
        self.env, self.StateSize, self.MeasureSize, self.ActionSize,self.eta, self.s_init, self.m_bias, self.measureCost = env, StateSize, MeasureSize, ActionSize,eta, s_init, m_bias, measureCost
        self.nmbrOptimisticTries = 10
        # Create all episode and run-specific variables
        self.reset_Run_Variables()

    def reset_Run_Variables(self):
        # Variables for one run
        self.QTable = np.ones( (self.StateSize,self.ActionSize) )
        self.QTableUnbiased = np.zeros ((self.StateSize, self.ActionSize)) + 0.1
        self.QTriesTable = np.zeros( (self.StateSize, self.ActionSize) )
        self.TransTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) ) + 1/self.StateSize
        self.TriesTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )     # should this not be [stateSize, actionSize] only?
        self.totalReward = 0
        # Variables for one epoch
        self.reset_Epoch_Vars()

    def update_QTable(self, s1, action, measure, s2, reward):
        
        # Update QTable_actual, according to reward etc.
        previousQ, previousTries = self.QTableUnbiased[s1,action], self.QTriesTable[s1,action]
        Q_s2 = np.max(self.QTableUnbiased[s2])
        self.QTriesTable[s1,action] += 1
        self.QTableUnbiased[s1,action] = (previousQ*previousTries + Q_s2 + reward) / (previousTries+1)

        # Update QTable according to QTable_actual and #visits s1 (optimism)
        if self.QTriesTable[s1,action] > self.nmbrOptimisticTries:
            self.QTable[s1,action] = self.QTableUnbiased[s1,action]
        
    def find_optimal_actionPair(self,s):
        
        # Find optimal value according to current Q-table
        a1 = np.argmax(self.QTable[s])

        # Find optimal next action, if not measuring:
        estimated_reward = np.zeros(self.ActionSize)
        for s_next in range(self.StateSize):
                if s_next != s: #dis-incentivize self-loops
                    for a2 in range(self.ActionSize):
                        estimated_reward[a2] += self.TransTable[s,a1,s_next] * np.max(self.QTable[s_next,a2])
        a2_opt = np.argmax(estimated_reward)

        # Compute possible loss
        P = 0
        for s_next in range(self.StateSize):
            P = self.TransTable[s,a1,s_next] * ( self.QTable[s_next, a2_opt] - np.max(self.QTable[s_next]) )
        
        # Decide if measuring
        if P > self.measureCost or np.sum(self.TriesTable[s,a1]) < self.nmbrOptimisticTries:
            return (a1,1)
        return (a1,0)










