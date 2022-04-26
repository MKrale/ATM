### Variants of AMRL-agent

from math import comb
import numpy as np
import AMRL_Agent as amrl_template

####################################################################
#           Variant 1 : Estimating Value of Measuring
####################################################################

class AMRL_Variant_1(amrl_template.AMRL_Agent):
    '''Variant of AMRL-agent that bases decision to measure on estimation of lost reward '''

    def __init__(self,env, eta=0.05, m_bias = 0.1):
        #load all environment-specific variables
        self.env = env
        self.StateSize, self.ActionSize, self.measureCost, self.s_init = env.get_vars()

        #load all algo-specific vars (if provided)
        self.eta, self.m_bias = eta, m_bias
        self.nmbrOptimisticTries = 10
        self.MeasureSize = 2
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
        # for each (s,p) in S:
        for s_next in range(self.StateSize):
                if s_next != s: #dis-incentivize self-loops
                    for a2 in range(self.ActionSize):
                        estimated_reward[a2] += self.TransTable[s,a1,s_next] * np.max(self.QTable[s_next,a2])
        a2_opt = np.argmax(estimated_reward)

        # Compute possible loss
        P = 0
        #for each (s,p) in S:
        for s_next in range(self.StateSize):
            P += self.TransTable[s,a1,s_next] * ( self.QTable[s_next, a2_opt] - np.max(self.QTable[s_next]) )
        
        # Decide if measuring
        if P > self.measureCost or np.sum(self.TriesTable[s,a1]) < self.nmbrOptimisticTries:
            return (a1,1)
        return (a1,0)

####################################################################
#           Variant 2 : Using current State Estimation
####################################################################

class AMRL_Variant_2(AMRL_Variant_1):
    nmbrParticles = 500
    maxDifferentParticles = 10

    def __init__(self,env,eta=0.05,  m_bias = 0.1,):
        #load all environment-specific variables
        self.env = env
        self.StateSize, self.ActionSize, self.measureCost, self.s_init = env.get_vars()

        #load all algo-specific vars (if provided)
        self.eta, self.m_bias = eta, m_bias
        self.nmbrOptimisticTries = 5
        self.MeasureSize = 2

        # Create all episode and run-specific variables
        self.reset_Run_Variables()

    def reset_Run_Variables(self):
        # Variables for one run
        self.QTable = np.ones( (self.StateSize,self.ActionSize) )
        self.QTableUnbiased = np.zeros ((self.StateSize, self.ActionSize)) + 0.1
        self.QTriesTable = np.zeros( (self.StateSize, self.ActionSize) )
        self.TransTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) ) + 1/self.StateSize
        self.TransTableUnbiased = np.zeros( ((self.StateSize, self.ActionSize, self.StateSize)) )
        self.TriesTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )     # should this not be [stateSize, actionSize] only?
        self.totalReward = 0
        # Variables for one epoch
        self.reset_Epoch_Vars()
    
    def update_TransTable(self, S1, s2, action):
        '''Updates Transition Table by adding transition from s1 to s2 using action'''
        for s1 in S1:
            p = S1[s1]
            for i in range(self.StateSize):
                previousT, previousTries = self.TransTableUnbiased[s1,action,i], self.TriesTable[s1,action,i]
                self.TriesTable[s1,action,i] = previousTries+p
                if i == s2:
                    self.TransTableUnbiased[s1,action,i] = (previousT*previousTries+p) / (previousTries+p)
                else:
                    self.TransTableUnbiased[s1,action,i] = (previousT*previousTries) / (previousTries+p)

                # Deal with initialisation bias
                if previousTries > self.nmbrOptimisticTries:
                    self.TransTable[s1, action,i] = self.TransTableUnbiased[s1,action,i]
                else:
                    previousT= self.TransTable[s1,action,i]
                    if i == s2:
                        self.TransTable[s1,action,i] = (previousT*previousTries+p) / (previousTries+p)
                    else:
                        self.TransTable[s1,action,i] = (previousT*previousTries) / (previousTries+p)

    
    def update_QTable(self, S1, action, measure, S2, reward):
        for s1 in S1:
            p1 = S1[s1]
            Qtot = 0
            previousQ, previousTries = self.QTableUnbiased[s1,action], self.QTriesTable[s1,action]
            self.QTriesTable[s1,action] += p1
            for s2 in S2:
                p2 = S2[s2]
                Qtot += p2*np.max(self.QTableUnbiased[s2])
            self.QTableUnbiased[s1,action] = (previousQ*previousTries + p1 * (Qtot + reward) ) / (previousTries+p1)

            # Update QTable according to QTable_actual and #visits s1 (optimism)
            if self.QTriesTable[s1,action] > self.nmbrOptimisticTries:
                self.QTable[s1,action] = self.QTableUnbiased[s1,action]
        

    def guess_current_State(self,S,action):
        Snext = {}
        # Put s, ps in array

        # Calculate probabilities for reaching each state given s, ps
        P = np.zeros(self.StateSize)
        for s in S:
            P += self.TransTable[s,action] * S[s]
        nonZero = np.nonzero(P)
        P,states = P[nonZero], np.arange(self.StateSize)[nonZero]

        # Sample current s according to p, then next s according to current s and TransTable
        SnextArray = np.random.choice(states, size=self.nmbrParticles, p=P)
        states, counts = np.unique(SnextArray, return_counts=True)

        for i in range(len(states)):
            s = states[i]
            Snext[s] = counts[i] * 1/self.nmbrParticles
        
        return Snext

    def find_optimal_actionPair(self,S):
        # Find optimal value according to current Q-table
        thisQ = np.zeros(self.ActionSize)
        for s in S:
            p = S[s]
            thisQ += self.QTable[s]*p
        #print(self.QTable[s])
        a1 = np.argmax(thisQ)
        # Randomise a1 if multiple equal occurances
        if thisQ[thisQ==thisQ[a1]].size > 1:
             np.random.choice(np.arange(self.ActionSize)[thisQ==thisQ[a1]])

        # Find optimal next action, if not measuring:
        estimated_reward = np.zeros(self.ActionSize)
        #print(S)
        for s in S:
            p = S[s]
            #print(p)
            for s_next in range(self.StateSize):
                if s_next != s: #dis-incentivize self-loops
                    for a2 in range(self.ActionSize):
                        estimated_reward[a2] += p * self.TransTable[s,a1,s_next] * self.QTable[s_next,a2]
        #print(estimated_reward)
        a2_opt = np.argmax(estimated_reward)

        # Compute possible loss
        P = 0
        for s in S:
            p = S[s]
            for s_next in range(self.StateSize):
                P += max(0, p * self.TransTable[s,a1,s_next] * (  np.max(self.QTable[s_next]) - self.QTable[s_next, a2_opt]) )

        # Decide if measuring
        if P > self.measureCost or np.sum(self.TriesTable[s,a1]) < self.nmbrOptimisticTries:
            print("measuring: P={}".format(P))
            return (a1,1)
        #print("not measuring: P={}".format(P))
        elif len(S) > self.maxDifferentParticles:
            print("Measuring because too many different particles: S={}".format(S))
            return (a1,1)
        return (a1,0)

    def train_epoch(self):
        '''Training algorithm of AMRL as given in paper'''
        S = {}          
        S[self.s_init] = 1
        done = False
        while not done and (self.steps_taken < 10*self.StateSize) :
            # Chose and take step:
            if np.random.random(1) < 1-self.eta:
                (action,measure) = self.find_optimal_actionPair(S) #Choose optimal action
            else:
                (action,measure) = self.find_nonOptimal_actionPair(S) #choose non-optimal action
            #if done and reward == 0:
                #reward -= 0.1
    	    
            
            # Update reward, Q-table and s_next
            s_next = {}
            if measure:
                (obs, reward, done) = self.env.step_and_measure(action)
                self.update_TransTable(S,obs,action)
                self.measurements_taken += 1
                s_next[obs] = 1
                self.update_QTable(S,action,measure,s_next, reward)
            else:
                (reward, done) = self.env.step_no_measure(action)
                s_next = self.guess_current_State(S, action)
            S = s_next
            self.currentReward += reward - self.measureCost*measure #this could be cleaner...
            self.steps_taken += 1
        
        if not done:
            print ("max nmbr of steps exceded!")

        # Reset after epoch, return output
        self.totalReward += self.currentReward
        (rew, steps, ms) = self.currentReward, self.steps_taken, self.measurements_taken
        self.reset_Epoch_Vars()
        return (rew,steps,ms)







