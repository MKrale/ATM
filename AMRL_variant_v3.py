'''
    AMRL-V3: 

Based on https://arxiv.org/abs/2005.12697, this file contains code for an Agent to determine both a Transition Table and 
(optimal) policy for a Active-Learning environment (meaning a partially observable env where observations have costs).

A full description can be found in the report corresponding to this code. Briefly, the following has been added/altered between v2 & v3:
    * A Global Q-update has been implemented (TBW...)

(I should probably add a brief description here, anyway...)

'''

from functools import total_ordering
import numpy as np
import math as m
from AM_Env_wrapper import AM_ENV


class AMRL_v3:

    #######################################################
    ###     INITIALISATION AND DEFINING VARIABLES:      ###
    #######################################################

    def __init__(self, env:AM_ENV, eta = 0.00, nmbr_particles = 100, update_globally = True):
        # Environment arguments:
        self.env = env
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()

        # Algo-specific arguments:
        self.eta, self.nmbr_particles = eta, nmbr_particles
        self.NmbrOptimiticTries = 10  #10 # smaller for smaller environments
        self.selfLoopPenalty = 0.95
        self.lossBoost = 1
        self.stopPenalty = 0.0 # smaller (0.01) for smaller/nondet environments
        self.updateAccuracy = 0.0001 # = 1/100 %
        self.max_estimated_loss = self.MeasureCost
        self.optimisticPenalty = 1
        self.update_globally = update_globally

        self.lr = 1 # Learning rate. 
        #Currently unused: instead, Q is re-calculate each pass using Trans-table and current Q-values
        self.df = 0.95 # Discount Factor


        self.init_run_variables()

    def init_run_variables(self):
        # Arrays keeping track of model:
        self.QTable             = np.ones ( (self.StateSize, self.ActionSize), dtype=np.longfloat ) * self.optimisticPenalty                         # as used by algos: includes initial optimitic bias
        self.QTableUnbiased     = np.zeros( (self.StateSize, self.ActionSize), dtype=np.longfloat )                          # only includes measured Q's
        self.QTriesTable        = np.zeros( (self.StateSize, self.ActionSize) )
        self.QTableRewards      = np.zeros( (self.StateSize, self.ActionSize) )
        self.Q_max              = np.zeros( (self.StateSize), dtype=np.longfloat)
        self.TransTable         = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )   # Prediction of transition probs: includes initial optimitic bias
        self.TransTableUnbiased = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )
        self.TTriesTable        = np.zeros( (self.StateSize, self.ActionSize) )
        self.TransTerminals     = np.zeros( (self.StateSize, self.StateSize) )
        self.ChangedStates      = {}
        # Other vars:
        self.totalReward        = 0
        self.episodeReward      = 0
        self.init_episode_variables()

    def init_episode_variables(self):
        # Logging variables
        self.episodeReward          = 0
        self.steps_taken            = 0
        self.measurements_taken     = 0

        # State variables        
        self.is_done                = False

        self.env.reset()
    
    #######################################################
    ###                 RUN FUNCTIONS:                  ###
    #######################################################    

    def run_episode(self):
        # Initialisation (all the s's...):
        self.init_episode_variables()

        s = {}
        if self.s_init == -1: # Random start
            for i in range(self.StateSize):
                s[i] = 1.0/self.StateSize
        else:
            s[self.s_init] = 1
        s_previous = s
        s_last_measurement = self.s_init
        reward, cost, action = 0, 0, 0
        H = [0]
 
        ### MAIN LOOP ###
        while not self.is_done:

            # Find next optimal action & Loss
            cost=0
            (action, estimated_loss) = self.obtain_optimal_action(s)

            # If Loss is "too big" or we do not have enough info about the current state, we do the following:
            if (estimated_loss * self.lossBoost > self.max_estimated_loss) or (not (self.has_transition_support(s_previous, H) )):
                # measure and update model:
                self.measurements_taken += 1
                (s_observed, cost) = self.env.measure()
                # update all vars  & update model
                s = {}
                s[s_observed] = 1
                self.update_model( s,s_last_measurement, s_previous, H, reward, type="QT")
                s_last_measurement = s_observed
                H = []

                #re-compute optimal action
                action = self.obtain_optimal_action(s, returnLoss = False)

            else:
                self.update_model(s,s_last_measurement, s_previous, H, reward, type="Q")

            # Take optimal action or random action (according to eta)

            if np.random.rand() < self.eta:
                action = m.floor(np.random.rand()*self.ActionSize)
            (reward, self.is_done) = self.env.step(action, s)
                
            # Update estimate s & logging variables
            s_previous = s
            s = self.guess_next_state(s, action)
            H.append(action)           

            self.episodeReward  += reward - cost
            self.steps_taken    += 1
            #print(s, s_previous)

            if self.steps_taken % 100 == 0:
                #print("{} steps taken in one episode: performing global update Q".format(self.steps_taken))
                #print(s,self.QTable)
                if self.update_globally:
                    self.update_Q_globally()
                #print(self.QTable)
                
        #print(reward,self.QTable,self.TransTable)
            
        # Update model after done

        if not (self.has_transition_support(s_previous, H) ):
            (s_observed, cost) = self.env.measure()
            #print(s,s_observed, s_previous,H)
            s={}
            s[s_observed]=1
            self.episodeReward -= cost
            self.update_model(s,s_last_measurement,s_previous, H, reward, type="T", isDone=True)
        
        if len(s_previous) <= 1:
            self.update_model(s,s_last_measurement,s_previous, H, reward, type="Q", isDone=True)
        if self.update_globally:
            self.update_Q_globally()
        #print(self.QTable)

        # update logging variables and return
        self.totalReward += self.episodeReward
        returnVars = (self.episodeReward, self.steps_taken, self.measurements_taken)
        return returnVars

    def run(self, nmbr_episodes, get_full_results=False, print_info = True, logmessages = True):
        self.init_run_variables()
        epreward,epsteps,epms = np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes))
        for i in range(nmbr_episodes):
            if (i > 0 and i%100 == 0 and logmessages):
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {})".format( i, nmbr_episodes, np.average(epreward[(i-100):i]), np.average(epsteps[(i-100):i]) ) )
            epreward[i], epsteps[i], epms[i]  = self.run_episode()
        if print_info:
            print("""
Run complete: 
Transition table: {}
Tries Table: {}
QTable: {}
Rewards Table: {}
Unbiased QTable: {}            
            """.format(self.TransTable, self.TTriesTable, self.QTable, self.QTableRewards, self.QTableUnbiased))
        print(self.QTable[0,0])
        if get_full_results:
            return(self.totalReward, epreward,epsteps,epms)
        return self.totalReward

    #######################################################
    ###              HELPER FUNCTIONS:                  ###
    #######################################################

    def obtain_optimal_action(self,S, returnLoss = True):

        #Compute optimal action
        thisQ = np.zeros(self.ActionSize)
        for s in S:
            p = S[s]
            thisQ += self.QTable[s]*p
        #print(thisQ)
        a_opt = np.random.choice(np.where(thisQ==thisQ.max())[0]) #randomize tiebreaks
        

        #Compute Loss
        if returnLoss:     
            Loss = 0
            for s in S:
                p = S[s]
                QTable_max = max (np.max(self.QTable[s]), np.max(self.QTableUnbiased[s]))
                Loss +=  p * max( 0.0, QTable_max - self.QTableUnbiased[s,a_opt] )

                #print("Testing loss of state {0} = {4}: bo-value = {1} (action {5}), actual optimal value = {2}, from Q-table {3}".format(s, self.QTable[s,a_opt], np.max(self.QTable[s]), self.QTable[s], np.max( [0.0, self.QTable[s,a_opt] - np.max(self.QTable[s])] ), a_opt))
            #print((S,Loss))
            #print("TRY: for S={0}, thisQ={1}, giving a_opt={2} with loss {3}".format(S,thisQ,a_opt,Loss))
            return (a_opt, Loss)

        #print("AFTER MEASURING: for S={0}, thisQ={1}, giving a_opt={2}".format(S,thisQ,a_opt))
        return a_opt
            
    def guess_next_state(self, S, action):

        # Create probability distribution of next states
        P = np.zeros(self.StateSize)
        for s in S:
            P += S[s] * self.TransTable[s,action]
        # Filter unlikely next states (for efficiency)

        nonZero = np.nonzero(P)
        P,states = P[nonZero], np.arange(self.StateSize)[nonZero]
        if np.sum(P) == 0:
            P, states = np.ones(self.StateSize)/self.StateSize, np.arange(self.StateSize)
        elif np.sum(P) < 1:
            P = P/np.sum(P)
        
        # Sample next states
        SnextArray = np.random.choice(states, size=self.nmbr_particles, p=P)

        # Combine them into a probability distr.
        S_next = {}
        states, counts = np.unique(SnextArray, return_counts=True)
        for i in range(len(states)):
            s = states[i]
            S_next[s] = counts[i] * 1/self.nmbr_particles

        return S_next

        # Using most likely state:
        S_next = {}
        for s in S:
            S_next[np.argmax(self.TransTable[S[s],action])] = 1
            return S_next
    
    def has_transition_support(self, S, H):
        if len(H) == 0:
            return True
        if len(S) > 0.5*self.nmbr_particles:
            return False
        support = 0
        #print("Support for S={}:".format(S))
        for s in S:
            if  S[s] > (1/self.StateSize):
                support += min( S[s] * 2 * self.NmbrOptimiticTries, S[s]*self.TTriesTable[s,H[-1]] )
                #print("s={}, TTries={}, support={}".format(s,self.TTriesTable[s,H[-1]],support))
        if support < self.NmbrOptimiticTries:
            #print(support)
            0
        return support >= self.NmbrOptimiticTries

    #######################################################
    ###                 MODEL UPDATING:                 ###
    #######################################################

    """ So, this is the part of the algorithm that I'm still experimenting with,
        so for now this section will just contain a couple of different updating 
        methods that might be good. It's messy now, it will be cleaned up later!    
    """

    def update_model(self, s_current, s_last_measurement, s_last, H, reward, type="QT", isDone=False):
        #self.update_T_MonteCarlo(s_current, s_last_measurement, H)
        #self.update_Q_MonteCarlo(s_current, s_last_measurement, H)
        if type=="QT" or type=="T":
            self.update_T_lastStep_only(s_last, s_current, H, isDone)
            #self.update_Q_lastStep_only(s_last, s_current, H, reward, isDone)
        if type=="QT" or type=="Q":
            self.update_Q_lastStep_only(s_last, s_current, H, reward, isDone)


    # Update function for last step only 
    def update_T_lastStep_only(self,S1,S2,H, isDone=False):
        's1 = last state, s2 = current state'
        if len(H)>0:
            action = H[-1]

            for s1 in S1:
                # Update unbiased transition table & tries table
                p1 = S1[s1]

                prevTries = self.TTriesTable[s1,action]
                self.TTriesTable[s1,action] += p1

                thisTriesTimesTrans = np.zeros(self.StateSize)
                if isDone:
                    self.TransTerminals[s1,action] += p1
                else:
                    for s2 in S2:
                        thisTriesTimesTrans[s2] = p1*S2[s2]
                
                prevTrans  = self.TransTableUnbiased[s1,action]
                TriesTimesTrans = prevTries * prevTrans
                    
                self.TransTableUnbiased[s1,action] = (TriesTimesTrans + thisTriesTimesTrans) / ( prevTries + p1)
                #if not m.isclose(1, np.sum(self.TransTableUnbiased[s1,action]) ):
                    #print(s1,action) #,self.TransTableUnbiased[s1,action])
                self.TransTable[s1,action] = self.TransTableUnbiased[s1,action] 
                #if (s1 ==1 & action==1):
                    #print("Testing T-update: S1={}, S2={}, action={}, prevTries={}, unbiased T-table non-zero at {} (with table={})".format(S1,S2,action,prevTries,np.nonzero(self.TransTableUnbiased[s1,action]),self.TransTableUnbiased[s1,action]))

            return
                
                #     # Decide to used biased or unbiased table:
                # if prevTries+p1 > self.NmbrOptimiticTries:
                    
                #     self.TransTable[s1,action] = self.TransTableUnbiased[s1,action]
                # else:
                #     unbiasfactor = ( 1/self.NmbrOptimiticTries * (prevTries+p1))
                #     biasfactor = (1-unbiasfactor) * 1/self.StateSize
                #     self.TransTable[s1,action] =  unbiasfactor * self.TransTableUnbiased[s1,action] + biasfactor
                #     #print("T-table for {0} with action {1} chaged to {2} (unbiased = {3})".format(s1,action,self.TransTable[s1,action],self.TransTableUnbiased[s1,action]))

    def update_Q_lastStep_only(self,S1, S2, H, reward, isDone = False):

        #PROBLEM: self-loops are incentivized: should be fixed somehow

        action = H[-1] 
        if isDone:
            reward -= self.stopPenalty

        for s1 in S1:
            p1 = S1[s1]
            thisQ = 0
            previousQ, previousTries = self.QTableUnbiased[s1,action], self.QTriesTable[s1,action]
            thisTries = previousTries + p1
            for s2 in S2:
                p2 = S2[s2]
                pt = p1*p2 #chance of this transition having occured
                if not isDone:
                    if s1 != s2:
                        thisQ += pt*np.max(self.QTableUnbiased[s2])
                    elif s1 == s2:
                        thisQ += pt*self.selfLoopPenalty*np.max(self.QTableUnbiased[s2])

            totQ = (previousQ*previousTries + (p1 * (self.df*thisQ + reward)) ) / (previousTries+p1) # Dynamic learning rate
            # totQ = (1 - self.lr) * previousQ + self.lr * p1 * (reward + self.df * thisQ) # Static learning rate (unused)

            #print(s1,action,previousQ,totQ, previousQ-totQ)
            #print("Q-update: s1={}, s2={}, totQ ={}, current Q={}".format(s1,s2,totQ,self.QTableUnbiased[s1,action]))
            self.QTriesTable[s1,action], self.QTableUnbiased[s1,action] = thisTries, totQ

            # Update reward table
            # if reward > 0:
            #     print(reward,S1)
            self.QTableRewards[s1,action] = (self.QTableRewards[s1,action]*previousTries + p1**2 * reward) / (previousTries+p1)

            # Update QTable according to QTable_actual and #visits s1 (optimism)
            if self.QTriesTable[s1,action] > self.NmbrOptimiticTries:
                self.QTable[s1,action] = totQ
            else:
                self.QTable[s1,action] = (thisTries*totQ + self.optimisticPenalty*(self.NmbrOptimiticTries - thisTries)) / self.NmbrOptimiticTries
                #self.QTable[s1,action] = 1

    def update_Q_globally(self):
        # Note: the efficiency of this algo could be hugely improved when using better datatypes (heaps to get max node, etc.)
        # However, to test if this idea even works I'll just write a quick-and-dirty version for now.

        # Initialise (should be done throughout algo!)

        self.Q_max = np.max(self.QTable, axis=1)

        for i in range(self.StateSize):
            self.ChangedStates[i] = self.Q_max[i]

        i = 0
        #print("Q: {}\n Qmax: {} \n ".format(self.QTable, self.Q_max))
        while self.ChangedStates:
            i+=1
            # Get current node
            s_source = max(self.ChangedStates) #key & val?

            # Go through all neighbours s:
            for action in range(self.ActionSize):
                to_update = np.unique( np.nonzero(self.TransTable[:,action,s_source]) ) #correct?
                for s in to_update:
                    #print(s_source,action,s,to_update)
                    # for each neighbour, update Q:
                    T_neighbours = (self.TransTableUnbiased[s, action])
                    T_neighbours[self.TransTable[s,action] < self.updateAccuracy] = 0
                    #print(T_neighbours)
                    Q_neighbours = T_neighbours * self.df * self.Q_max
                    Q_neighbours[s] = Q_neighbours[s]*self.selfLoopPenalty
                    #print(Q_neighbours)
                    #print("/n/n")
                    Q_tot =  np.sum(Q_neighbours) + self.QTableRewards[s, action] # ADD ? SIZE!
                    #print(s,Q_tot)
                    self.QTableUnbiased[s,action] = Q_tot
                    #print(np.sum(Q_neighbours), np.sum(Q_neighbours)/len(np.nonzero(Q_neighbours)[0]))
                    # if s == 0:
                    #     print(Q_neighbours, Q_tot)
                    # Add optimism if applicable:
                    Tries = self.QTriesTable[s,action]
                    if Tries > self.NmbrOptimiticTries:
                        if Q_tot > 0:
                            has_changed = ( Q_tot*(1-self.updateAccuracy) > np.max(self.QTable[s]))
                        else:
                            has_changed = ( Q_tot*(1+self.updateAccuracy) > np.max(self.QTable[s]))
                        self.QTable[s,action] = Q_tot
                    else:
                        #has_changed = (self.QTable[s,action]*self.NmbrOptimiticTries - (self.NmbrOptimiticTries -Tries)) / Tries < Q_tot
                        has_changed = False
                        #has_changed = self.QTableUnbiased[s,action] < Q_tot + self.updateAccuracy
                        self.QTable[s,action] = (Tries*Q_tot + self.optimisticPenalty * (self.NmbrOptimiticTries - Tries)) / self.NmbrOptimiticTries
                    
                    # Add s to ChangedStates if new Q is maximal Q of state:
                    if has_changed:
                        self.ChangedStates[s] = Q_tot
                        self.Q_max[s] = Q_tot
            del self.ChangedStates[s_source]
            if i > 3*self.StateSize and i > 120:
                print("Warning: on step {} of Global Q-table update (more than 3x the statesize)!",i)
                #print(self.QTable)
                        


                    
                    

