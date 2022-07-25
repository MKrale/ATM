### AMRL-variant using particles to keep track of states, and Loss function to determine whether or not to measure.

import this
import numpy as np
from AM_Env_wrapper import AM_ENV


class AMRL_v2:

    #######################################################
    ###     INITIALISATION AND DEFINING VARIABLES:      ###
    #######################################################

    def __init__(self, env:AM_ENV, eta = 0.1, nmbr_particles = 20):
        # Environment arguments:
        self.env = env
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()

        # Algo-specific arguments:
        self.eta, self.nmbr_particles = eta, nmbr_particles
        self.NmbrOptimiticTries = 10
        self.selfLoopPenalty = 0.8
        self.lossBoost = 1

        self.init_run_variables()

    def init_run_variables(self):
        # Arrays keeping track of model:
        self.QTable             = np.ones ( (self.StateSize, self.ActionSize) )                          # as used by algos: includes initial optimitic bias
        self.QTableUnbiased     = np.zeros( (self.StateSize, self.ActionSize) ) + 0.5                    # only includes measured Q's
        self.QTriesTable        = np.ones ( (self.StateSize, self.ActionSize) )
        self.TransTable         = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) ) + 1/self.StateSize  # Prediction of transition probs: includes initial optimitic bias
        self.TransTableUnbiased = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )
        self.TTriesTable        = np.zeros( (self.StateSize, self.ActionSize) )
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
        self.max_estimated_loss     = self.MeasureCost


        self.env.reset()
    
    #######################################################
    ###                 RUN FUNCTIONS:                  ###
    #######################################################    

    def run_episode(self):
        # Initialisation:
        self.init_episode_variables()
        s_next = {}
        s_next[self.s_init] = 1
        s_previous = s_next
        s_last_measurement = self.s_init
        reward = 0
        H = [0]

        ### MAIN LOOP ###
        while not self.is_done:
            s = s_next
            cost = 0
            (action, estimated_loss) = self.obtain_optimal_action(s)

            # If Loss is "too big" or we do not have enough info about the current state, we do the following:
            if estimated_loss * self.lossBoost > self.max_estimated_loss or (not (self.has_transition_support(s, H) )):
                # measure and update model:
                #print("hello?")
                self.measurements_taken += 1
                (s_observed, cost) = self.env.measure()
                self.update_model( s_observed,s_last_measurement, s_previous, H, reward)

                # update all vars to reflect measurement
                s_last_measurement = s_observed
                s = {}
                s[s_observed] = 1
                H = []

                #re-compute optimal action
                action = self.obtain_optimal_action(s, returnLoss = False)

            # Take action and update state & history
            (reward, self.is_done) = self.env.step(action, s)
            H.append(action)
            s_next = self.guess_next_state(s, action)
            s_previous = s

            # Update logging variables
            self.episodeReward  += reward - cost
            self.steps_taken    += 1
            
        # Update model after done
        #print("done!")
        self.update_model(s,s_last_measurement,s_previous, H, reward, isDone = True)

        # update logging variables and return
        self.totalReward += self.episodeReward
        returnVars = (self.episodeReward, self.steps_taken, self.measurements_taken)
        return returnVars

    def run(self, nmbr_episodes, get_full_results=False):
        self.init_run_variables
        epreward,epsteps,epms = np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes))
        for i in range(nmbr_episodes):
            epreward[i], epsteps[i], epms[i]  = self.run_episode()
        #print(self.TransTable, self.TTriesTable, self.QTable)
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
        a_opt = np.random.choice(np.where(thisQ==thisQ.max())[0]) #randomize tiebreaks
        

        #Compute Loss
        if returnLoss:     
            Loss = 0
            for s in S:
                p = S[s]
                Loss +=  p*max( 0.0, np.max(self.QTable[s]) - self.QTable[s,a_opt] )
                #print("Testing loss of state {0} = {4}: bo-value = {1}, actual optimal value = {2}, from Q-table {3}".format(s, self.QTable[s,a_opt], np.max(self.QTable[s]), self.QTable[s], np.max( [0.0, self.QTable[s,a_opt] - np.max(self.QTable[s])] )))
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
        if len(H) == 0 or len(S) == 1:
            return True
        support = 0
        for s in S:
            support += S[s]*self.TTriesTable[s,H[-1]]
        #print (support)
        return support > self.NmbrOptimiticTries

    #######################################################
    ###                 MODEL UPDATING:                 ###
    #######################################################

    """ So, this is the part of the algorithm that I'm still experimenting with,
        so for now this section will just contain a couple of different updating 
        methods that might be good. It's messy now, it will be cleaned up later!    
    """

    def update_model(self, s_current, s_last_measurement, s_last, H, reward, isDone = False):
        #self.update_T_MonteCarlo(s_current, s_last_measurement, H)
        #self.update_Q_MonteCarlo(s_current, s_last_measurement, H)
        if not isDone:
            self.update_T_lastStep_only(s_last, s_current, H)
            self.update_Q_lastStep_only(s_last, s_current, H, reward)
        if isDone:
            self.update_Q_lastStep_only(s_last, s_current, H, reward, isDone)


    # Update function for last step only 
    def update_T_lastStep_only(self,S1,s2,H):
        's1 = last state, s2 = current state'
        action = H[-1]
        for s1 in S1:
            # Update unbiased transition table & tries table
            p = S1[s1]
            prevTrans, prevTries = self.TransTableUnbiased[s1,action],self.TTriesTable[s1,action]
            TriesTimesTrans = prevTries * prevTrans
            TriesTimesTrans[s2] += p
            self.TransTableUnbiased[s1,action] = TriesTimesTrans / ( prevTries + p)
            self.TTriesTable[s1,action] += p
            

            # Decide to used biased or unbiased table:
            if prevTries+p > self.NmbrOptimiticTries:
                self.TransTable[s1,action] = self.TransTableUnbiased[s1,action]
            else:
                unbiasfactor = ( 1/self.NmbrOptimiticTries * (prevTries+p))
                biasfactor = (1-unbiasfactor) * 1/self.StateSize
                self.TransTable[s1,action] =  unbiasfactor * self.TransTableUnbiased[s1,action] + biasfactor
            #print("T-table for {0} with action {1} chaged to {2} (unbiased = {3})".format(s1,action,self.TransTable[s1,action],self.TransTableUnbiased[s1,action]))

    def update_Q_lastStep_only(self,S1, s2, H, reward, isDone = False):

        #PROBLEM: self-loops are incentivized: should be fixed somehow

        action = H[-1]
        for s1 in S1:
            p1 = S1[s1]
            thisQ = 0
            previousQ, previousTries = self.QTableUnbiased[s1,action], self.QTriesTable[s1,action]
            thisTries = previousTries + p1
            if not isDone:
                if s1 != s2:
                    thisQ += np.max(self.QTableUnbiased[s2])
                elif s1 == s2:
                    thisQ += self.selfLoopPenalty*np.max(self.QTableUnbiased[s2])
            if thisQ > 1:
                print ("thisQ = {}, s = {}".format(thisQ,s1))
            totQ = (previousQ*previousTries + (p1 * (thisQ + reward)) ) / (previousTries+p1)
            if totQ > 1:
                print (totQ)
            self.QTriesTable[s1,action], self.QTableUnbiased[s1,action] = thisTries, totQ

            # Update QTable according to QTable_actual and #visits s1 (optimism)
            if self.QTriesTable[s1,action] > self.NmbrOptimiticTries:
                self.QTable[s1,action] = totQ
            else:
                self.QTable[s1,action] = (thisTries*totQ + (self.NmbrOptimiticTries - thisTries)) / self.NmbrOptimiticTries
                if self.QTable[s1,action] > 1:
                    print ("aha!")
