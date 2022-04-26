### AMRL-variant using particles to keep track of states, and Loss function to determine whether or not to measure.

import numpy as np
from AM_Env_wrapper import AM_ENV


class AMRL_V2:

    #######################################################
    ###     INITIALISATION AND DEFINING VARIABLES:      ###
    #######################################################

    def __init__(self, env:AM_ENV, eta = 0.1):
        # Environment arguments:
        self.env = env
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()

        # Algo-specific arguments:
        self.eta = eta
        self.NmbrOptimiticTries = 5

        self.init_run_variables()

    def init_run_variables(self):
        # Arrays keeping track of model:
        self.QTable             = np.ones ( (self.StateSize, self.ActionSize) )                 # as used by algos: includes initial optimitic bias
        self.QTableUnbiased     = np.zeros( (self.StateSize, self.ActionSize) )                 # only includes measured Q's
        self.QTriesTable        = np.zeros( (self.StateSize, self.ActionSize) )
        self.TransTable         = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) ) # Prediction of transition probs: includes initial optimitic bias
        self.TransTableUnbiased = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )
        self.TTriesTable        = np.zeros( (self.StateSize, self.ActionSize) )
        # Other vars:
        self.totalReward        = 0
        self.episodeReward      = 0
        self.init_episode_variables()

    def init_episode_variables(self):
        # reset episode variables
        self.episodeReward          = 0
        self.steps_taken            = 0
        self.measurements_taken     = 0
        self.is_done                = False
        self.max_estimated_loss     = self.MeasureCost
        self.env.reset()
    
    #######################################################
    ###                 RUN FUNCTIONS:                  ###
    #######################################################    

    def run_episode(self):
        # Initialisation:
        self.init_episode_variables()
        s_next = {(self.s_init,1)}
        H = {}

        ### MAIN LOOP ###
        while not self.is_done:
            s = s_next
            cost = 0
            (action, estimated_loss) = self.obtain_optimal_action(s)

            # If Loss is too big, take measurement, update model, and re-compute optimal action
            if estimated_loss > self.max_estimated_loss:
                self.measurements_taken += 1
                (s_observed, cost) = self.env.measure()
                s = {(s_observed,1)}
                self.update_model(s,H)
                H = {}
                action = self.obtain_optimal_action(s, returnLoss = False)

            # Take action and update state & history
            (reward, self.is_done) = self.env.step(action)
            H.add(action)
            s_next = self.guess_next_state(s, action)

            # Update logging variables
            self.episodeReward  += reward - cost
            self.steps_taken    += 1
            
        # Update model 
        self.update_model(s,action, isDone = True)

        # update logging variables and return
        self.totalReward += self.episodeReward
        returnVars = (self.episodeReward, self.steps_taken, self.measurements_taken)
        return returnVars

    def run(self, nmbr_episodes, get_full_results=False):
        self.init_run_variables
        results = np.zeros((nmbr_episodes,3))
        for i in range(nmbr_episodes):
            results[i] = self.run_episode()
        if get_full_results:
            return(self.totalReward, results)
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
        a_opt = np.argmax(thisQ)

        #Compute Loss
        if returnLoss:     
            Loss = 0
            for s in S:
                p = S[s]
                Loss += p * max( 0, self.QTable[s,a_opt] - np.argmax(self.QTable[s]) )
            return (a_opt, Loss)
        
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
        SnextArray = np.random.choice(states, size=self.nmbrParticles, p=P)

        # Combine them into a probability distr.
        S_next = {}
        states, counts = np.unique(SnextArray, return_counts=True)
        for i in range(len(states)):
            s = states[i]
            S_next[s] = counts[i] * 1/self.nmbrParticles

        return S_next

    #######################################################
    ###                 MODEL UPDATING:                 ###
    #######################################################

    def update_model(self, S, H, isDone = False):
        "TBW"
        raise NotImplementedError


