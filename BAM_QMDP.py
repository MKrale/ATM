'''
    BAM-QMDP, but bayesian!: 

This file contains an agent for BAM-QMDP, an algorithm to find policies through reinforcement for ACNO-MDP environments.
(In short, these are POMDP-settings where observations are either complete when taking a measurement as some cost, and {} otherwise)
A full and formal description can be found in the accompanying paper. 
'''

from csv import QUOTE_ALL
from functools import total_ordering
import numpy as np
import math as m
from AM_Env_wrapper import AM_ENV


class BAM_QMDP:
    "Implementation of BAM-QMPD agent as a python class."

    #######################################################
    ###     INITIALISATION AND DEFINING VARIABLES:      ###
    #######################################################

    def __init__(self, env:AM_ENV, eta = 0.00, nmbr_particles = 100, update_globally = True):
        # Environment arguments:
        self.env = env
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()

        # Meta-variables:
        self.eta = eta                              # Chance of picking a non-greedy action (should be called epsilon...)
        self.nmbr_particles = nmbr_particles        # Number of particles used to represent the belief state.
        self.NmbrOptimiticTries = 20                # Meta variable determining for how many tries a transition should be biased.
        self.selfLoopPenalty = 0.95                 # Penalty applied to Q-value for self loops (1 means no penalty)
        self.lossBoost = 1                          # Testing variable to boost the effect of Measurement Loss/Regret (1 means no boost)
        self.stopPenalty = 0.0                      # Penalty aplied to Q-values achieved in the last step (0 means no penalty)
        self.updateAccuracy = 0.0001                # Smallest change in Q-values for which global Q-update will re-compute Q-values for neighbours
        self.max_estimated_loss = self.MeasureCost  # Minimum Measurement Regret for which a measurement is taken (currently equal to measurement cost)
        self.optimisticPenalty = 1                  # Maximum return estimate (Rewards in all environments are normalised such that this is always 1)
        self.update_globally = update_globally      # Boolean to distinguish between BAM-QMDP (without global Q-update, False) and BAM-QMDP+ (with global Q-update, True)

        self.initPrior = 10**-6                     # Initial alpha-values, as used in the prior for T
        self.optimism_type = "RMAX+"                # RMAX+ (old), UCB, RMAX
        self.UCB_Cp = 0.3
        self.P_noMeasureRate = 0.0
        self.Q_noMeasureRate = 1
        
        self.dynamicLR = False
        self.lr = 0.1                               # Learning rate, as used in standard Q updates. Currently unused, since we use a dynamic learning rate
        self.df = 0.95                              # Discount Factor, as used in Q updates
        
        self.init_run_variables()

    def init_run_variables(self):
        "Initialises all variables that should be reset each run."
        # Arrays keeping track of model:
        self.QTable             = np.ones ( (self.StateSize, self.ActionSize), dtype=np.longfloat ) * self.optimisticPenalty    # Q-table as used by other functions, includes initial bias
        self.QTableUnbiased     = np.zeros( (self.StateSize, self.ActionSize), dtype=np.longfloat )                             # Q-table based solely on experience (unbiased)
        self.QTriesTable        = np.zeros( (self.StateSize, self.ActionSize) )                                                 # Tallies how often (s,a) has been visited (called N_q in paper)
        self.QTableRewards      = np.zeros( (self.StateSize, self.ActionSize) )                                                 # Record average immidiate reward for (s,a) (called \hat{R} in report)
        self.Q_max              = np.zeros( (self.StateSize), dtype=np.longfloat)                                               # Q-value of optimal action as given by Q (used for readability)
        self.alpha              = np.ones( (self.StateSize, self.ActionSize, self.StateSize) ) * self.initPrior# Alpha-values used for dirichlet-distributions.
        self.wasLast            = np.zeros( (self.StateSize, self.ActionSize) )
        self.ChangedStates      = {}
        self.T                  = np.zeros((self.StateSize, self.ActionSize, self.StateSize), dtype=np.longfloat) # States to be checked in global Q update
        # Other vars:
        self.totalReward        = 0     # reward over all episodes
        self.totalSteps         = 0     # steps over all episodes
        self.init_episode_variables()

    def init_episode_variables(self):
        "Initialises all episode-specific variables"
        # Logging variables
        self.episodeReward          = 0 # episodic reward
        self.steps_taken            = 0 # steps taken this episode
        self.measurements_taken     = 0 # measurements taken this episode

        # State variables        
        self.is_done                = False

        self.env.reset()
    
    #######################################################
    ###                 RUN FUNCTIONS:                  ###
    #######################################################    

    def run_episode(self):
        "Performes one episode of BAM-QMPD algorithm."
        # Initialise all variables:
        self.init_episode_variables()

        # Initialise state, history, and previous state vars
        s = {}
        if self.s_init == -1: # Random start
            for i in range(self.StateSize):
                s[i] = 1.0/self.StateSize
        else:
            s[self.s_init] = 1
        s_previous = s
        s_last_measurement = self.s_init
        reward, cost, action = 0, 0, 0
        H = []
 
        ### MAIN LOOP ###
        while not self.is_done:
            
            # Find next optimal action & Loss
            cost=0
            (action, estimated_loss) = self.obtain_optimal_action(s)
            
            #print(action, self.T, self.alpha)

            # If Loss is "too big" or we do not have enough info about the current state, we do the following:
            if (estimated_loss * self.lossBoost > self.max_estimated_loss) or (not (self.has_transition_support(s_previous, H) )) and len(H)>0:
                
                # measure model:
                self.measurements_taken += 1
                (s_observed, cost) = self.env.measure()

                # update all vars, then update model
                s = {}
                s[s_observed] = 1
                s_last_measurement = s_observed

                #re-compute optimal action
                action = self.obtain_optimal_action(s, returnLoss = False)
            
            if len(H) > 0:
                self.update_model(s,s_last_measurement, s_previous, H, reward, type="QT")

            # Take optimal action or random action (according to eta, which is 0 by default)

            if np.random.rand() < self.eta:
                action = m.floor(np.random.rand()*self.ActionSize)
            (reward, self.is_done) = self.env.step(action, s)
                
            # Update estimate s & logging variables
            s_previous = s
            s = self.guess_next_state(s, action)
            H.append(action)           
            self.episodeReward  += reward - cost
            self.steps_taken    += 1
            self.totalSteps     += 1

            # If BAM-QMDP+, do global updates periodically
            if self.steps_taken % 100 == 0:
                if self.update_globally:
                    self.update_Q_globally()
            
        self.update_model(s,s_last_measurement,s_previous, H, reward, type="QT", isDone=True)
        if self.update_globally:
            self.update_Q_globally()

        # update logging variables and return
        self.totalReward += self.episodeReward
        returnVars = (self.episodeReward, self.steps_taken, self.measurements_taken)
        return returnVars

    def run(self, nmbr_episodes, get_full_results=False, print_info = True, logmessages = True):
        "Performes the specified number of episodes of BAM-QMDP."
        self.init_run_variables()
        epreward,epsteps,epms = np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes)), np.zeros((nmbr_episodes))
        for i in range(nmbr_episodes):
            if (i > 0 and i%100 == 0 and logmessages):
                print(self.QTable)
                print(self.alpha)
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {})".format( i, nmbr_episodes, np.average(epreward[(i-100):i]), np.average(epsteps[(i-100):i]) ) )
            epreward[i], epsteps[i], epms[i]  = self.run_episode()
            
        if print_info:
            print("""
Run complete: 
Alpha table: {}
QTable: {}
Rewards Table: {}
Unbiased QTable: {}            
            """.format(self.alpha,  self.QTable, self.QTableRewards, self.QTableUnbiased))
        if get_full_results:
            return(self.totalReward, epreward,epsteps,epms)
        return self.totalReward

    #######################################################
    ###                 HELPER FUNCTIONS:               ###
    #######################################################

    def obtain_optimal_action(self,S, returnLoss = True):
        "Obtains the most greedy action (and its Measurement Regret) according to current belief and model."

        #Compute optimal action
        thisQ = np.zeros(self.ActionSize) # weighted value of each action, according to current belief
        for s in S:
            p = S[s]
            thisQ += self.QTable[s]*p
        a_opt = np.random.choice(np.where(thisQ==thisQ.max())[0]) #randomize tiebreaks
        
        #Compute Loss
        if returnLoss:     
            Loss = 0
            for s in S:
                p = S[s]
                QTable_max = max (np.max(self.QTable[s]), np.max(self.QTableUnbiased[s])) # expected return if we were in state s
                Loss +=  p * max( 0.0, QTable_max - self.QTableUnbiased[s,a_opt] )
            return (a_opt, Loss)
            
        return a_opt
    
    def _dict_to_arrays_(self, S):
        "Returns array of states and probabilities for belief state"
        array = np.array(list(S.items()))
        return array[:,0], array[:,1] #states, then probs
       
    
    def _dict_to_particles_(self, S):
        states, probs = self._dict_to_arrays_(S)
        particles = []
        for (i,s) in enumerate(states):
            for j in range(round(probs[i]*self.nmbr_particles)):
                particles.append(int(s))
        if len(particles) != self.nmbr_particles:
            print(states, probs, particles, len(particles))
            print("Problem in dict_to_particles!")
        return particles
    
    def sample_T(self,s, action, nmbr=1):
        if nmbr != 1:
            print("Average samples not implemented yet!")
        return self.dirichlet_approx(alpha = self.alpha[s,action], nmbr_samples = 1)

    def sample_T_full(self, nmbr_samples = 10 , excludeDones = False): # make nmbr samples global?
        T = np.zeros((self.StateSize, self.ActionSize, self.StateSize))
        DoneFact = 0
        
        
        for s in range(self.StateSize):
            for a in range(self.ActionSize):
                if excludeDones:
                    DoneFact = 0
                    if self.wasLast[s,a] > 0:
                        DoneFact =  self.wasLast[s,a] / (np.sum(self.alpha[s,a]) + self.wasLast[s,a]+ 0.1)
                T[s,a] = np.average(self.dirichlet_approx(self.alpha[s,a], nmbr_samples, DoneFact=DoneFact), axis=0)      
        return T
    
    def dirichlet_approx(self, alpha, nmbr_samples = 1, accuracy = 0.01, DoneFact = 0):
        lenAlpha = len(alpha)
        distr = np.zeros((nmbr_samples, lenAlpha), dtype=np.longfloat)
        
        # Check which values > accuracy
        to_check = np.arange(lenAlpha)[alpha > accuracy]
        if len(to_check) < 1:
            distr = np.ones((nmbr_samples, lenAlpha)) / lenAlpha
        else:
            i=0
            while i < nmbr_samples:
                totalDist = 0
                for j in to_check:
                    distr[i,j] = np.random.standard_gamma(alpha[j])
                    totalDist += distr[i,j]
                if totalDist > 0:
                    distr[i] = distr[i]/totalDist
                    i += 1
        if DoneFact > 0:
            distr = distr *(1-DoneFact)                
        if nmbr_samples == 1:
            return distr[0]
        return distr
                
        
    def guess_next_state(self, S, action):
        "Samples a next belief state after action a, according to current belief and model."

        # Sample probability distribution for next state
        particles = self._dict_to_particles_(S)       
        probDist = np.zeros(self.StateSize)
        for s in particles:
            probDist += self.sample_T(s, action)            
        probDist = probDist / self.nmbr_particles
        
        # Filter all zero-probability states for efficiency
        states = np.arange(self.StateSize)
        filter = np.nonzero(probDist)
        probDist, states = probDist[filter], states[filter]

        # Fix normalisation problems
        if np.sum(probDist) == 0:
            probDist, states = np.ones(self.StateSize)/self.StateSize, np.arange(self.StateSize)
        elif np.sum(probDist) < 1:
            probDist = probDist/np.sum(probDist)
        
        # Sample next states
        SnextArray = np.random.choice(states, size=self.nmbr_particles, p=probDist)

        # Combine states into a probability distr.
        S_next = {}
        states, counts = np.unique(SnextArray, return_counts=True)
        for i in range(len(states)):
            s = states[i]
            S_next[int(s)] = counts[i] * 1/self.nmbr_particles
        return S_next
    
    def has_transition_support(self, S, H):
        "Compute transition support of current belief-action pair"

        # If first action, we do not need to measure
        if len(H) == 0:
            return True
        # If belief state becomes too big, always measure
        if len(S) > 0.5*self.nmbr_particles:
            return False
        
        # Calculate support:
        support = 0
        for s in S:
            if  S[s] > (1/self.StateSize):
                thisSup = S[s]*(np.sum(self.alpha[s,H[-1]] ) - self.StateSize*self.initPrior ) # We take away initial values to get # visits
                support += thisSup #min( S[s] * 2 * self.NmbrOptimiticTries, thisSup )   # Making sure no one state contributes too much support.
        return support >= self.NmbrOptimiticTries

    #######################################################
    ###                 MODEL UPDATING:                 ###
    #######################################################

    def update_model(self, s_current, s_last_measurement, s_last, H, reward, type="QT", isDone=False):
        "Convenience function to simplify model update notation (if type has Q in we do a Q-update, and the same for T)"
        if type=="QT" or type=="T":
            self.update_T_lastStep_only(s_last, s_current, H, isDone)
        if type=="QT" or type=="Q":
            self.update_Q_lastStep_only(s_last, s_current, H, reward, isDone)

    def update_T_lastStep_only(self,S1,S2,H, isDone=False):
        'Updates probability function P according to transition (S1, a, S2)'
        
        
        if len(H)>0:
            action = H[-1]
            
            for s1 in S1:
                # Unpack some variables
                p1 = S1[s1]
                if isDone:
                    self.wasLast[s1,action] += p1
                else:
                    thisAlpha = np.zeros(self.StateSize)

                    for s2 in S2:
                        thisAlpha[s2] += p1*S2[s2]
                    
                    if len(S2) == 1:
                        self.alpha[s1,action] += thisAlpha
                    else:
                        alpha_norm = np.sum(self.alpha[s1,action]-self.initPrior)
                        alpha_unbiased_norm = (self.alpha[s1,action]-self.initPrior) / alpha_norm
                        self.alpha[s1,action] += self.P_noMeasureRate / m.log(self.totalSteps+10) * alpha_unbiased_norm
        return
                
    def update_Q_lastStep_only(self,S1, S2, H, reward, isDone = False):
        'Updates Q-table according to transition (S1, a, S2)'

        action = H[-1] 
        if isDone:
            reward -= self.stopPenalty # penalise stopping with some reward (currently unused)

        for s1 in S1:
            # Unpack some variables
            p1 = S1[s1]
            if p1 < 1:
                p1 = p1 * self.Q_noMeasureRate
            thisQ = 0
            
            previousQ, previousTries = self.QTableUnbiased[s1,action], self.QTriesTable[s1,action]
            thisTries = previousTries + p1


            for s2 in S2:
                #Compute chance of transition:
                p2 = S2[s2]
                pt = p1*p2 

                # Update Q-table according to transition
                if not isDone:
                    if s1 != s2:
                        thisQ += p2*np.max(self.QTable[s2])
                    elif s1 == s2:
                        thisQ += p2*self.selfLoopPenalty*np.max(self.QTable[s2]) # We dis-incentivize self-loops by applying a small penalty to them
                            

                # Update Q-unbiased, N_Q & \hat{R}
                if self.dynamicLR:
                    totQ = (previousQ*previousTries + (p1 * (self.df*thisQ + reward)) ) / (previousTries+p1) # Dynamic learning rate
                else: 
                    thisLR = self.lr * p1
                    totQ = (1-thisLR) * self.QTableUnbiased[s1,action] + thisLR * (reward + self.df*thisQ)
                self.QTriesTable[s1,action], self.QTableUnbiased[s1,action] = thisTries, totQ
                self.QTableRewards[s1,action] = (self.QTableRewards[s1,action]*previousTries + p1 * reward) / (previousTries+p1)
            
            # Implement bias
            if self.QTriesTable[s1,action] > self.NmbrOptimiticTries and self.optimism_type != "UCB":
                        self.QTable[s1,action] = totQ
            else:
                match self.optimism_type:
                    case "RMAX+":
                        self.QTable[s1,action] = (thisTries*totQ + self.optimisticPenalty*(self.NmbrOptimiticTries - thisTries)) / self.NmbrOptimiticTries
                    case "UCB":
                        optTerm = np.sqrt(2 * np.log10(self.steps_taken+2) / (self.QTriesTable+1)) # How do we do this +2 cleanly?
                        self.QTableUnbiased[s1,action] = totQ
                        self.QTable = self.QTableUnbiased + ( self.UCB_Cp * optTerm )
                    case "RMAX":
                        self.QTable[s1,action] = 1
                    
                    
    def update_Q_globally(self):
        'Global Q-update function, which updates all Q-values according to current model. (Used by BAM-QMDP+)'
        # Note: this algorithm is extremely unoptimised: using smarter datatypes and/or keeping track of changed Q-values throughout the rest of the algorithm
        # could make it much quicker. As a proof of concept, though, the function should work fine.

        # Intialise Q_max & ChangedStates
        self.Q_max = np.max(self.QTable, axis=1)

        for i in range(self.StateSize):
            self.ChangedStates[i] = self.Q_max[i] # Array with all states to be checked for updates
        
        # Sample T from distribution:
        self.T = self.sample_T_full(excludeDones=True)
        i = 0
        
        # Reset all Q-values to reward + optimism
        # NOTE: this means we always need to re-compute Q completely, highering average time complexity.
        self.QTable = self.QTableRewards + np.maximum( 0, ( self.NmbrOptimiticTries - self.QTriesTable)/self.NmbrOptimiticTries)
        
        ## MAIN LOOP ##
        while self.ChangedStates:
            i+=1
            # Get current node
            s_source = max(self.ChangedStates)

                # Go through all possible neighbours s:
            for action in range(self.ActionSize):
                to_update = np.unique( np.nonzero(self.T[:,action,s_source]) ) #correct?
                for s in to_update:
                    if np.sum(self.alpha[s,action]) > 1:
                        # for each neighbour, update Q:
                        T_neighbours = np.copy(self.T[s, action]) # Array of probabilities of going to any neighbour (with p > updateAccuracy)
                        T_neighbours[self.T[s,action] < 1/self.StateSize] = 0
                        Q_neighbours = T_neighbours * self.df * self.Q_max # Future estimated return (aka Psi in report)
                        Q_neighbours[s] = Q_neighbours[s]*self.selfLoopPenalty
                        Q_tot =  np.sum(Q_neighbours) + self.QTableRewards[s, action]
                        self.QTableUnbiased[s,action] = Q_tot

                        # Add optimistic bias & compute if Q-values have changed enough to warrent updating it's neigbours:
                        Tries = self.QTriesTable[s,action]
                        if Tries > self.NmbrOptimiticTries:
                            if Q_tot > 0:
                                has_changed = ( Q_tot*(1-self.updateAccuracy) > np.max(self.QTable[s]))
                            else:
                                has_changed = ( Q_tot*(1+self.updateAccuracy) > np.max(self.QTable[s]))
                            self.QTable[s,action] = Q_tot
                        else:
                            has_changed = False # For efficiency reasons, we never add still-biased states back to ChangedStates
                            self.QTable[s,action] = (Tries*Q_tot + self.optimisticPenalty * (self.NmbrOptimiticTries - Tries)) / self.NmbrOptimiticTries
                        
                        # Add s to ChangedStates if new Q is maximal Q of state:
                        if has_changed:
                            self.ChangedStates[s] = Q_tot
                            self.Q_max[s] = Q_tot
            
            # Remove current state from changedStates
            del self.ChangedStates[s_source]

            if i % (5*self.StateSize) == 0 and i > 150:
                print("Warning: on step {} of Global Q-table update (more than 5x the statesize)!".format(i))
                        


                    
                    

