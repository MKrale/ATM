"""
File containing a number of basic RL-algorithms for MDPs, reworked for the ACNO-MDP setting.
Currently unused and untested, so use at own risk!
"""

from AM_Gyms.ModelLearner import ModelLearner
from AM_Gyms.AM_Env_wrapper import AM_ENV
import numpy as np


class QBasic:
    """Class for standard Q-learning of AM-environments"""
    
    def __init__(self, ENV:AM_ENV):
        self.env = ENV
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = self.env.get_vars()
        
        self.T_counter = build_dictionary(self.StateSize, self.ActionSize)
        self.T_counter_total = build_dictionary(self.StateSize, self.ActionSize, cumulative=True)
        self.T = build_dictionary(self.StateSize, self.ActionSize)
        self.Q = np.zeros((self.StateSize, self.ActionSize)) + 0.8
        
        self.lr = 0.5
        self.df = 0.95
        self.selfLoopPenalty = self.MeasureCost
        self.includeCost = False
        
        
    def update_Q(self,s,action,reward):
        Psi = 0
        for (snext, p) in self.T[s][action].items():
            Psi += p* np.max(self.Q[snext])
        self.Q[s,action] = (1-self.lr) * self.Q[s,action] + self.lr * ( reward + self.df * Psi   )
    
    def update_T(self,s,a,obs):
        if obs in self.T_counter[s][a]:
            self.T_counter[s][a][obs] += 1
        else:
            self.T_counter[s][a][obs] = 1
        self.T_counter_total[s][a] += 1
        for s2 in self.T_counter[s][a].keys():
            self.T[s][a][s2] = self.T_counter[s][a][s2] / self.T_counter_total[s][a]
        
    def pick_action(self,s):
        return np.argmax(self.Q[s])
    
    def run_step(self,s):
        action = self.pick_action(s)
        #print(s,action)

        (reward, done) = self.env.step(action)
        (obs, cost) = self.env.measure()
        
        if self.includeCost:
            reward -= cost
        
        self.update_Q(s, action, reward)
        if not done:
            self.update_T(s, action, obs)
        return obs, reward, done
    
    def run_episode(self): 
        s = self.s_init
        done = False
        totalReward, steps = 0, 0
        self.env.reset()
        
        while not done:
            
            obs, reward, done = self.run_step(s)
            totalReward += reward
            steps += 1
            s = obs
        return totalReward, steps
        
    def run(self, episodes, logging = False):
        rewards, steps = np.zeros(episodes), np.zeros(episodes)
        for i in range(episodes):
            rewards[i], steps[i] = self.run_episode()
            if logging and i%100 == 0:
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {})".format( 
                        i, episodes, np.average(rewards[(i-100):i]), np.average(steps[(i-100):i]) ) )
        return np.sum(rewards), rewards, steps, np.ones(episodes)
    
class QOptimistic(QBasic):
    
    def __init__(self, ENV):
        super().__init__(ENV)
        self.Q_unbiased = np.zeros((self.StateSize, self.ActionSize)) + 0.8
        self.N_since_last_tried = np.ones((self.StateSize, self.ActionSize))
        self.optBias = 10**-600
        
    
    def update_Q(self, s, action, reward, obs):
        Psi = 0
        Psi += np.sum(self.T[s,action] * np.max(self.Q, axis=1))
        self.Q_unbiased[s,action] = (1-self.lr) * self.Q[s,action] + self.lr * ( reward + self.df * Psi )
        
        self.Q[s,action] = self.Q_unbiased[s,action] #+ self.optBias*np.sqrt(self.N_since_last_tried)
        self.N_since_last_tried += 1
        self.N_since_last_tried[s,action] = 1 
        self.Q = self.Q + self.optBias * ( np.sqrt(self.N_since_last_tried-1) + np.sqrt(self.N_since_last_tried)  )
    
class QDyna(QBasic):
    
    def __init__(self, ENV: AM_ENV):
        super().__init__(ENV)
        self.R_counter = build_dictionary(self.StateSize, self.ActionSize, cumulative=True)
        self.trainingSteps = 10
        
    def update_R(self,s,action,reward):
        self.R_counter[s][action] += reward
        
    def update_Q(self,s,action,reward, isReal=True):
        super().update_Q(s,action,reward)
        if isReal:
            self.update_R(s,action,reward)
    
    def run_step(self, s):
        obs,reward, done = super().run_step(s)
        for i in range(self.trainingSteps):
            s = np.random.randint(self.StateSize)
            action = self.pick_action(s)
            if self.T_counter_total[s][action] > 3:
                self.simulate_step(s,action)
        return obs, reward, done
    
    def simulate_step(self,s,action):
        states, probs = [], []
        for (snext,prob) in self.T[s][action].items():
            states.append(snext), probs.append(prob)
        snext = np.random.choice(states, p=probs)
        r = self.R_counter[s][action]/self.T_counter_total[s][action]
        #r -= self.MeasureCost
        self.update_Q(s,action,r, isReal=False) #NO COST!!!


def build_dictionary(statesize, actionsize, copyarray:np.ndarray = None, cumulative = False):
    "As edited from ModelLearner_V2"
    dict = {}
    for s in range(statesize):
        dict[s] = {}
        for a in range(actionsize):
            if cumulative == True:
                dict[s][a] = 0
            else:
                dict[s][a] = {}
                if copyarray is not None:
                    for snext in range(statesize):
                        dict[s][a][snext] = copyarray[s,a,snext]
    return dict