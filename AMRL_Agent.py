### Implementation of AMRL-Algorithm as described in https://arxiv.org/abs/2005.12697

import numpy as np

class AMRL_Agent:
    '''Creates a AMRL-Agent, as described in https://arxiv.org/abs/2005.12697'''

    def __init__(self,env,StateSize,MeasureSize,ActionSize, eta=0.05, s_init = 1, m_bias = 0.1, measureCost=0.01):
        self.env, self.StateSize, self.MeasureSize, self.ActionSize,self.eta, self.s_init, self.m_bias, self.measureCost = env, StateSize, MeasureSize, ActionSize,eta, s_init, m_bias, measureCost

        # Tables for Algorithm
        self.QTable = np.zeros( (StateSize,ActionSize,MeasureSize) )
        self.QTable[:,:,1] = m_bias
        self.QTriesTable = np.zeros( (StateSize, ActionSize, MeasureSize) )
        self.TransTable = np.zeros( (StateSize, ActionSize, StateSize) )
        self.TriesTable = np.zeros( (StateSize, ActionSize, StateSize) )

        # Variables for one epoch:
        self.totalReward = 0
        self.currentReward = 0
        self.steps_taken = 0

    def reset_Variables(self):
        self.QTable = np.zeros( (self.StateSize,self.ActionSize,self.MeasureSize) )
        self.QTable[:,:,1] = self.m_bias
        self.QTriesTable = np.zeros( (self.StateSize, self.ActionSize, self.MeasureSize) )
        self.TransTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )
        self.TriesTable = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )

        # Variables for one epoch:
        self.totalReward = 0
        self.currentReward = 0
        self.steps_taken = 0


    def update_TransTable(self,s1, s2, action):
        '''Updates Transition Table by adding transition from s1 to s2 using action'''
        for i in range(self.StateSize):
            previousT, previousTries = self.TransTable[s1,action,i], self.TriesTable[s1,action,i]
            self.TriesTable[s1,action,i] = previousTries+1
            if i == s2:
                self.TransTable[s1,action,i] = (previousT*previousTries+1) / (previousTries+1)
            else:
                self.TransTable[s1,action,i] = (previousT*previousTries) / (previousTries+1)

    def update_QTable(self, s1, action, measure, s2, reward):
        '''Updates Q Table according to action and reward'''
        previousQ, previousTries = self.QTable[s1,action,measure], self.QTriesTable[s1,action,measure]
        Q_s2 = np.max(self.QTable[s2])
        self.QTriesTable[s1,action,measure] += 1
        self.QTable[s1,action,measure] = (previousQ*previousTries + Q_s2 + reward) / (previousTries+1)
        if measure:
            self.QTable[s1,action,0] = (previousQ*previousTries + Q_s2 + reward + 0.01) / (previousTries+1)
        #print("Current Q_table segment:"+str(self.QTable[s1,action,measure]))

    def guess_current_State(self,s,action):
        return (np.argmax(self.TransTable[s,action]))

    def find_optimal_actionPair(self,s):
        '''Returns optimal actionPair according to Q-table'''
        #print(self.QTable[s])
        return (np.unravel_index(np.argmax(self.QTable[s]), self.QTable[s].shape))
    def find_nonOptimal_actionPair(self,s):
        '''Returns random actionPair'''
        return  ( np.random.randint(0,self.ActionSize), np.random.randint(0,self.MeasureSize) )


    def train(self, nmbr_epochs, printResults = False):
        '''Training algorithm of AMRL as given in paper'''
        s_current = self.s_init
        prevReward = 0

        while self.steps_taken < nmbr_epochs:
            if np.random.random(1) < 1-self.eta:
                (action,measure) = self.find_optimal_actionPair(s_current) #Choose optimal action
                #print("Optimal action:" +str((action, measure)))
            else:
                (action,measure) = self.find_nonOptimal_actionPair(s_current) #choose non-optimal action

            (obs, reward, done, info) = self.env.step(action)
            if measure:
                reward -= self.measureCost
            if measure:
                self.update_TransTable(s_current,obs,action)
                s_next = obs
            else:
                s_next = self.guess_current_State(s_current, action)
            #print("current vars:" +str( (s_current,action,measure, reward)))
            self.update_QTable(s_current,action,measure,s_next, reward)
            s_current = s_next
            self.totalReward += reward
            if done:
                self.env.reset()
                s_current = self.s_init
                if printResults:
                    print("Completed epoch "+str(self.steps_taken)+": reward = "+str(self.totalReward-prevReward))
                prevReward = self.totalReward
                self.steps_taken+=1
        if printResults:
            print ("Training completed: total reward = "+str(self.totalReward))

    def train_N(self, N, nmbr_epochs):
        rewardsArray = []
        for i in range(N):
            self.train(nmbr_epochs)
            print("Session {0}: Reward = {1}".format(i+1,self.totalReward))
            rewardsArray.append(self.totalReward)
            self.reset_Variables()
        print("Total results: avg reward = {0}, with std {1}".format(np.mean(rewardsArray), np.std(rewardsArray)))
