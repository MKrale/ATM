import numpy as np
from collections import deque 


from AM_Gyms.AM_Tables import AM_Environment_tables, RAM_Environment_tables
from AM_Gyms.AM_Env_wrapper import AM_ENV

class ACNO_Planner_Batch():
    
    def __init__(self, Env:AM_ENV, tables = AM_Environment_tables, df=0.95):
        
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = tables.get_vars()
        self.P, self.R, self.Q = tables.get_tables()
        
        self.df         = df
        self.epsilon    = 0
        self.batchSize  = 25
    
    def run(self, eps, logging=False):
        
        if logging:
            print("starting learning of model...")
        rewards, steps, measurements = np.zeros(eps), np.zeros(eps), np.zeros(eps)
        log_nmbr = 100
        
        if logging:
            print("Start planning:")
        
        for i in range(eps):
            reward, step, measurement = self.run_episode()
            rewards[i], steps[i], measurements[i] = reward, step, measurement
            if (i > 0 and i%log_nmbr == 0 and logging):
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {}, nmbr measures = {})".format( 
                        i, eps, np.average(rewards[(i-log_nmbr):i]), np.average(steps[(i-log_nmbr):i]), np.average(measurements[(i-log_nmbr):i]) ) )
            
        return (np.sum(rewards), rewards, steps, measurements)
    
    def run_episode(self):
        
        arraysize = self.batchSize + 2
        BeliefArray = np.empty(arraysize, dtype=dict)
        ActionArray, MeasureArray = np.zeros(arraysize, dtype=int), np.zeros(arraysize,dtype=int)
        BeliefArray[0] = {0:1}
        t = 0
        done = False
        total_reward, total_steps, total_measures = 0, 0, 0
        
        while not done:
            if t == 0:
                ActionArray[t] = self.determine_action(BeliefArray[t])
                
            BeliefArray[t+1] = self.compute_next_belief   (BeliefArray[t], ActionArray[t]) 
            ActionArray[t+1] = self.determine_action      (BeliefArray[t]) 
            MeasureArray[t]  = self.determine_measurement (BeliefArray[t+1], ActionArray[t+1]) 
            
            if MeasureArray[t]:
                reward, done = self.execute_actions_all(ActionArray, BeliefArray)
                BeliefArray[0], cost = self.measure()
                
                total_reward += reward - cost
                total_measures += 1
                total_steps += t
                t = 0
            
            elif t >= self.batchSize-2: # Maybe also if we think chance of being done is sufficiently big?
                reward, done, deltaT = self.execute_action(ActionArray, BeliefArray)
                np.roll(BeliefArray, deltaT), np.roll(ActionArray, deltaT), np.roll(MeasureArray, deltaT)
                    
                total_reward += reward
                total_steps += deltaT
                t -= deltaT
            else:
                t += 1
        
        return total_reward, total_steps, total_measures
    
    
    def determine_action(self, b):
        return optimal_action(b, self.Q, None)
    
    def compute_next_belief(self, b, a):
        return next_belief(b,a,self.P)
    
    def determine_measurement(self, b_next, a_next):
        return measuring_value(b_next, a_next, self.Q) > 0
    
    def execute_actions_all(self, ActionArray, BeliefArray):
        totalReward = 0
        done = False
        for a in ActionArray:
            reward, done = self.env.step(a)
            totalReward += reward
            print(done)
            if done:
                break
        return totalReward, done
    
    def execute_action(self, ActionArray, BeliefArray):
        reward, done = self.env.step(ActionArray[0])
        return reward, done, 1
    
    def measure(self):
        return self.env.measure()
        
        
    


# Generalized functions:

def optimal_action(b:dict, Q1:np.ndarray, Q2:np.ndarray = None):
    """Returns the optimal action for belief state b as given by Q1.
    Ties are broken according to Q2, then randomly."""
    for (i,(state, prob)) in enumerate(b.items()):
        if i == 0:
            actionsize = np.size(Q1[state])
            thisQ = prob * Q1[state]
        else:
            thisQ += prob * Q1[state]
    
    thisQMax = np.max(thisQ)
    filter = np.isclose(thisQ, thisQMax, rtol=0.01, atol= 0.001)
    optimal_actions = np.arange(actionsize)[filter]
    
    if np.size(optimal_actions) > 1 and Q2 != None:
        print("TBW")

    return int(np.random.choice(optimal_actions))

def next_belief(b:dict, a:int, P:np.ndarray, min_probability_considered:float = 0.01):
    """computes next belief state, according to current belief b, action a and transition function P.
    All belief probabilities smaller than min_probability_considered are ignored (for efficiency reasons)."""
    for (i, (state, prob)) in enumerate(b.items()):
        if i == 0:
            StateSize = np.size(P[state,a])
            b_next_array = P[state,a] * prob
        else:
            b_next_array += P[state,a] * prob
    
    filter = b_next_array >= min_probability_considered
    states = np.arange(StateSize)[filter]
    b_next_array = b_next_array[filter]
    b_next_array = b_next_array / np.cumsum(b_next_array)
    
    b_next = dict()
    for (state, prob) in zip(states, b_next_array):
        b_next[state] = prob
    
    return b_next

def measuring_value(b:dict, a_b:int, Q:np.ndarray):
    MV = 0
    for (state, prob) in b.items():
        q_measuring = np.max(Q[state])
        MV += prob * max (0.0, q_measuring - Q[state,a_b])
    return MV
        