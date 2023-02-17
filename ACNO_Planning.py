import numpy as np
from collections import deque 


from AM_Gyms.AM_Tables import AM_Environment_tables, RAM_Environment_tables
from AM_Gyms.AM_Env_wrapper import AM_ENV

class ACNO_Planner():
    
    def __init__(self, Env:AM_ENV, tables = AM_Environment_tables, df=0.95):
        
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = tables.get_vars()
        self.P, _R, self.Q = tables.get_tables()

        self.df          = df
        self.epsilon     = 0
        self.loopPenalty = 0.95
    
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
        
        self.env.reset()
        currentBelief, nextBelief = {0:1}, {}
        nextAction:int; currentAction:int; currentMeasuring:bool
        done = False
        total_reward, total_steps, total_measures = 0, 0, 0
        
        currentAction = self.determine_action(currentBelief)
        
        while not done:

            nextBelief = self.compute_next_belief   (currentBelief, currentAction)
            if currentBelief == nextBelief:
                currentMeasuring = True
            else:
                nextAction = self.determine_action      (nextBelief)
                currentMeasuring  = self.determine_measurement (currentBelief, currentAction, nextBelief, nextAction) 
            
            reward, done = self.execute_action(currentAction, currentBelief, currentMeasuring)
            if currentMeasuring:
                nextBelief, cost = self.measure()
                nextAction = self.determine_action (nextBelief)
                total_measures += 1
            else:
                cost = 0
            #print(currentBelief, currentAction, currentMeasuring, nextBelief)
            total_reward += reward - cost
            total_steps += 1
            currentBelief, currentAction = nextBelief, nextAction
        
        return total_reward, total_steps, total_measures
    
    
    def determine_action(self, b):
        return optimal_action(b, self.Q, None)
    
    def compute_next_belief(self, b, a):
        return next_belief(b,a,self.P)
    
    def determine_measurement(self, b, a, b_next=None, a_next=None):
        if b_next is None:
            b_next = self.compute_next_belief(b,a)
        if a_next is None:
            a_next = self.determine_action(b_next)
        return measuring_value(b_next, a_next, self.Q) > self.cost
    
    def execute_action(self, action, belief, measuring):
        reward, done = self.env.step(action)
        return reward, done
    
    def measure(self):
        s, cost = self.env.measure()
        return {s:1}, cost

class ACNO_Planner_Robust(ACNO_Planner):
    
    def __init__(self, Env:AM_ENV, tables = RAM_Environment_tables, df=0.95):
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = tables.get_vars()
        self.PReal, _R, self.QReal = tables.get_tables()
        self.P, self.Q = tables.get_robust_MDP_tables()
        
        self.df         = df
        self.epsilon    = 0
    
    def determine_action(self, b):
        return optimal_action(b, self.Q, self.QReal)

class ACNO_Planner_Control_Robust(ACNO_Planner_Robust):

    def determine_measurement(self, b, a, b_next, a_next):
        b_next_real = next_belief(b,a,self.PReal)
        return (measuring_value(b_next_real, a_next, self.QReal, self.Q) > self.cost
                or  measuring_value(b_next, a_next, self.Q) > self.cost )


# Generalized functions:

def optimal_action(b:dict, Q1:np.ndarray, Q2:np.ndarray = None, loopPenalty = None):
    """Returns the optimal action for belief state b as given by Q1.
    Ties are broken according to Q2, then randomly."""
    if loopPenalty == None:
        loopPenalty = 1
    
    actionsize = np.shape(Q1)[1]
    thisQ1 = np.zeros(actionsize)
    
    for (state, prob) in b.items():
        
        thisQ1 += prob * Q1[state]
    
    thisQ1Max = np.max(thisQ1)
    filter = np.isclose(thisQ1, thisQ1Max, rtol=0.01, atol= 0.001)
    optimal_actions = np.arange(actionsize)[filter]

    
    if np.size(optimal_actions) > 1 and Q2 is not None:
        
        thisQ2 = np.zeros(actionsize)
        for (state, prob) in b.items():
            thisQ2 += prob * Q1[state]
            
        thisQ2Max = np.max(thisQ2[filter])
        filter2 = np.isclose(thisQ2, thisQ2Max, rtol=0.001, atol= 0.0001)
        filtersCombined = np.logical_and(filter, filter2)
        optimal_actions = np.arange(actionsize)[filtersCombined]

    
    return int(np.random.choice(optimal_actions))

def next_belief(b:dict, a:int, P:np.ndarray, min_probability_considered:float = 0.001):
    """computes next belief state, according to current belief b, action a and transition function P.
    All belief probabilities smaller than min_probability_considered are ignored (for efficiency reasons)."""
    
    statesize = np.shape(P)[0]
    b_next_array = np.zeros(statesize)
    
    for (state, prob) in b.items():
        b_next_array += P[state,a] * prob
    
    filter = b_next_array >= min_probability_considered
    states = np.arange(statesize)[filter]
    b_next_array = b_next_array[filter]
    b_next_array = b_next_array / np.sum(b_next_array)

    
    b_next = dict()
    for (state, prob) in zip(states, b_next_array):
        b_next[state] = prob
    
    return b_next

def measuring_value(b:dict, a_b:int, Q:np.ndarray, Q_decision = None):
    if Q_decision is None:
        Q_decision = Q
    MV = 0
    for (state, prob) in b.items():
        a_m = np.argmax(Q_decision[state])
        MV += prob * max (0.0, Q[state, a_m] - Q[state,a_b])
    return MV