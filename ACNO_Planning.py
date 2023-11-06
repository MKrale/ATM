"""File containting all (R)ACNO-MDP planner used in the paper. RMDP-values are pre-computed in seperate files. """
import numpy as np
import math as m
import pytest
import cvxpy as cp
import AM_Gyms.DroneInCorridor as drone
# from functools import lru_cache

from AM_Gyms.AM_Tables import  RAM_Environment_Explicit
from AM_Gyms.AM_Env_wrapper import AM_ENV

# Globally used tolerances for floating numbers
rtol=0.0001
atol= 1e-10

cache_size = 1_000

class ACNO_Planner():
    """Generic ATM planner from Krale et al (2023)."""
    
    t = 0 # for debugging
    epsilon_measuring = 0 # 0.05
    loopPenalty = 1
    
    def __init__(self, Env:AM_ENV, tables:RAM_Environment_Explicit, use_robust:bool = False, df=0.95):
        
        self.env        = Env
        # Read (pre-computed) model description from 'tables'
        self.StateSize, self.ActionSize, self.cost, self.s_init = tables.get_vars()
        if use_robust:
            self.P, self.Q, _R = tables.get_robust_tables()
        else:
            self.P, _R, self.Q = tables.get_avg_tables()
        
        self.df          = df
        

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
        
        # ATM planning loop, with two minor adjustements:
        # 1) if beliefs do not change, we force a measuring action (to prevent infinite selfloops);
        # 2) for efficiency, we choose control actions for the next step before chosing measurement for the current step (since this is used to compute MV).
        
        self.env.reset()
        currentBelief, nextBelief = {self.s_init:1}, {}
        nextAction:int; currentAction:int; currentMeasuring:bool
        done = False
        total_reward, total_steps, total_measures = 0, 0, 0
        
        currentAction = self.determine_action(currentBelief)
        
        while not done:

            nextBelief = self.compute_next_belief (currentBelief, currentAction)
            if currentBelief == pytest.approx(nextBelief, rtol, atol):
                if len(currentBelief) == 1:
                    currentAction = np.random.choice(self.ActionSize)
                currentMeasuring = True
                print("hey!")
            else:
                nextAction = self.determine_action      (nextBelief)
                currentMeasuring  = self.determine_measurement (currentBelief, currentAction, nextBelief, nextAction)
            reward, done = self.execute_action(currentAction, currentBelief, currentMeasuring)
            if currentMeasuring or np.random.random() < self.epsilon_measuring:
                nextBelief, cost = self.measure()
                nextAction = self.determine_action (nextBelief)
                total_measures += 1
            else:
                cost = 0
            total_reward += reward - cost
            total_steps += 1
            # if True: #currentMeasuring:
                # print(measuring_value(currentBelief, currentAction, self.Q), self.cost, measuring_value(currentBelief, currentAction, self.Q) > self.cost)
                # print(currentBelief, self.measure(), nextBelief, currentAction, currentMeasuring)
            print(self.measure(), reward)
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
        MV = measuring_value(b_next, a_next, self.Q)
        return MV > self.cost
    
    def execute_action(self, action, belief, measuring):
        reward, done = self.env.step(action)
        return reward, done
    
    def measure(self):
        s, cost = self.env.measure()
        return {s:1}, cost

class ACNO_Planner_Robust(ACNO_Planner):
    """The R-ATM algorithm. Uses the same loop as ACNO_Planner, but with altered functions to calculte beliefs & action-pairs"""
    
    def __init__(self, Env:AM_ENV, tables:RAM_Environment_Explicit, df=0.95):
        self.env        = Env
        # Read (pre-computed) model description and RMDP-values from 'tables'
        self.StateSize, self.ActionSize, self.cost, self.s_init = tables.get_vars()
        self.Pmin, self.Pmax, _R = tables.get_uncertain_tables()
        self.P, self.Q, _R =  tables.get_robust_tables()
        
        self.df         = df
        self.epsilon_measuring = super().epsilon_measuring # = 0
        
    def determine_measurement(self, b, a, b_next=None, a_next=None):
        b_next_measuring = next_belief(b,a, self.P)
        if a_next is None:
            a_next = self.determine_action(b_next)
        MV = measuring_value(b_next, a_next, self.Q, bm=b_next_measuring)
        return MV > self.cost and not np.isclose(MV, 0, rtol=rtol, atol=atol)

    def determine_action(self, b):
        return optimal_action(b, self.Q, None)
    
    def compute_next_belief(self, b:dict, a:int):
        b_hashable = frozenset(b.items())
        return self.compute_next_belief_(b_hashable, a)
    
    # @lru_cache(maxsize=cache_size)
    def compute_next_belief_(self, b_hashable, a):
        b = {s:p for (s, p) in b_hashable}
        return custom_worst_belief(b, a, self.P, self.Pmin, self.Pmax, self.Q)
        

class ACNO_Planner_Control_Robust(ACNO_Planner_Robust):

    def __init__(self, Env:AM_ENV, PlanEnv:RAM_Environment_Explicit, MeasureEnv:RAM_Environment_Explicit, df=0.95):
        self.env = Env
        # Read (pre-computed) model description and RMDP-values from 'tables'
        self.StateSize, self.ActionSize, self.cost, self.s_init = PlanEnv.get_vars()
        self.P, self.Q, _R = PlanEnv.get_robust_tables()
        self.Pmin, self.Pmax, _R = PlanEnv.get_uncertain_tables()
        self.Pmeasure, self.Qmeasure, _R = MeasureEnv.get_robust_tables()
        
        self.df = df
        self.epsilon_measuring = super().epsilon_measuring # =0
        self.b_measure:dict = {}; self.b_measure_next:dict = {}
        self.s_init, _c = Env.measure()
    
    def run_episode(self):
        
        # Same basic control loop as for ACNO_Planner, but we now need to keep track of b_CR as well.
        
        self.env.reset()
        currentBelief, nextBelief = {self.s_init:1}, {}
        currentMeasureBelief, nextMeasureBelief = {self.s_init:1}, {}
        nextAction:int; currentAction:int; currentMeasuring:bool
        done = False
        total_reward, total_steps, total_measures = 0, 0, 0
        
        currentAction = self.determine_action(currentBelief)
        
        while not done:

            nextBelief = self.compute_next_belief (currentBelief, currentAction)
            nextMeasureBelief = self.compute_next_measure_belief(currentMeasureBelief, currentAction)
            if currentBelief == pytest.approx(nextBelief, rtol, atol):
                if len(currentBelief) == 1:
                    currentAction = np.random.choice(self.ActionSize)
                currentMeasuring = True
            else:
                nextAction = self.determine_action      (nextBelief)
                currentMeasuring  = self.determine_measurement (currentBelief, currentAction, currentMeasureBelief, nextBelief, nextAction, nextMeasureBelief) 
                    
            reward, done = self.execute_action(currentAction, currentBelief, currentMeasuring)
            if currentMeasuring: 
                nextBelief, cost = self.measure()
                nextMeasureBelief = nextBelief
                nextAction = self.determine_action (nextBelief)
                total_measures += 1
            else:
                cost = 0
            total_reward += reward - cost
            total_steps += 1
            currentBelief, currentAction, currentMeasureBelief = nextBelief, nextAction, nextMeasureBelief
        return total_reward, total_steps, total_measures
    
    def compute_next_measure_belief(self, b, a):
        return next_belief(b,a,self.Pmeasure)
        
    def determine_measurement(self, b, a, bm, b_next:None, a_next:None, bm_next:None):
        if b_next is None or a_next is None or bm_next is None:
            print("ERROR: determine_measurement not fully implemented for non-given next beliefs/actions")
        bnext_if_measuring = next_belief(b,a,self.P)
        #NOTE: since this is already 'less conservative' then the robust belief update, even control-robust ATM with P_Rmdp has an effect!
        MV_robust = measuring_value(bm_next, a_next, self.Qmeasure, Q_decision=self.Q )
        MV_extra  = measuring_value(b_next, a_next, self.Q, bm=bnext_if_measuring)
        
        return MV_robust > self.cost or MV_extra > self.cost


# Generalized functions:

def optimal_action(b:dict, Q1:np.ndarray, Q2:np.ndarray = None, returnvalue = False):
    """Returns the optimal action for belief state b as given by Q1.
    Ties are broken according to Q2, then randomly."""

    
    actionsize = np.shape(Q1)[1]
    thisQ1 = np.zeros(actionsize)
    
    for (state, prob) in b.items():
        
        thisQ1 += prob * Q1[state]
    
    thisQ1Max = np.max(thisQ1)
    filter = np.isclose(thisQ1, thisQ1Max, rtol=rtol, atol=atol)
    optimal_actions = np.arange(actionsize)[filter]

    
    if np.size(optimal_actions) > 1 and Q2 is not None:
        
        thisQ2 = np.zeros(actionsize)
        for (state, prob) in b.items():
            thisQ2 += prob * Q1[state]
            
        thisQ2Max = np.max(thisQ2[filter])
        filter2 = np.isclose(thisQ2, thisQ2Max, rtol=rtol, atol=atol)
        filtersCombined = np.logical_and(filter, filter2)
        optimal_actions = np.arange(actionsize)[filtersCombined]

    optimal_action = int(np.random.choice(optimal_actions))
    
    if returnvalue:
        return optimal_action, thisQ1[optimal_action]
    return optimal_action

def next_belief(b:dict, a:int, P:dict):
    """computes next belief state, according to current belief b, action a and transition function P."""
    
    b_next = {}
    
    for (state, beliefprob) in b.items():
        for (next_state, transitionprob) in P[state][a].items():
            if next_state in b_next:
                b_next[next_state] += beliefprob * transitionprob
            else:
                b_next[next_state] = beliefprob * transitionprob
    return b_next

def measuring_value(b:dict, a_b:int, Q:np.ndarray, bm:dict = None, Q_decision = None):
    """Returns the measuring value, assuming for future states action are chosen according to Q_decision, but real values are given by Q"""
    if Q_decision is None:
        Q_decision = Q
    if bm is None:
        bm = b
    MV = 0
    for (state, prob) in b.items():
        MV -= prob * Q[state, a_b]
    for (state, prob) in bm.items():
        a_m = np.argmax(Q_decision[state])
        MV += prob * Q[state, a_m]
    return MV

def index_flatten_P(statesize, s, snext):
    return snext + statesize * s

def index_unflatten_P(statesize, index):
    s = index // (statesize)
    snext = (index % statesize) // statesize
    return (s,snext)

def check_valid_P(P, Pmin, Pmax):
    """Checks whether or not transition function P is valid given uncertainty set Pmin/Pmax"""
    indexsize = np.size(P)
    for i in range(indexsize):
        if not (m.isclose(np.sum(P[i]), 1) and P[i] < Pmax[i] and P[i] > Pmin[i]):
            return False
    return True

def get_partial_b(state_indexes:np.ndarray, b:dict):
    """Returns belief state as np.ndarray, containing only those states specified in state_indexes"""
    b_array = np.zeros(np.size(state_indexes))
    for (i,s) in enumerate(state_indexes):
        if s in b:
            b_array[i] = b[s]
    return np.array(b_array)
    
def get_partial_P(state_indexes:np.ndarray, P:dict, a:int, flat = False):
    """Returns probability matrix as np.ndarray, containing only those transitions between states specified in state_indexes""" 
    statesize = np.size(state_indexes)
    if flat:
        P_array = np.zeros(statesize * statesize)
    else:
        P_array = np.zeros((statesize, statesize))
    
    for (s_i,s) in enumerate(state_indexes):
        for (snext_i, snext) in enumerate(state_indexes):
            if snext in P[s][a]:
                if flat:
                    P_array[index_flatten_P(statesize,s_i, snext_i)] = P[s][a][snext]
                else:
                    P_array[s_i,snext_i] = P[s][a][snext]
    return P_array

def b_array_to_dict(b_array:np.ndarray, indexes:np.ndarray, min_probability_considered:float = 0.001):
    b_array[b_array < min_probability_considered] = 0
    b_array = b_array / np.sum(b_array)
    
    b_dict = {}
    for (index, prob) in enumerate(b_array):
        if prob > 0:
            b_dict[indexes[index]] = prob
    return b_dict

def get_Ps_for_belief(state_indexes:np.ndarray, b:dict, a:int, P:dict, Pmin:dict, Pmax:dict):
    """Returns arrays of current belief and next possible beliefs, according to  """
    
    # Initialize arrays
    size = np.size(state_indexes)
    b_array = np.zeros(size)
    bnext, bnext_min, bnext_max = np.zeros(size), np.zeros(size), np.zeros(size)
    
    # For each non-zero b[s], add to b_array, then add P*b[s] to all bnext-arrays
    for (i,s) in enumerate(state_indexes):
        if s in b:
            b_array[i] = b[s]
            for (i_next, s_next) in enumerate(state_indexes):
                if s_next in Pmax[s][a]:
                    bnext     [i_next] += b[s] * P    [s][a][s_next]
                    bnext_min [i_next] += b[s] * Pmin [s][a][s_next]
                    bnext_max [i_next] += b[s] * Pmax [s][a][s_next]

    return (b_array, bnext, bnext_min, bnext_max)
            
def custom_worst_belief(b:dict, a:int, Pguess, Pmin:dict, Pmax:dict, Q:np.ndarray, min_probability_considered:float = 0.01):
    """Computes the worst-case next belief when taking action a form belief b, given the specified uncertain transition- and Q-value functions."""
    
    # Unpack P & Q into arrays of required sizes
    relevant_current_states, relevant_next_states = [], []
    for s in b.keys():
        relevant_current_states.append(s)
        for snext in Pmax[s][a].keys():
            relevant_next_states.append(snext)
    state_indexes = np.unique(np.append(relevant_current_states, relevant_next_states))
    statesize, actionsize = np.size(state_indexes), np.shape(Q)[1]
    
    Pmin_small   = get_partial_P(state_indexes, Pmin, a, flat=False)
    Pmax_small   = get_partial_P(state_indexes, Pmax, a, flat=False)
    Q_small = Q[state_indexes]
    b_array = get_partial_b(state_indexes, b)
        
    # Define as Convex problem & solve
    
    P = cp.Variable( (statesize,statesize) )
    bnext = cp.Variable(statesize)
    
    Qmax = cp.Variable()
    objective = cp.Minimize(Qmax)
    constraints = [bnext == b_array @ P]
    for a in range(actionsize):
        constraints.append(Qmax >= bnext @ Q_small[:,a])
    for (index, state) in enumerate(state_indexes):
        if state in b:
            constraints.append(cp.sum(P[index]) == 1)
            constraints.append(P[index] >= Pmin_small[index])
            constraints.append(P[index] <= Pmax_small[index])
    
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.GLPK, kwargs={"time_limit_sec":1})
    except cp.error.SolverError:
        bnext.value = None
    if bnext.value is None:
        print("Error: solver failed!: {}".format(problem.status))
        return next_belief(b,a,Pguess)
    

    b_worst_dict = b_array_to_dict(bnext.value, state_indexes)
    return b_worst_dict