import numpy as np


from ICVaR_MDP import ICVaR
from AM_Gyms.ModelLearner import ModelLearner
from AM_Gyms.AM_Env_wrapper import AM_ENV

class ACNO_Planning():
    """Class for planning in ACNO-MDP environements. In contrast to Dyna-ATM, in this method
    we assume the model dynamics are known (i.e. we do not do RL)."""
    
    def __init__(self, Env:AM_ENV ):
        
        # Unpacking variables from environment:
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = self.env.get_vars()
        print(self.ActionSize)
        self.StateSize  += 1 #include done state
        self.doneState  = self.StateSize -1
        
        # Setting up tables:
        self.P          = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )
        self.R          = np.zeros( (self.StateSize, self.ActionSize) )
        self.Q          = np.zeros( (self.StateSize, self.ActionSize) )
        self.Q_max      = np.zeros( self.StateSize)
        
        
        # Other variables:
        self.df = 0.95
        self.epsilon = 0
        
        # TODO: Add ICVaR, ICVaR_max, alpha!
        
    def learn_model(self, eps = 10_000, logging = False):
        model = ModelLearner(self.env, self.df)
        model.sample(eps, logging=logging)
        self.P, R, _ = model.get_model()
        self.R = R[:,self.ActionSize:]
        self.Q = model.get_Q()[:,self.ActionSize:]
        self.Q_max = np.max(self.Q, axis = 1)
    
    #######################################################
    ###                 MAIN LOOP FUNCTIONS:            ###
    #######################################################  
 
    def run(self, eps, logging = False):
        
        self.learn_model(logging=logging)
        rewards, steps, measurements = np.zeros(eps), np.zeros(eps), np.zeros(eps)
        
        for i in range(eps):
            self.run_episode()
            rewards[i], steps[i], measurements[i] = self.r, self.steps_taken, self.measurements_taken
            
        return (np.sum(rewards), rewards, steps, measurements)
            
    def run_episode(self):
        
        self.initialise_episode()
        
        while not self.done:
            self.determine_action()
            self.guess_next_state()
            self.determine_measurement()
            self.take_action()
            self.take_measurement()
            self.update_model()
            self.update_step_vars()
  
    #######################################################
    ###                 LOOP STEPS FUNCTIONS:           ###
    #######################################################          

    def initialise_episode(self):
        self.env.reset()
        self.b = np.zeros(self.StateSize)
        self.b[self.s_init] = 1
        self.b_dict = {}
        self.b_dict[self.s_init] = 1
        self.done=False
        self.c = 0
        
        # Logging variables
        self.measurements_taken = 0
        self.episode_reward     = 0
        self.steps_taken        = 0

    
    def update_model(self):
        # In this version, we assume the model is already known, so we do not need to do any updating
        pass
    
    def update_step_vars(self):
        self.b = self.b_next
        self.b_dict = self.b_next_dict
        self.episode_reward += self.r - self.c
        self.steps_taken += 1
        if self.m:
            self.measurements_taken += 1    
    
    def determine_action(self):
        self.a = self.determine_action_general(self.b_dict)
    
    def determine_action_general(self, b):
        thisQ = np.zeros(self.ActionSize)
        # Determine 'Q-table' for current belief
        for s in b:
            p = b[s]
            thisQ += self.Q[s]*p
        # Choose optimal action according to thisQ, break ties randomly
        thisQMax = np.max(thisQ)
        return int(np.random.choice(np.where(np.isclose(thisQ, thisQMax))[0]))
    
    def guess_next_state(self):
        self.b_next, self.b_next_dict =  self.guess_next_state_general(self.b, self.a)
        
    def guess_next_state_general(self, b, a):
        """Returns the next state given b and a, both as a np-array and dictionary"""
        b_next = np.matmul(b,self.P[:,a,:]) #does this work?
        filter = np.nonzero(b_next)
        
        b_next_dict = {}
        for s,p in zip(np.arange(self.StateSize)[filter], b_next[filter]):
            b_next_dict[s] = p

        return b_next, b_next_dict
    
    def take_action(self):
        if np.random.rand() < self.epsilon:
            self.a = np.random.randint(self.ActionSize)
            self.m = True
        self.r, self.done = self.env.step(self.a)
    
    def take_measurement(self):
        if self.m:
            s_next, cost = self.env.measure()
            self.b_next.clear()
            self.b_next[s_next] = 1
    
    def determine_measurement(self):
        self.a_next = self.determine_action_general(self.b_next_dict)
        MV = self.get_MV_general(self.b_next_dict, self.a_next)
        self.m = MV >= self.cost
        
    def get_MV_general(self, b, a):
        MV = 0
        for s in b:
            p = b[s]
            qmax = np.max(self.Q[s])
            MV += p* max(0.0, qmax - self.Q[s,a])
        return MV


# from AM_Gyms.frozen_lake_v2 import FrozenLakeEnv_v2
# from AM_Gyms.frozen_lake import FrozenLakeEnv

# # Semi-slippery, larger   
# # Env = AM_ENV( FrozenLakeEnv_v2(map_name = "8x8", is_slippery=True), StateSize=64, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# # semi-slippery, small
# # Env = AM_ENV( FrozenLakeEnv_v2(map_name = "4x4", is_slippery=True), StateSize=16, ActionSize=4, s_init=0, MeasureCost = 0.1 )

# # slippery, small
# Env = AM_ENV( FrozenLakeEnv(map_name = "4x4", is_slippery=True), StateSize=16, ActionSize=4, s_init=0, MeasureCost = 0.1 )


# planner = ACNO_Planning(Env)
# print(planner.run(1000, logging = True))