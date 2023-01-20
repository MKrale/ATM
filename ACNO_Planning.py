import numpy as np


from AM_Gyms.ModelLearner import ModelLearner
from AM_Gyms.AM_Env_wrapper import AM_ENV

class ACNO_Planner():
    """Class for planning in ACNO-MDP environements. In contrast to Dyna-ATM, in this method
    we assume the model dynamics are known (i.e. we do not do RL)."""
    
    def __init__(self, Env:AM_ENV ):
        
        # Unpacking variables from environment:
        self.env        = Env
        self.StateSize, self.ActionSize, self.cost, self.s_init = self.env.get_vars()
        self.StateSize  += 1 #include done state
        self.doneState  = self.StateSize -1
        
        # Setting up tables:
        self.P          = np.zeros( (self.StateSize, self.ActionSize, self.StateSize) )
        self.R          = np.zeros( (self.StateSize, self.ActionSize) )
        self.Q          = np.zeros( (self.StateSize, self.ActionSize) )
        self.Q_max      = np.zeros( self.StateSize)
        
        
        # Other variables:
        self.df = 0.80
        self.epsilon = 0
        self.particles = 100
        
        # TODO: Add ICVaR, ICVaR_max, alpha!
        
    def learn_model(self, eps = 5_000, logging = False):
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
        
        if logging:
            print("starting learning of model...")
        self.learn_model(logging=logging)
        print(self.Q)
        rewards, steps, measurements = np.zeros(eps), np.zeros(eps), np.zeros(eps)
        log_nmbr = 100
        
        if logging:
            print("Start planning:")
        
        for i in range(eps):
            self.run_episode()
            rewards[i], steps[i], measurements[i] = self.r, self.steps_taken, self.measurements_taken
            if (i > 0 and i%log_nmbr == 0 and logging):
                print ("{} / {} runs complete (current avg reward = {}, nmbr steps = {}, nmbr measures = {})".format( 
                        i, eps, np.average(rewards[(i-log_nmbr):i]), np.average(steps[(i-log_nmbr):i]), np.average(measurements[(i-log_nmbr):i]) ) )
            
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
        for s in self.b_dict:
            self.Q[s,self.a] = self.df * np.sum(self.P[s,self.a] * self.Q_max) + self.R[s,self.a]
            self.Q_max[s] = np.max(self.Q[s])
    
    def update_step_vars(self):
        if not self.done:
            self.b_dict , self.b = self.check_validity_belief(self.b_next_dict, self.b_next)
            self.episode_reward += self.r - self.c
            self.steps_taken += 1
            if self.m:
                self.measurements_taken += 1
            
    def check_validity_belief(self, b_dict:dict, b_array = None):
        
        # Check if belief includes done-states
        scaling_factor = 1
        if self.doneState in b_dict:
            p_done = b_dict[self.doneState]
            if p_done == 1:
                print("Warning: belief state contains only impossible states!")
            else:
                b_dict.pop(self.doneState)
                scaling_factor = 1 / (1-p_done) 
        for s in b_dict:
            b_dict[s] = b_dict[s] * scaling_factor
        if b_array is not None:
            b_array = b_array * scaling_factor
            return b_dict, b_array
        return b_dict
            
    
    def determine_action(self):
        self.a = self.determine_action_general(self.b_dict, self.Q)
    
    def determine_action_general(self, b, Q):
        thisQ = np.zeros(self.ActionSize)
        # Determine 'Q-table' for current belief
        for s in b:
            p = b[s]
            thisQ += Q[s]*p
        # Choose optimal action according to thisQ, break ties randomly
        thisQMax = np.max(thisQ)
        return int(np.random.choice(np.where(np.isclose(thisQ, thisQMax))[0]))
    
    def guess_next_state(self):
        self.b_next, self.b_next_dict =  self.guess_next_state_general(self.b_dict, self.a, self.P)
        
    def guess_next_state_general(self, b, a, P):
        """Returns the next state given b and a, both as a np-array and dictionary"""
        b_next = np.zeros(self.StateSize)
        for s in b:
            b_next += P[s,a] * b[s]
        filter = np.nonzero(b_next)
        
        
        states, probs = np.arange(self.StateSize)[filter], b_next[filter]
        b_next_discretized = np.random.choice(states, size=self.particles, p=probs)
        states, counts = np.unique(b_next_discretized, return_counts = True)
        
        b_next_dict = {}
        for i in range(len(states)):
            s, p = states[i], counts[i] * 1/self.particles
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
            self.b_next_dict.clear()
            self.b_next_dict[s_next] = 1
            
            self.b_next = np.zeros(self.StateSize)
            self.b_next[s_next] = 1
            
    
    def determine_measurement(self):
        self.a_next = self.determine_action_general(self.b_next_dict, self.Q)
        MV = self.get_MV_general(self.b_next_dict, self.a_next, self.Q)
        self.m = MV >= self.cost
        
    def get_MV_general(self, b, a, Q):
        MV = 0
        for s in b:
            p = b[s]
            qmax = np.max(Q[s])
            MV += p* max(0.0, qmax - Q[s,a])
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