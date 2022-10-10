import numpy as np

from AM_Gyms.AM_Env_wrapper import AM_ENV


class ModelLearner():
    """Class for learning ACNO-MDP """

    def __init__(self, env:AM_ENV):
        # Set up AM-environment
        self.env = env
        
        # Model variables
        self.StateSize, self.CActionSize, self.cost, self.s_init = env.get_vars()
        self.StateSize += 1
        self.ActionSize = self.CActionSize * 2  #both measuring & non-measuring actions
        self.EmptyObservation = self.StateSize +100
        self.doneState = self.StateSize -1
        
        self.init_model()

    def init_model(self):
        # Model tables:
        self.counter = np.zeros((self.StateSize, self.ActionSize)) + 1
        self.T = np.zeros((self.StateSize, self.ActionSize, self.StateSize))
        self.T_counter = np.zeros((self.StateSize, self.ActionSize, self.StateSize)) + 1/self.StateSize
        self.T_counter[self.doneState,:,:] = 0
        self.T_counter[self.doneState,:,self.doneState] = 1
        self.R = np.zeros((self.StateSize, self.ActionSize))
        self.R_counter = np.zeros((self.StateSize, self.ActionSize))
        
        # Variables for learning:
        self.Q = 1/self.counter[:,:self.CActionSize]
        self.lr = 1
        self.df = 0.8
    
    def get_model(self):
        """Returns T & R"""
        return self.T, self.R
    
    def get_vars(self):
        """returns StateSize, ActionSize, cost, s_init, doneState"""
        return (self.StateSize, self.ActionSize, self.cost, self.s_init, self.doneState)
    
    def filter_T(self):
        """Filters all transitions with p<1/|S| from T"""
        p = 1/self.StateSize
        # mask = self.T_counter<p*self.counter[:,:,np.newaxis]
        mask = self.T < p
        self.T[mask] = 0
        self.T = self.T / np.sum(self.T, axis =2)[:,:,np.newaxis]        
    
    def sample(self, N, max_steps = 500, logging = True):
        """Learns the model using N episodes, returns episodic costs and steps"""
        # Intialisation
        self.init_model()
        self.sampling_rewards = np.zeros(N)
        self.sampling_steps = np.zeros(N)
        
        for eps in range(N):
            self.sample_episode(eps, max_steps)
            if eps % 100 == 0 and logging:
                print("{} exploration episodes completed!".format(eps))
        self.filter_T()
        return self.sampling_rewards, self.sampling_steps
    
    def sample_episode(self, episode, max_steps):
        """Samples one episode, following method proposed in https://hal.inria.fr/hal-00642909"""
        self.env.reset()
        done = False
        (s_prev, _cost) = self.env.measure()
        for step in range(max_steps):
            
            # Greedily pick action from Q
            a = np.argmax(self.Q[s_prev])
            
            # Take step & measurement
            reward, done = self.env.step(a)
            if done:
                s = self.doneState
            else:
                (s, cost) = self.env.measure()
                
            # Update logging variables
            self.sampling_rewards[episode] += reward - self.cost
            self.sampling_steps[episode] += 1
            
            # Update model
            self.update_step(s_prev, a, s, reward) 
            
            # Update learning Q-table
            CAS = self.CActionSize
            Psi = np.sum(self.T[s_prev,:CAS] * np.max(self.Q, axis=1), axis=1) #axis?
            self.Q[s_prev] = (1-self.lr)*self.Q[s_prev] + self.lr*(1/self.counter[s_prev,:CAS] + self.df*Psi )
            s_prev = s
            if done:
                break
    
    def update_step(self, s_prev, a, s_next, reward):
        ac, ao = a % self.CActionSize, a // self.CActionSize
        
        # update measuring action counters
        self.counter[s_prev,ac] += 1
        self.T_counter[s_prev,ac,s_next] += 1
        self.R_counter[s_prev,ac] += reward - self.cost
        # update non-measuring actions counters
        anm = ac + self.CActionSize
        self.counter[s_prev,anm] += 1
        self.T_counter[s_prev,anm,s_next] += 1
        self.R_counter[s_prev,anm] += reward
        
        # update model
        self.T = self.T_counter / self.counter[:,:,np.newaxis]
        self.R = self.R_counter / self.counter
        
    def reset_env(self):
        self.env.reset()
        
    def real_step(self, action):
        return self.env.step(action)
    
    def measure_env(self):
        return self.env.measure()