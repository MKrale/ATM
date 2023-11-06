


from AM_Gyms.AM_Env_wrapper import AM_ENV
import numpy as np

class ModelLearner():
    
    
    def __init__(self, env:AM_ENV, df = 0.95, record_done = None):
         
         self.env = env
         
         self.StateSize, self.ActionSize, self.cost, self.s_init = env.get_vars()
         print(self.StateSize)
         self.doneState = self.StateSize
         self.StateSize += 1
         self.df = df
         self.fullStepUpdate = True
         
         if record_done == None:
             record_done = (env.horizon() is None)
         self.record_done = record_done
         self.init_model()
    
    
    def init_model(self):
        
        self.counter = np.zeros((self.StateSize, self.ActionSize))
        self.P = build_dictionary(self.StateSize, self.ActionSize)
        for a in range(self.ActionSize):
            self.P[self.doneState][a] = {self.doneState:1}
        self.P_counter = build_dictionary(self.StateSize, self.ActionSize)
        
        self.R_counter = build_dictionary(self.StateSize, self.ActionSize)
        self.R = build_dictionary(self.StateSize, self.ActionSize)
        self.R_expected = np.zeros((self.StateSize, self.ActionSize))
        
        for a in range(self.ActionSize):
            self.P_counter[self.doneState][a][self.doneState] = 1
            self.R_counter[self.doneState][a][self.doneState] = 0

        self.Q = np.zeros((self.StateSize, self.ActionSize))
        self.Q[self.doneState, :] = 0
        self.Q_max = np.zeros((self.StateSize))
        
        self.Q_learning = np.ones((self.StateSize, self.ActionSize))
        self.Q_learning[self.doneState, :] = 0
        self.Q_learning_max = np.max(self.Q_learning, axis=1)
        self.df_learning = 0.90

    def get_model(self):
        """Returns P, R, Q"""
        return self.P, self.R, self.Q
    
    def run_visits(self, min_visits = 100, max_eps = np.inf, logging = True):
        
        i = 0
        final_updates = 100
        batch_size = 2500
        nmbr_batches = 0
        min_visists = 5
        done = False
        if logging:
            print("Learning MDP-model started:")
        while not done :
            nmbr_batches += 1
            s_avg = 0
            for i in range(batch_size):
                _r, s, _m = self.run_episode()
                s_avg += s/batch_size
            counter_nonzero = np.nonzero(self.counter)
            done = np.min(self.counter[counter_nonzero]) > min_visits or nmbr_batches*batch_size > max_eps
            print(np.argmin(self.counter[counter_nonzero]), np.size(self.counter[counter_nonzero]))
            if logging:
                print("{} episodes completed (with {} avg steps)".format(batch_size*nmbr_batches, s_avg))
        if self.record_done:
            self.insert_done_transitions()
        print("Learning completed in {} episodes!\n\n".format(i))
        
        
        for i in range(final_updates):
            self.update_model()
            
    def run_setStates(self, SA_updates = 100, logging = True):
        if logging:
            print("Learning MDP-model started:")
        
        for i in range(SA_updates):
            
            sorted_states = np.argsort(self.Q_max[:-1])
            for s in sorted_states:
                for a in range(self.ActionSize):
                    # Set env to state, take action & update model.
                    self.env.set_state(s)
                    reward, done = self.env.step(a)
                    if done:
                        print(s, reward)
                        snext = self.doneState
                    else:
                        (snext,_cost) = self.env.measure()
                        
                    self.update_counters(s, a, snext, reward)
                    self.update_model([(s,a)])
            print("{} iterations completed".format(i))
        
    def run_episode(self):
        self.env.reset()
        done = False
        (s, _cost) = self.env.measure()
        totalreward, totalsteps = 0, 0
        
        while not done:
            
            a = np.argmax(self.Q_learning[s])
            reward, done = self.env.step(a)
            if done:
                snext = self.doneState
            else:
                (snext,_cost) = self.env.measure()
                
            self.update_counters(s, a, snext, reward)
            self.update_model([(s,a)])
            
            totalreward += reward; totalsteps += 1
            s = snext
        return totalreward, totalsteps, totalsteps
        
        
    def update_counters(self, s, a, snext, reward):
        
        # Update counters
        self.counter[s,a] += 1
        if snext in self.P[s][a]:
            self.P_counter[s][a][snext] += 1
            self.R_counter[s][a][snext] += reward
        else:
            self.P_counter[s][a][snext] = 1
            self.P[s][a][snext] = 1
            self.R_counter[s][a][snext] = reward
            self.R[s][a][snext] = 1
        
    
    def update_model(self, state_action_pairs, include_learning = True):
        
        
        if state_action_pairs is not None:
            for (s,a) in state_action_pairs:
                
                Psi, Psi_learning, self.R_expected[s,a] = 0, 0, 0
                self.R_expected[s,a]
                Q_next, Psi_learning_next = 0, 0
                for (s_next, _p) in self.P[s][a].items():
                    self.R[s][a][s_next] = self.R_counter[s][a][s_next] / self.counter[s,a]
                    self.P[s][a][s_next] = self.P_counter[s][a][s_next] / self.counter[s,a]
                    self.R[s][a][s_next] = self.R_counter[s][a][s_next] / self.P_counter[s][a][s_next]
                    
                    Q_next += self.P[s][a][s_next] * (self.R[s][a][s_next] + self.df * self.Q_max[s_next])
                    Psi_learning_next += self.P[s][a][s_next] * self.Q_learning_max[s_next]
                
                # Update Q-values
                Q_prev, Q_learning_prev = self.Q[s,a], self.Q_learning[s,a]
                self.Q[s,a] = Q_next
                self.Q_learning[s,a] = 1/self.counter[s,a] + self.df_learning * Psi_learning
                
                # update Qmax-values (without iteration over actions)
                if Q_prev == self.Q_max[s] and Q_prev < Q_next:
                    self.Q_max[s] = np.max(self.Q[s])
                else:
                    self.Q_max[s] = max([self.Q_max[s], Q_next])
                
                if include_learning and Q_learning_prev == self.Q_learning_max[s] and Q_learning_prev < self.Q_learning[s,a] :
                    self.Q_learning_max[s] = np.max(self.Q_learning[s])
                else:
                    self.Q_learning_max[s] = max([self.Q_learning_max[s], self.Q_learning[s,a]])
                    
    
    def insert_done_transitions(self):

        for s in range(self.StateSize):
            for a in range(self.ActionSize):
                if not self.P[s][a]:
                    self.P[s][a][self.doneState] = 1

def build_dictionary(statesize, actionsize, array:np.ndarray = None):
    dict = {}
    for s in range(statesize):
        dict[s] = {}
        for a in range(actionsize):
            dict[s][a] = {}
            if array is not None:
                for snext in range(statesize):
                    dict[s][a][snext] = array[s,a,snext]
    return dict