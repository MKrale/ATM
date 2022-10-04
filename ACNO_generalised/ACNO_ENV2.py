
from AM_Env_wrapper import AM_ENV
import numpy as np
from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool
from pomdpy.discrete_pomdp import DiscreteAction
from pomdpy.discrete_pomdp import DiscreteState
from pomdpy.discrete_pomdp import DiscreteObservation
from pomdpy.pomdp import HistoricalData
from pomdpy.pomdp import Model, StepResult

    #########################################
    #       Data classes for POMDP:             #
    #########################################

class PositionData(HistoricalData):
    def __init__(self, model, position, solver):
        self.model = model
        self.solver = solver
        self.position = position
        self.legal_actions = self.generate_legal_actions 

    def generate_legal_actions(self):
        legal_actions = []
        for i in range(self.model.actions_n):
            legal_actions.append(i)
        return legal_actions

    def shallow_copy(self):
        return PositionData(self.model, self.position, self.solver)
    
    def copy(self):
        return self.shallow_copy()

    def update(self, other_belief):
        pass

    def create_child(self, action, observation):
        next_data = self.copy() # deep/shallow copy
        next_position, is_legal = self.model.make_next_position(self.position, action.bin_number)
        next_data.position = next_position
        return next_data

class BoxState(DiscreteState):
    def __init__(self, position, is_terminal=False, r=None):
        self.position = position
        self.terminal = is_terminal
        self.final_rew = r

    def __eq__(self, other_state):
        return self.position == other_state.position

    def copy(self):
        return BoxState(self.position, 
                        is_terminal=self.terminal, 
                        r=self.final_rew)
        
    def to_string(self):
        return str(self.position)

    def print_state(self):
        pass

    def as_list(self):
        pass

    def distance_to(self):
        pass

class BoxAction(DiscreteAction):
    def __init__(self, bin_number):
        self.bin_number = bin_number

    def __eq(self, other_action):
        return self.bin_number == other_action.bin_number

    def print_action(self):
        pass

    def to_string(self):
        return str(self.bin_number)

    def distance_to(self):
        pass

    def copy(self):
        return BoxAction(self.bin_number)

class BoxObservation(DiscreteObservation):
    def __init__(self, position):
        self.position = position
        self.bin_number = position

    def __eq__(self, other_obs):
        return self.position == other_obs.position

    def copy(self):
        return BoxObservation(self.position)

    def to_string(self):
        return str(self.position)

    def print_observation(self):
        pass

    def as_list(self):
        pass

    def distance_to(self):
        pass


#########################################
    #       The ENV:             #
#########################################



class ACNO_ENV():
    """
    Class acting as a wrapper around AM_ENV environments (i.e. openAI gyms
    with a AM-wrapper), which builds a model that can be used by POMCP by
    sampling from the environment offline.
    """
    
    #########################################
    #       Sampling the model:             #
    #########################################
    
    def __init__(self, env:AM_ENV):
        
        # Environment & env-related variables (BAM-QMDP naming)
        self.env = env
        
        self.StateSize, self.CActionSize, self.cost, self.s_init = env.get_vars()
        self.StateSize += 1
        self.ActionSize = self.CActionSize * 2  #both measuring & non-measuring actions
        self.EmptyObservation = -1
        
        # Renaming for POMCP-algo:
        self.init_state, self.actions_n, self.states_n = self.s_init, self.ActionSize, self.StateSize
        # Other POMCP-variables
        self.solver = 'POMCP'
        self.n_start_states = 200
        self.ucb_coefficient = 0 #no optimism required
        self.n_sims = 0 #set by agent
        self.seed = np.random.seed() 
        self.min_particle_count = 180
        self.max_particle_count = 220
        self.max_depth = 20
        self.action_selection_timeout = 120
        self.particle_selection_timeout = 0.2
        self.n_sims = 200
        self.preferred_actions = False
        self.timeout = 7_200_000
        self.discount = 1
        
        self.doneState = self.StateSize -1
        self.sampling_rewards = []
        self.sampling_steps = []
        
    def init_model(self):
        self.counter = np.zeros((self.StateSize, self.ActionSize)) + 1
        self.T = np.zeros((self.StateSize, self.ActionSize, self.StateSize))
        self.T_counter = np.zeros((self.StateSize, self.ActionSize, self.StateSize)) + 1/self.StateSize
        self.R = np.zeros((self.StateSize, self.ActionSize))
        self.R_counter = np.zeros((self.StateSize, self.ActionSize))
    
    def filter_T(self):
        """Filters all transitions with p<1/|S| from T"""
        p = 1/self.StateSize
        # mask = self.T_counter<p*self.counter[:,:,np.newaxis]
        mask = self.T < p
        self.T[mask] = 0
        self.T = self.T / np.sum(self.T, axis =2)[:,:,np.newaxis]        
    
    def sample_model_AL(self, N, max_steps = 500):
        """Learns the model according to method proposed in https://hal.inria.fr/hal-00642909"""
        
        # Intialisation
        self.init_model()
        Q = 1/self.counter[:,:self.CActionSize]
        lr = 1
        df = 0.8
        self.sampling_rewards = np.zeros(N)
        self.sampling_steps = np.zeros(N)
        AS = self.CActionSize
        
        for walk in range(N):
            self.env.reset()
            done = False
            s_prev = self.s_init
            for step in range(max_steps):
                
                # Greedily pick action from Q
                a = np.argmax(Q[s_prev])
                
                # Take step & measurement
                reward, done = self.env.step(a)
                if done:
                    s = self.doneState
                else:
                    (s, cost) = self.env.measure()
                    
                # Update logging variables
                self.sampling_rewards[walk] += reward - self.cost
                self.sampling_steps[walk] += 1
                
                # Update model
                self.update_model(s_prev, a, s, reward)
                
                # Update learning Q-table
                Psi = np.sum(self.T[s_prev,:AS] * np.max(Q, axis=1), axis=1) #axis?
                Q[s_prev] = (1-lr)*Q[s_prev] + lr*(1/self.counter[s_prev,:AS] + df*Psi )
                s_prev = s
                if done:
                    break
            if walk % 100 == 0:
                print("{} exploration episodes completed!".format(walk))
        self.filter_T()
        return self.sampling_rewards, self.sampling_steps
    
    def update_model(self, s_prev, a, s_next, reward):
        ac, ao = a % self.CActionSize, a // self.CActionSize
        
        # update measuring actions 
        self.counter[s_prev,ac] += 1
        self.T_counter[s_prev,ac,s_next] += 1
        self.R_counter[s_prev,ac] += reward - self.cost
        #update non-measuring actions
        anm = ac + self.CActionSize
        self.counter[s_prev,anm] += 1
        self.T_counter[s_prev,anm,s_next] += 1
        self.R_counter[s_prev,anm] += reward
        
        self.T = self.T_counter / self.counter[:,:,np.newaxis]
        self.R = self.R_counter / self.counter
        
    #########################################
    #       POMDP-model functions:             #
    #########################################
    
    def reset():
        '''I don't quiet now what uses this function: TBW!''' #TODO
        pass
    
    def generate_particles(self, prev_belief, action, obs, n_particles, prev_particles, mdp=False):
        '''Sample new belief particles according to model approximation.'''
        if type(obs) is not int:
            obs = obs.position
        if type(action) is not int:
            action = action.bin_number
        # If obs not empty, than we know the next state for sure: return that!
        if obs != self.StateSize:
            print("Unimplemented!")
            pass    # TODO: return just the observation!
            
        new_particles = []
        
        # Otherwise, sample new states according to model:
        while new_particles.__len__() < n_particles:
            prev_state = np.random.choice(prev_particles).position
            next_state = np.random.choice(np.arange(self.StateSize), p=self.T[prev_state,action])
            terminal = (next_state == self.doneState)
            
            new_particles.append(BoxState(next_state, terminal))
        
        # The original has changes to self-sampling if time runs to high: I don't expect that to be necessary...
        return new_particles

    def model_step(self, state, action):
        """Estimates the next state and reward, using exisiting model."""
        next_state = np.random.choice(self.StateSize, p=self.T[state,action])
        reward = self.R[state,action]
        done = False
        if next_state == self.doneState:
            done = True
        return next_state, reward, done
    
    def take_real_step(self, action, ignoreMeasuring = True, asStepResults = True):
        """Takes a real step in the environement, returns (reward, done). """
        # Take action
        ac, ao = action % self.CActionSize, action // self.CActionSize
        (reward, done) = self.env.step(ac)
        
        #Measure (if applicable)
        if not ignoreMeasuring:
            if action < self.CActionSize: # measuring:
                obs, c = self.env.measure()
                reward -= c
            else:
                obs = self.EmptyObservation
            print(action, obs, reward)
            return(reward, obs, done)
        return  (reward, done)

    def generate_step(self, state, action, is_mdp=False, real = False):
        '''As used by POMCP: models a step & return in POMCP-format'''
        # Unpack actions and states if required
        if type(action) is not int:
            action = action.bin_number
        if type(state) is not int:
            state = state.position
        
        # Simulate a step:
        (next_state, reward, done) = self.model_step(state, action)
        
        # Deal with measuring/not measuring
        if action > self.CActionSize:
            obs = next_state
            reward -= self.cost
        else:
            obs = self.EmptyObservation
        
        return self.to_StepResult(action, obs, next_state, reward, done ), True
        # Reformat & return
        
    def to_StepResult(self, action, obs, nextState, reward,done):
        results = StepResult()
        results.action = BoxAction(action)
        results.observation = BoxObservation(obs)
        results.reward = reward
        results.is_terminal = done
        results.next_state = BoxState(nextState)
        return results
    
    
    # Model setters/getters and renamings:
    
    def update(self, step_results):
        '''Does nothing: we do not need to update according 
        to results, we only used the sampled results. '''
        pass
    
    def get_all_observations(self):
        obs = {}
        for i in range(self.StateSize):
            obs[i] = i
        obs[i+1] = self.EmptyObservation
        return obs
    
    def create_action_pool(self):
         return DiscreteActionPool(self)
     
    def get_all_actions(self):
        '''Return all possible actions in BoxAction-format'''
        all_actions = []
        for i in range(self.ActionSize):
            all_actions.append(BoxAction(i))
        return all_actions

    def get_legal_actions(self, state):
        return self.get_all_actions()
    
    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)
    
    def reset_for_simulation(self):
        pass

    def sample_an_init_state(self):
        return BoxState(self.init_state)

    def create_root_historical_data(self, solver):
        return PositionData(self, self.init_state, solver)

    def belief_update(self, old_belief, action, observation):
        pass

    def get_max_undiscounted_returns(self):
        return 1.0

    def reset_for_epoch(self):
        self.env.reset()
        self.real_state = self.init_state
        self.t = 0

    def render(self):
        pass
                
    def make_next_position(self, state, action):
        if type(action) is not int:
            action = action.bin_number
        if type(state) is not int:
            state = state.position
        state, reward, done = self.model_step(state, action)
        return BoxState(state, done, reward), True


