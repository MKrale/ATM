import math
import random
import numpy as np
import pickle
from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool
from pomdpy.discrete_pomdp import DiscreteAction
from pomdpy.discrete_pomdp import DiscreteState
from pomdpy.discrete_pomdp import DiscreteObservation
from pomdpy.pomdp import HistoricalData
from pomdpy.pomdp import Model, StepResult
from pomdpy.util import console, config_parser
from AM_Env_wrapper import AM_ENV
import time

debug_mode = False

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

class ACNO_ENV():
    '''
    Generalised verion of the ACNO-environment from https://github.com/nam630/acno_mdp .
    Acts as wrapper around AM_ENV class for ACNO-POMCP agent.
    '''
    def __init__(self, env:AM_ENV, missing=True, cost=-0, is_mdp=0):
        
        # get variables from openAI env
        self.sim = env
        self.states_n, self.actions_n, self.cost, self.init_state = env.get_vars()
        self.cost = - self.cost
        self.actions_n = self.actions_n * 2
        self.c_actions = self.actions_n // 2

        self.obs_n = self.states_n
        self.missing = bool(is_mdp==0)
        if self.missing:
            self.na = self.states_n # TODO: check if this works!
        self.max_steps = 1000 
        self.t = 0
        self.seed = random.seed() 
        self.n_start_states = 1
        self.ucb_coefficient = 3.0
        self.min_particle_count = 80
        self.max_particle_count = 120
        self.max_depth = 5
        self.action_selection_timeout = 60
        self.particle_selection_timeout = 0.2
        self.n_sims = 100 # same as default params in pomcpy.py
        if is_mdp == 0:
            self.solver = 'POMCP'
        else:
            self.solver = 'MCP'
        #print('Solving with ', self.solver)
        self.preferred_actions = True # not used
        self.test = 10
        self.epsilon_start = 0.9
        self.epsilon_minimum = 0.1
        self.epsilon_decay = 0.9
        self.discount = 0.7
        self.n_epochs = 500
        self.save = False
        self.timeout = 7_200_000
        
        # starts with an empty model 
        self.t_estimates = np.zeros((self.states_n, self.actions_n, self.states_n))
        self.r_estimates = np.zeros((self.states_n, self.actions_n))
        
        # use a prior of 1 for all (s, a, s') 
        self.n_counts = np.ones((self.states_n, self.actions_n)) * self.states_n
        self.r_counts = np.zeros((self.states_n, self.actions_n))
        self.t_counts = np.ones((self.states_n, self.actions_n, self.states_n))

    def update(self, step_result):
        pass

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def get_all_actions(self):
        all_actions = []
        for i in range(self.actions_n):
            all_actions.append(BoxAction(i))
        return all_actions

    def get_legal_actions(self, state):
        all_actions = []
        for i in range(self.actions_n):
            all_actions.append(i)
        return all_actions

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def reset(self):
        self.t = 0

    
    def generate_particles(self, prev_belief, action, obs, n_particles, prev_particles, mdp):
        '''
        Samples a new set of belief particles, according to previous belief and action.
        Sampling implemented using generate_step function.
        '''
        #Initialisation
        particles = []
        action_node = prev_belief.action_map.get_action_node(action)
        if action_node is None:
            return particles
        else:
            obs_map = action_node.observation_map
        child_node = obs_map.get_belief(obs)
        start = time.time()
        
        ### MODEL-BASED SAMPLING ###
        
        while particles.__len__() < n_particles:
            # Chose random current state
            state = random.choice(prev_particles)
            # Sample a sucessor
            if mdp:
                result, is_legal = self.generate_step(state, action, is_mdp=True)
            else:
                result, is_legal = self.generate_step(state, action)
            # if null (i.e., 720) obs, any state particle CAN be added
            
            # TODO: see how this works (checking for null-observations)
            if result.observation.position == self.states_n or result.observation == obs: 
                assert(result.next_state.position < self.states_n)
                particles.append(result.next_state)
                if particles.__len__() % 500 == 0: # logging for debugging
                    pass
                    #print(particles.__len__(), time.time() - start)
            
            #  Break if particle selection takes too long
            if time.time() - start > self.particle_selection_timeout:
                if obs.position != self.states_n:
                    pass
                    #print(state.position, action.bin_number, obs.position)
                    #print('prev pos:', state.position, particles.__len__(), 'prob:', self.t_estimates[state.position, action.bin_number % self.c_actions, obs.position])
                    #print('real pos{}, real transition:'.format(self.real_state), self.t_estimates[self.real_state, action.bin_number, obs.position])
                if particles.__len__() > 3: # 3 too long?
                    #print('REJECTION timeout:', time.time() - start)
                    break
        
        ### EXPANDING SAMPLE ###
        
        # If timed for real sampling, expand our current sample instead
        while particles.__len__() < n_particles:
            state = random.choice(particles)
            new_state = state.copy()
            particles.append(new_state)

        return particles

    '''
    Can use a plug-in empirical simulator
    Inp: action (dtype int), state (dtype int)
    1) # of samples
    2) exact noise level
    '''
    def empirical_simulate(self, state, action):
        """Estimates the next state and reward, using exisiting model."""
        action = action % self.c_actions
        p = self.t_estimates[state,action,:]
        next_state = np.random.choice(self.states_n, 1, p=p)[0] # sample according to probs
        rew = self.r_estimates[state, action]
        
        # TODO: figure out if terminal-check is required!
        terminal = False
        return BoxState(int(next_state), is_terminal=terminal, r=rew), True

    def take_real_step(self, action):
        '''Takes specified action in environment, returns next state and reward.'''
        if type(action) is not int:
            action = action.bin_number
        (ac, ao) = action % self.c_actions, action // self.c_actions
        reward, done = self.sim.step(ac)
        (obs, measureCost) = self.sim.measure()
        self.real_state = obs
        return BoxState(obs, is_terminal=done, r=reward), True

    def make_next_position(self, state, action):
        '''Returns next state as sampled by current model'''
        if type(action) is not int:
            action = action.bin_number
        if type(state) is not int:
            state = state.position
        return self.empirical_simulate(state, action)

    def get_all_observations(self):
        obs = {}
        for i in range(self.obs_n):
            obs[i] = i

    def make_observation(self, action, next_state, always_obs=False):
        '''Returns observation if measuring, NA otherwise'''
        if action.bin_number < self.actions_n // 2 or always_obs:
            obs = next_state.position # not observe
        else:
            obs = self.na
        return BoxObservation(obs)
        
    def make_reward(self, state, action, next_state, is_legal, always_obs=False):
        '''Returns costed reward'''
        rew = 0
        if not is_legal:
            return rew
        if action.bin_number < self.actions_n // 2 or always_obs:
            rew = rew + self.cost
        rew += next_state.final_rew
        return rew

    def generate_step(self, state, action, _true=False, is_mdp=False):
        if type(action) is int:
            action = BoxAction(action)
        result = StepResult()
        
        # Based on the simulator, next_state is true next state or imagined by the simulator
        if _true:
            #print("taking true step")
            result.next_state, is_legal = self.take_real_step(action)
            #print("next true state:", result.next_state.position)
        # Use true simulator for taking actions (only for eval)
        else:
            result.next_state, is_legal = self.make_next_position(state, action)
        
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state, always_obs=is_mdp)
        result.reward = self.make_reward(state, action, result.next_state, is_legal, always_obs=is_mdp)
        result.is_terminal = result.next_state.terminal
        return result, is_legal


    def reset_for_simulation(self):
        pass

    def sample_an_init_state(self):
        return BoxState(self.init_state)

    def create_root_historical_data(self, solver):
        return PositionData(self, self.init_state, solver)

    def belief_update(self, old_belief, action, observation):
        pass

    def get_max_undiscounted_return(self):
        return 1.0

    def reset_for_epoch(self):
        self.sim.reset()
        self.real_state = self.init_state
        self.t = 0

    def render(self):
        pass
