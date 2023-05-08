import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from AM_Gyms.AM_Env_wrapper import AM_ENV


class GenericAMGym(AM_ENV):
    
    def __init__(self, P:dict, R:dict, StateSize:int, ActionSize:int, MeasureCost:float, s_init:int, has_terminal_state:bool=True, max_steps:int = 10_000):
        
        self.P      = P
        self.R      = R
        # print(self.R)
        self.MeasureCost = MeasureCost
        self.s_init = s_init
        self.state  = s_init
        self.max_steps = max_steps
        
        self.StateSize = StateSize+1 #including donestate
        self.ActionSize = ActionSize
        
        self.action_space = spaces.Discrete(self.ActionSize)
        self.observation_space = spaces.Discrete(self.StateSize)
        
        
        # We assume the last state is the terminal state
        self.has_terminal_state = has_terminal_state
        self.steps_taken = 0
        
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    def step(self, action):
        assert self.action_space.contains(action)
        
        # Make transition
        prev_state = self.state
        p_dict = self.P[self.state][action]
        states, probs = list(p_dict.keys()), list(p_dict.values())
        self.state = np.random.choice(a=states, p=probs)
        
        # Get reward
        reward = 0
        if self.state in self.R[prev_state][action]:
            reward = self.R[prev_state][action][self.state]
        
        # Check if done
        self.steps_taken += 1
        #print(self.state, self.StateSize, self.has_terminal_state)
        done = (self.has_terminal_state and self.state == self.StateSize-1
                or self.steps_taken > self.max_steps)
        
        return reward, done
    
    def measure(self):
        return self.state, self.MeasureCost
        
    def reset(self):
        self.state = 0
        self.steps_taken = 0
        return self.state