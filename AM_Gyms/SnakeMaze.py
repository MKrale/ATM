import gym
import numpy as np
from gym.utils import seeding

# Weird order, but it helps in our formulas...
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3


class SnakeMaze(gym.Env):

    def __init__(self, breakChance=0.5, max_steps = 500, size=10):
        
        self.Xmin, self.Xmax = 0, size-1
        self.Ymin, self.Ymax = 0, size-1

        self.state_shape = (size, size)

        self.step_penalty = -0.005
        self.GoalReward = 1

        self.breakChance = breakChance
        self.max_steps = max_steps

        self.reset()
        
    
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def at_goal(self):
        if self.x == 0 and self.y == self.Ymax:
            return True
        return False
    
    def get_state(self):
        state_tuple = (self.y, self.x)
        return int(np.ravel_multi_index(state_tuple, self.state_shape))
    
    def set_state(self, state):
        self.y, self.x = np.unravel_index(state, self.state_shape)
        self.steps = 0

    def step_agent(self, a):
        if   (a == LEFT):
            self.x = max(self.x-1, self.Xmin)
        elif (a == RIGHT):
            self.x = min(self.x+1, self.Xmax)

        elif (a == DOWN and (self.y % 2 == 0) and (self.x == self.Xmax) ):
            self.y = min(self.y+1, self.Ymax)
        elif (a == DOWN and (self.y % 2 == 1) and (self.x == self.Xmin) ):
            self.y = min(self.y+1, self.Ymax)
                
    def step(self, a):

        if self.steps >= self.max_steps:
            return self.get_state(), self.step_penalty, True, {}
        
        elif self.at_goal():
            # self.x, self.y = self.Xmax-1, self.Ymax-1
            return self.get_state(), self.GoalReward - self.step_penalty, True, {}
             
        self.step_agent(a)

        return self.get_state(), self.step_penalty, False, {}

    
    def reset(self):
        self.x, self.y = 0, 0
        self.steps = 0

    def getname(self):
        return "SnakeMaze_p{}".format(self.breakChance)
    
    def get_size(self):
        return (self.Ymax+1) * (self.Xmax+1)