import gym
import numpy as np
from gym.utils import seeding

# Weird order, but it helps in our formulas...
LEFT = 0
RIGHT = 1
RISKYLEFT = 2
RISKYRIGHT = 3
DOWN = 4

# cumulative chances of going n steps
NORMAL_CHANCES = [0, 0.6, 1, 1]
RISKY_CHANCES  = [0.5, 0.5, 0.5, 1]

class SnakeMaze(gym.Env):
    """A Custom AM environment.
    An agent needs to traverse a snaking maze.
    For each cardinal direction, it has both a risky and safe action available, both with different stochastic outcomes.
    """

    def __init__(self, max_steps = 500, size=10):
        
        self.Xmin, self.Xmax = 0, size-1
        self.Ymin, self.Ymax = 0, size-1

        self.state_shape = (size, size)

        self.step_penalty = -0.005
        self.GoalReward = 1

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

        if a == DOWN:
            if ((self.y % 2 == 0) and (self.x == self.Xmax) ):
                self.y = min(self.y+1, self.Ymax)
            if ((self.y % 2 == 1) and (self.x == self.Xmin) ):
                self.y = min(self.y+1, self.Ymax)
            return
        else:
            if a == RISKYLEFT or a == RISKYRIGHT:
                chances = RISKY_CHANCES
            else:
                chances = NORMAL_CHANCES
            
            if a == LEFT or a == RISKYLEFT:
                sign = -1
            else:
                sign = 1

            roll = np.random.random()
            for (i, p) in enumerate(chances):
                if roll < p:
                    # print(a, sign*i, self.x, min(self.Xmax, max(self.Xmin, self.x + sign * i)))
                    self.x = min(self.Xmax, max(self.Xmin, self.x + sign * i))
                    return



        # if   (a == LEFT):
        #     self.x = max(self.x-1, self.Xmin)
        # elif (a == RIGHT):
        #     self.x = min(self.x+1, self.Xmax)

        # elif (a == DOWN and (self.y % 2 == 0) and (self.x == self.Xmax) ):
        #     self.y = min(self.y+1, self.Ymax)
        # elif (a == DOWN and (self.y % 2 == 1) and (self.x == self.Xmin) ):
        #     self.y = min(self.y+1, self.Ymax)
                
    def step(self, a):

        if self.steps >= self.max_steps:
            return self.get_state(), self.step_penalty, True, {}
        
        elif self.at_goal():
            return self.get_state(), self.GoalReward + self.step_penalty, True, {}
        
        # if np.random.random() > self.slipChance:
        #     self.step_agent(a)
        self.step_agent(a)

        return self.get_state(), self.step_penalty, False, {}

    
    def reset(self):
        self.x, self.y = 0, 0
        self.steps = 0

    def getname(self):
        return "SnakeMazeV2"
    
    def get_size(self):
        return (self.Ymax+1) * (self.Xmax+1)