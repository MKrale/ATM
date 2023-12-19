import gym
import numpy as np
from gym.utils import seeding

# Weird order, but it helps in our formulas...
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
INTERACT = 4


class CoalOrGold(gym.Env):
    """A custom Active Measure environment. 
    An agent needs to collect coal from a number of spots. H
    owever, when collecting it has a small chance to find gold instead: if he does, he can hand it in immidiatly to get a large reward.
    The agent can inspect it's current inventory by measuring.
    """

    def __init__(self, goldchance=0.1, max_steps = 50, Xmax=4, Ymax=4):
        
        self.Xmin, self.Xmax = 0, Xmax
        self.Ymin, self.Ymax = 0, Ymax

        self.state_shape = (self.Xmax+1, self.Ymax+1, 2, 2, 2, 2, 2)

        self.interactPenalty = -1
        self.step_penalty = -0.01
        self.CoalReward = 0.5
        self.GoldReward = 1

        self.goldChance = goldchance
        self.max_steps = max_steps

        self.reset()
        
    
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def at_goal(self):
        if self.y == np.floor(self.Ymax / 2) and self.x == 0:
            return True
        return False
    
    def get_state(self):
        state_tuple = (self.x, self.y, self.hasGold)
        for i in range(len(self.deposits_mined)):
            state_tuple = state_tuple + (self.deposits_mined[i],)
        return int(np.ravel_multi_index(state_tuple, self.state_shape))
    
    def set_state(self, state):
        self.x, self.y, self.hasGold, d1, d2, d3, d4 = np.unravel_index(state, self.state_shape )
        self.deposits_mined = [d1, d2, d3, d4]
        self.steps = 0

    def step_agent(self, a):
        if   (a == LEFT):
            self.x = max(self.x-1, self.Xmin)
        elif (a == DOWN):
            self.y = min(self.y+1, self.Ymax)
        elif (a == RIGHT):
            self.x = min(self.x+1, self.Xmax)
        elif (a == UP):
            self.y = max(self.y-1, self.Ymin)
    
    def mine(self):
        for (i, pos) in enumerate(self.deposits):
            if (not self.deposits_mined[i]) and (self.x == pos[0] and self.y == pos[1]):
                self.deposits_mined[i] = True
                self.hasGold = self.hasGold or bool(np.random.binomial(1,self.goldChance))
                return True
        return False
                
    def step(self, a):

        if self.steps >= self.max_steps:
            return self.get_state(), self.step_penalty, True, {}

        elif a == INTERACT:
            
            if self.at_goal() and np.all(self.deposits_mined):
                return self.get_state(), self.CoalReward, True, {}
            elif self.mine():
                return self.get_state(), self.step_penalty, False, {}
            elif self.hasGold:
                return self.get_state(), self.GoldReward, True, {}
            else:
                return self.get_state(), self.interactPenalty, False, {}                
        else: 
            self.step_agent(a)
            return self.get_state(), self.step_penalty, False, {}

        


    
    def reset(self):
        self.x, self.y = 0, np.floor(self.Ymax / 2)
        self.steps = 0

        self.deposits_mined = [False, False, False, False]
        self.deposits = [[0,0], [0,self.Ymax], [self.Xmax, 0], [self.Xmax, self.Ymax]]
        
        self.hasGold = False

    def getname(self):
        return "CoalOrGold_p{}".format(self.goldChance)
    
    def get_size(self):
        return (self.Ymax+1) * (self.Xmax+1) * 2**5 -1




