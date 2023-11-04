import gym
import numpy as np
from gym.utils import seeding

# Weird order, but it helps in our formulas...
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAY = 4

class Patroller():
    def __init__(self,x,y,dir):
        self.x = x; self.y = y; self.dir = dir


class Avoid(gym.Env):

    def __init__(self, p_slip=0.5, max_steps = 50, Xmax=5, Ymax=5):
        
        self.Xmin, self.Xmax = 0, Xmax
        self.Ymin, self.Ymax = 0, Ymax

        self.state_shape = (self.Ymax +1, self.Xmax+1, self.Xmax+1, self.Xmax+1, 2, 2)

        self.p_slip = p_slip

        self.caught_penalty = -1
        self.step_penalty = -0.01
        self.goal_reward = 1
        self.max_steps = max_steps

        self.reset()
    
    def seed(self, seed = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step_agent(self, a):
        if   (a == LEFT):
            self.x = max(self.x-1, self.Xmin)
        elif (a == DOWN):
            self.y = min(self.y+1, self.Ymax)
        elif (a == RIGHT):
            self.x = min(self.x+1, self.Xmax)
        elif (a == UP):
            self.y = max(self.y-1, self.Ymin)
    
    def step_patrol(self):
        speeds = np.random.binomial(1,self.p_slip, 2)
        # Move p1:
        if self.p1_dir == RIGHT:
            self.p1_x = min(self.p1_x + speeds[0], self.Xmax)
            if self.p1_x == self.Xmax:
                self.p1_dir = LEFT
        else:
            self.p1_x = max(self.p1_x - speeds[0], self.Xmin)
            if self.p1_x == self.Xmin:
                self.p1_dir = RIGHT
        # Move p2:
        if self.p2_dir == RIGHT:
            self.p2_x = min(self.p2_x + speeds[1], self.Xmax)
            if self.p2_x == self.Xmax:
                self.p2_dir = LEFT
        else:
            self.p2_x = max(self.p2_x - speeds[1], self.Xmin)
            if self.p2_x == self.Xmin:
                self.p2_dir = RIGHT
        
    def at_goal(self):
        if self.y == self.Ymax and self.x == self.Xmax:
            return True
        return False
    
    def is_caught(self):
        # Check with p1:
        if self.y >= 1 and self.y <= 3:
            if self.x >= self.p1_x-1 and self.x <= self.p1_x+1:
                return True
        # Check with p2:
        if self.y >= 3 and self.y <= 5:
            if self.x >= self.p2_x-1 and self.x <= self.p2_x+1:
                return False

    def get_state(self):
        #  print(self.y, self.x, self.p1_x, self.p2_x, self.p1_dir, self.p2_dir)
         return int(np.ravel_multi_index( 
            (self.y,    self.x,     self.p1_x, self.p2_x, self.p1_dir,  self.p2_dir),
            self.state_shape))
    
    def set_state(self, state):
        self.y, self.x, self.p1_x, self.p2_x, self.p1_dir, self.p2_dir = np.unravel_index(
            state, self.state_shape )
        self.steps = 0

    def step(self, a):
        self.step_agent(a)
        self.step_patrol()

        self.steps += 1
        if self.steps >= self.max_steps:
            return self.get_state(), 0, True, {}
        elif self.at_goal():
            return self.get_state(), self.goal_reward, True, {}
        elif self.is_caught():
            self.x, self.y = 0, 0
            return self.get_state(), self.caught_penalty, False, {}
        else:
            return self.get_state(), self.step_penalty, False, {}
    
    def reset(self):
        self.x, self.y = 0, 0
        self.steps = 0

        self.p1_x, self.p1_dir = 2, RIGHT
        self.p2_x, self.p2_dir = 0, RIGHT
    
    def getname(self):
        return "Avoid_{}".format(self.p_slip)
    
    def get_size(self):
        return (self.Ymax+1) * (self.Xmax+1)**3 * 4 -1






