import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class uMV_Env(gym.Env):
    """
    The Lucky-Unlucky Environment: see Krale et al (2024) for a full explanation.
    """
    def __init__(self, p=0.05, rbig = 1, rsmall = 0.0):

        self.p = p
        self.rbig = rbig
        self.rsmall = rsmall

        self.state = 0 
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(4)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        done = False
        reward = 0

        # Else if in state 0, we always go forward
        if self.state == 0:
            if np.random.rand() < self.p:   # s-
                self.state = 2
            else:
                self.state = 1
        # For safe actions, we get the same reward
        else:
            if action == 0:
                reward = self.rsmall
            elif action == 1 and self.state == 1:
                reward = self.rbig
            elif action == 1 and self.state == 2:
                reward = -self.rbig
            else: print("ERROR: impossible state-action pair reached!")
            self.state = 3
            done = True
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

    def getname(self):
        return "uMV_{}".format(float_to_str(self.p))
    
    def set_state(self, s):
        self.s = s
    
    
def float_to_str(float):
        if np.isclose(float, 0):
                return "0"
        elif float<0:
                return "-" + float_to_str(-float)
        elif float >= 1:
                prefix = int(np.floor(float))
                return str(prefix) + float_to_str(float - prefix)[1:]
        elif float < 1:
                return "0" + str(float)[2:]