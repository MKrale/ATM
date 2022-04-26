# Wrapper to turn Open AI Gym-environments into active measure environments

class AM_ENV():

    def __init__(self, env, StateSize, ActionSize, MeasureCost, s_init):
        self.env = env
        self.StateSize = StateSize
        self.ActionSize = ActionSize    # Is there any way to get these two from the env. itself?
        self.MeasureCost = MeasureCost
        self.s_init = s_init

    def get_vars(self):
        return (self.StateSize, self.ActionSize, self.MeasureCost, self.s_init)
    
    def step_and_measure(self,action):
        (obs, reward, done, info) = self.env.step(action)
        return (obs, reward, done)
    
    def step_no_measure(self,action):
        (obs, reward, done, info) = self.env.step(action)
        return (reward, done)
 
    def reset(self):
        self.env.reset()
