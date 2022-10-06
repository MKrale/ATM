import numpy as np
import time
from pomdpy.solvers.pomcp import POMCP
from pomdpy.pomdp.history import Histories, HistoryEntry

from ACNO_generalised.ACNO_ENV import ACNO_ENV


class ACNO_Agent:
    
    def __init__(self, env=ACNO_ENV):
        self.model = env
        
        self.explore_episodes = 2000
        self.max_steps = 1000
        
        # Variables needed for POMCP
        self.histories = Histories()
        self.action_pool = self.model.create_action_pool()
        self.observation_pool = self.model.create_observation_pool(self)
        
        # The POMCP-solver
        self.solver = POMCP(self)
        self.solver_factory = POMCP
        
        
    
    def run(self, nmbr_episodes, get_full_results=True):
        
        # Declare variables
        rewards, steps, measurements = np.zeros(nmbr_episodes), np.zeros(nmbr_episodes), np.zeros(nmbr_episodes)
        
        # Run exploration phase
        exp_eps = self.explore_episodes #readibilty re-define
        rewards[:exp_eps], steps[:exp_eps] = self.model.sample_model(self.explore_episodes)
        measurements[:exp_eps] = steps[:exp_eps]
        print(self.model.T)
        print(self.model.R)
        
        # Run episodes
        for i in range(self.explore_episodes, nmbr_episodes):
            rewards[i], steps[i], measurements[i] =  self.run_episode(i)
        return rewards, steps, measurements
    
    
    def run_episode(self, epoch):
        
        # Remake solver & reset env
        self.histories = Histories() #maybe this reset helps?
        self.model.reset_for_epoch()
        solver = self.solver_factory(self)
        
        # Initialise episode variables
        state = solver.belief_tree_index.sample_particle()
        steps, measurements, totalreward, past_obs =0, 0, 0, state.position

        done = False
        while not done and steps<self.max_steps:
            start_time = time.time()
            # Get optimal action from solver
            action = solver.select_eps_greedy_action(0, start_time, greedy_select=False).bin_number
            
            # Perform action
            ac, ao = action % self.model.CActionSize, action // self.model.CActionSize,
            (reward, obs, done) = self.model.take_real_step(action, False)
            print(ac, obs, reward)
            #TODO: take_real_step does not return next state: should it?
            
            # Update solver & history
            stepResult = self.model.to_StepResult(action, obs, obs, reward, done)
            if not done:
                solver.update(stepResult)
            
            new_hist_entry = solver.history.add_entry()
            HistoryEntry.update_history_entry(new_hist_entry, reward, action, obs, obs)
            
            # Update logging variables
            
            steps += 1
            totalreward += reward
            past_obs = obs
            if action < self.model.CActionSize:
                measurements += 1
            state = obs
        print ("{} runs complete (current reward = {}, nmbr steps = {}, nmbr measurements = {})"
               .format( epoch,  totalreward, steps+1, measurements ))
        
        return totalreward, steps, measurements