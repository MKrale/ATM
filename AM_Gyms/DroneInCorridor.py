from gym import Env
import numpy as np
import math as m

class DroneInCorridor(Env):
    """Custom environment describing a drone flying through a corridor, with disturbance
    in the system dynamics representing wind."""
    
    def __init__(self):
        
        # Positions
        self.Xmin, self.Xmax = 0, 30
        self.Ymin, self.Ymax = 0, 30
        self.Xnmbr = self.Xmax - self.Xmin + 1
        self.Ynmbr = self.Ymax - self.Ymin + 1
        
        # Speeds
        self.Vmin, self.Vmax = -5, 5
        self.Vnmbr = self.Vmax - self.Vmin + 1
        
        # Accelarations
        self.Amin, self.Amax = -2, 2
        self.Anmbr = self.Amax - self.Amin + 1
        
        # Walls
        self.WallXmin, self.WallXmax = 0,  20
        self.WallYmin, self.WallYmax = 10, 30
        
        # Goal area
        self.GoalXmin, self.GoalXmax = 20, 30
        self.GoalYmin, self.GoalYmax = 30, np.inf
        
        self.s_init = 0
        self.reset() #sets s and other variables.  
    
    # Functions for going to/from (1D) actions/states to (2/4D) variables
    def get_state(self):
        return self.vars_to_state(self.x, self.y, self.vx, self.vy)
    
    def gerenic_index_to_position(index, dimensions, mins):
        """Given an index and hyperrectangle dimensions, returns the corresponding position"""
        position = []
        for i in range(len(dimensions)):
            size = dimensions[i][1] - dimensions[i][0]
            pos = (index // size ** i) % size + dimensions[i][0]
            position.append(pos - mins[i])
        return tuple(position)
    def generic_position_to_index(position, dimensions, mins):
        """Given a position and hyperrectangle dimensions, returns the corresponding index"""
        index = 0
        for i in range(len(position)):
            size = dimensions[i][1] - dimensions[i][0]
            index += (position[i] + mins[i] - dimensions[i][0]) * (size ** i)
        return index
        
    def vars_to_state(self, x, y, vx, vy):
        """Given state variables, returns 1D state"""
        x, y, vx, vy = x - self.Xmin, y - self.Ymin, vx - self.Vmin, vy - self.Vmin
        if y<self.WallYmin:
              return vx + self.Vnmbr * (vy + self.Vnmbr * (x + self.Xnmbr * (y)))
        else:
            nmbr_previous_states = self.Vnmbr + self.Vnmbr * (self.Vnmbr + self.Vnmbr * (self.Xnmbr + self.Xnmbr * self.WallYmin) )
            this_i = vx + self.Vnmbr * (vy + self.Vnmbr * ( (y-self.WallYmin) + self.Ynmbr * (x-self.WallXmax)))
            return nmbr_previous_states + this_i
    
    def state_to_vars(self, state):
        """Given 1D state, returns state variables (x, y, vx, vy)"""
        # Written by chatgpt, not tested but I trust our AI overlords!
        nmbr_previous_states = self.Vnmbr + self.Vnmbr * (self.Vnmbr + self.Vnmbr * (self.Xnmbr + self.Xnmbr * self.WallYmin))
        if state < nmbr_previous_states:
            y =   state                                           // (self.Xnmbr * self.Vnmbr * self.Vnmbr)
            x =  (state % (self.Xnmbr * self.Vnmbr * self.Vnmbr)) // (self.Vnmbr * self.Vnmbr)
            vy = (state % (self.Vnmbr * self.Vnmbr))              //  self.Vnmbr
            vx = (state % (self.Vnmbr))                           
            return x + self.Xmin, y + self.Ymin, vx + self.Vmin, vy + self.Vmin
        else:
            state = state-nmbr_previous_states
            x =   state                                           // (self.Xnmbr * self.Vnmbr * self.Vnmbr)
            y =  (state % (self.Xnmbr * self.Vnmbr * self.Vnmbr)) // (self.Vnmbr * self.Vnmbr)
            vy = (state % (self.Vnmbr * self.Vnmbr))              //  self.Vnmbr
            vx = (state % (self.Vnmbr))  
            return x + self.Xmin + self.WallXmax, y + self.Ymin + self.WallYmin, vx + self.Vmin, vy + self.Vmin
    
    def vars_to_action(self, ax, ay):
        """Given action variables, returns 1D action"""
        ax, ay = ax + self.AXmin, ay + self.AYmin
        return ax + self.AXnmbr * ay
    
    def action_to_vars(self, a):
        """Given action, returns action variables (ax, ay)"""
        ax = a % self.AXnmbr
        ay = ( (a - ax) / self.AXnmbr ) % self.AYnmbr
        
        return (ax - self.AXmin, ay - self.AYmin)
    
    @staticmethod
    def Gaussian_disturb(a, amax):
        p = np.random.rand()
        if p < 0.68:
            pass
        elif p < 0.82:
            a += 1
        elif p < 0.96:
            a-= 1
        elif a < 0.98:
            a += 2
        else:
            a -= 2
        return max([a,amax])
        
    # Gym Functionality:
    
    def check_inside_field(self, x, y):
        return not( x > self.Xmax or x < self.Xmin or
                    y > self.Ymax or y < self.Ymin)
    
    def check_collision_walls(self, xys:set):
        for (x,y) in xys:
            self.check_collision_walls_el(x,y)
    def check_collision_walls_el(self, x, y):
        return (    y < self.WallYmin or y > self.WallYmax or
                    x < self.WallXmin or x > self.WallXmax )
        
    def in_goal(self, x, y):
        return (    x > self.GoalXmin and x < self.GoalXmax and
                    y > self.GoalYmin and y < self.GoalYmax)
    
    def step(self, a):
        
        # Read action & perform disturbation
        ax, ay = self.action_to_vars(a)
        ax, ay = self.Gaussian_disturb(ax, self.Amax), self.Gaussian_disturb(ay, self.Amax)
        
        # Calculate (avg and final) speeds
        vx_prev, vy_prev = self.vx, self.vy
        self.vx = min([ self.Vmin, max([ self.Vmax, self.vx + ax ]) ])
        self.vy = min([ self.Vmin, max([ self.Vmax, self.vy + ay ]) ])
        dx, dy = (vx_prev + self.vx) / 2, (vy_prev + self.vy) / 2
        
        # Find gridcells crossed
        slope = dy/dx
        crossed_cells = set([])
        y_last_step = self.y
        for xi in range(dx):
            for yi in range(m.ceil(slope)):
                crossed_cells.add((self.x+xi,m.floor(y_last_step+yi)))
            y_last_step += slope
        
        # Test if path was valid:
        self.x, self.y = self.x+dx, self.y+dy
        if not (self.check_inside_field(self.x, self.y) or
                self.check_collision_walls(crossed_cells)):
            return 0, 0, True, {}
        
        # Test if in goal area:
        if self.in_goal(self.x, self.y):
            return 0, 1, True, {}
        
        return self.get_state(), 0, False, {}
        
    def reset(self):
        self.s = self.s_init
        self.x, self.y, self.vx, self.vy = self.state_to_vars(self.s_init)
        
    def getname(self):
        return "Drone"
    
env = DroneInCorridor()