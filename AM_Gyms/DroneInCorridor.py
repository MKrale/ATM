from gym import Env
import numpy as np

class DroneInCorridor(Env):
    """Custom environment describing a drone flying through a corridor, with disturbance
    in the system dynamics representing wind."""
    
    def __init__(self):
        
        self.Xmin, self.Xmax = 0, 30
        self.Xnmbr = self.Xmax - self.Xmin
        self.Ymin, self.Ymax = 0, 30
        self.Ynmbr = self.Ymax - self.Ymin
        self.VXmin, self.VXmax = -5, 5
        self.VXnmbr = self.VXmax - self.VXmin
        self.VYmin, self.VYmax = -5, 5
        self.VYnmbr = self.VYmax - self.VYmin
        self.AXmin, self.AXmax = -2, 2
        self.AXnmbr = self.AXmax - self.AXmin
        self.AYmin, self.AYmax = -2, 2
        self.AYnmbr = self.AYmax - self.AYmin
        
        self.WallXmin, self.WallXmax = 0, 10
        self.WallYmin, self.WallYmax = 0, 10
        
        self.s_init = 0
        self.reset() #sets s and other variables.
        
        self.set_wind_generic("still")
    
    def set_wind_generic(self, name):
        """Choose wind disturbance from presets"""
        
        self.DisturbanceArrayX = np.zeros( (self.Xnmbr, self.Ynmbr, self.VXnmbr))
        self.DisturbanceArrayY = np.zeros( (self.Xnmbr, self.Ynmbr, self.VYnmbr))
        
        match name:
            
            case "still":
                # No wind, so chance w=0 is 1
                self.DisturbanceArrayX[:,:,-self.VXmin] = 1
                self.DisturbanceArrayY[:,:,-self.VYmin] = 1
                self.disturbance_name = name
                
            case "uniform":
                # Equal probability of each wind disturbance
                self.DisturbanceArrayX = np.zeros( (self.Xnmbr, self.Ynmbr, self.VXnmbr)) + 1/self.VXnmbr
                self.DisturbanceArrayY = np.zeros( (self.Xnmbr, self.Ynmbr, self.VYnmbr)) + 1/self.VYnmbr
                self.disturbance_name = name
                
            case "Gaussian_X":
                # Probability gaussian with deviation X
                print("UNIMPLEMENTED: disturbance kept at previous setting ({})".format(self.disturbance_name))
                
            case "Naive_worst_case":
                # Sets wind maximally towards closest wall
                print("UNIMPLEMENTED")
        
    
    # Functions for going to/from (1D) actions/states to (2/4D) variables    
    def vars_to_state(self, x, y, vx, vy):
        """Given state variables, returns 1D state"""
        x, y, vx, vy = x+self.Xmin, y + self.Ymin, vx + self.VXmin, vy + self.VYmin
        return x + self.Xnmbr * ( y + self.Ynmbr * ( vx + self.VXnmbr * (vy) ) )
    
    def state_to_vars(self, state):
        """Given 1D state, returns state variables (x, y, vx, vy)"""
        x   = state % self.Xnmbr
        y   = ( (state - x) / self.Xnmbr) % self.Ynmbr
        vx  = ( (state - ( x + self.Xnmbr * (y) ) ) / (self.Xnmbr * self.Ynmbr) ) % self.VXnmbr
        vy  = ( (state - ( x + self.Xnmbr * (y + self.Ynmbr (vx)))) 
               / (self.Xnmbr * self.Ynmbr * self.VXnmbr) ) % self.VYnmbr
        
        return (x-self.Xmin, y-self.Ymin, vx-self.VXmin, vy-self.VYmin)
    
    def vars_to_action(self, ax, ay):
        """Given action variables, returns 1D action"""
        ax, ay = ax + self.AXmin, ay + self.AYmin
        return ax + self.AXnmbr * ay
    
    def action_to_vars(self, a):
        """Given action, returns action variables (ax, ay)"""
        ax = a % self.AXnmbr
        ay = ( (a - ax) / self.AXnmbr ) % self.AYnmbr
        
        return (ax - self.AXmin, ay - self.AYmin)
    
    
    # Functions checking if the drone is in a certain area:
    def out_of_bounds(self, x, y):
        return (  
            # Outside of area  
            x < self.Xmin or x > self.Xmax or
            y < self.Ymin or y > self.Ymax or
            # Collided with cut-out square
            ( y > self.WallYmin and y < self.WallYmax and
              x > self.WallXmin and x < self.WallXmax ) )
    
    def has_collided(self, x, y):
        """UNIMPLEMENTED!"""
        return False 
    
    def has_reached_goal(self, x, y):
        return x > self.Xmax
    
    
    # Gym Functionality:
    
    def step(self, a):
        
        ax, ay = self.action_to_vars(a)
        
        xindex, yindex = self.x - self.Xnmbr, self.y - self.Ynmbr
        
        # System dynamics with bounds
        self.vx +=  ax + self.DisturbanceArrayX[xindex, yindex]
        self.vx  = np.min(self.VXmax, np.max(self.VXmin, self.vx))
        self.vy +=  ay + self.DisturbanceArrayY[xindex, yindex]
        self.vy  = np.min(self.VYmax, np.max(self.VYmin, self.vy))
        
        self.x += self.vx + np.floor (ax / 2)
        self.y += self.vy + np.floor (ay / 2)
        self.s = self.vars_to_state(self.x, self.y, self.vx, self.vy)
        
        # Check if out-of-bounds:
        done = False
        reward = 0
        if self.has_reached_goal(self.x,self.y):
            done = True
            reward = 1
            self.s = 0
        
        elif self.out_of_bounds(self.x, self.y) or self.has_collided(self.x, self.y, self.vx, self.vy):
            done = True
            self.s = 0
        
        return (self.s, reward, done, {} )
        
    def reset(self):
        self.s = self.s_init
        self.x, self.y, self.vx, self.vy = self.state_to_vars(self.s_init)
        
    def getname(self):
        return "Drone_{}".format(self.disturbance_name)