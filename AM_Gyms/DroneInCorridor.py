from gym import Env
import numpy as np
import math as m

class DroneInCorridor(Env):
    """Custom environment describing a drone flying through a corridor, with disturbance
    in the system dynamics representing wind."""
    
    def __init__(self,):
        
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
        self.WallXmin, self.WallXmax = 11, np.infty
        self.WallYmin, self.WallYmax = 11, np.infty
        
        # Goal area
        self.GoalXmin, self.GoalXmax = 0, 10
        self.GoalYmin, self.GoalYmax = 27, np.inf
        
        x_init, y_init, vx_init, vy_init = 30,5,0,0
        self.s_init = self.vars_to_state(x_init, y_init, vx_init, vy_init)
        self.reset() #sets s and other variables.  
    
    # Functions for going to/from (1D) actions/states to (2/4D) variables
    def get_state(self):
        return self.vars_to_state(self.x, self.y, self.vx, self.vy)
        
    def vars_to_state(self, x, y, vx, vy) -> int:
        """Given state variables, returns 1D state"""
        x, y, vx, vy = x - self.Xmin, y - self.Ymin, vx - self.Vmin, vy - self.Vmin
        if y<self.WallYmin:
              return int(vx + self.Vnmbr * (vy + self.Vnmbr * (x + self.Xnmbr * (y))))
        else:
            states_first_rectangle = 1 + (self.Vnmbr-1) + self.Vnmbr * ((self.Vnmbr-1) + self.Vnmbr * ((self.Xnmbr-1) + self.Xnmbr * (self.WallYmin-1)) )
            y -= (self.WallYmin)
            this_i = int(vx + self.Vnmbr * (vy + self.Vnmbr * (x + (self.WallXmin) * (y))))
            return int(states_first_rectangle + this_i)
    
    def state_to_vars(self, state):
        """Given 1D state, returns state variables (x, y, vx, vy)"""
        # Written by chatgpt, not tested but I trust our AI overlords!
        states_first_rectangle = (self.Vnmbr-1) + self.Vnmbr * ((self.Vnmbr-1) + self.Vnmbr * ((self.Xnmbr-1) + self.Xnmbr * (self.WallYmin-1)) )
        if state <= states_first_rectangle:
            y =   state                                                 // (self.Xnmbr * self.Vnmbr * self.Vnmbr)
            x =  (state % (self.Xnmbr * self.Vnmbr * self.Vnmbr))       // (self.Vnmbr * self.Vnmbr)
            vy = (state % (self.Vnmbr * self.Vnmbr))                    //  self.Vnmbr
            vx = (state % (self.Vnmbr))
        
        else:
            state -= 1+states_first_rectangle
            y =   state                                                 // (self.WallXmin * self.Vnmbr * self.Vnmbr)
            x =  (state % (self.WallXmin * self.Vnmbr * self.Vnmbr))    // (self.Vnmbr * self.Vnmbr)
            vy = (state % (self.Vnmbr * self.Vnmbr))                    //  self.Vnmbr
            vx = (state % (self.Vnmbr))
            y += (self.WallYmin)                           
        
        return x + self.Xmin, y + self.Ymin, vx + self.Vmin, vy + self.Vmin
    
    def vars_to_action(self, ax, ay):
        """Given action variables, returns 1D action"""
        ax, ay = ax - self.Amin, ay - self.Amin
        return ax + self.Anmbr * ay
    
    def action_to_vars(self, a):
        """Given action, returns action variables (ax, ay)"""
        ay = a // self.Anmbr
        ax = a % self.Anmbr
        
        return (ax + self.Amin, ay + self.Amin)
    
    @staticmethod
    def Gaussian_disturb(a, amax):
        p = np.random.rand()
        if p < 0.68:
            pass
        elif p < 0.82:
            a += 1
        elif p < 0.96:
            a-= 1
        # elif a < 0.98:
        #     a += 2
        # else:
        #     a -= 2
        if abs(a)> amax:
            a = amax * np.sign(a)
        return a
        
    # Gym Functionality:
    
    def in_field(self, x, y):
        return (    x <= self.Xmax and x >= self.Xmin and
                    y <= self.Ymax and y >= self.Ymin)
    
    def in_wall(self, xys:set):
        for (x,y) in xys:
            if self.in_wall_el(x,y):
                return True
        return False
            
    def in_wall_el(self, x, y):
        return (    (y >= self.WallYmin and y <= self.WallYmax)
                and (x >= self.WallXmin and x <= self.WallXmax) )
        
    def in_goal(self, x, y):
        return (    x > self.GoalXmin and x < self.GoalXmax and
                    y > self.GoalYmin and y < self.GoalYmax)
    
    def step(self, a):
        
        # Read action & perform disturbation
        ax, ay = self.action_to_vars(a)
        ax, ay = self.Gaussian_disturb(ax, self.Amax), self.Gaussian_disturb(ay, self.Amax)
        
        # Calculate (avg and final) speeds
        vx_prev, vy_prev = self.vx, self.vy
        self.vx = max([ self.Vmin, min([ self.Vmax, self.vx + ax ]) ])
        self.vy = max([ self.Vmin, min([ self.Vmax, self.vy + ay ]) ])
        dx, dy = round((vx_prev + self.vx) / 2), round((vy_prev + self.vy) / 2)

        # Check if endpoint within env
        if (not self.in_field(self.x+dx, self.y+dy)):
            return 0, 0, True, {}
        
        # Find gridcells crossed
        # Note: we just assume any obstacle in the box from start to end would block us.
        # To make this more realistic, we could maybe use the slope to find what gridcels are crossed
        # if we go in a straight path and use that, but I had problems implementing that correctly...
        
        if dx >= 0:
            xs = np.arange(start=self.x, step=1, stop=self.x+dx+1)
        else: 
            xs = np.arange(start=self.x+dx, step=1, stop=self.x+1)
            
        if dy >= 0:
            ys = np.arange(start=self.y, step=1, stop=self.y+dy+1)
        else: 
            ys = np.arange(start=self.y+dy, step=1, stop=self.y+1)
        
        crossed_cells = []
        for xi in xs:
            for yi in ys:
                crossed_cells.append((xi, yi))
        #print(self.x, self.y, ax, ay, self.vx, self.vy, dx, dy, crossed_cells)
        # Test if path was valid:
        if  ( self.in_wall(crossed_cells)):
            return 0, 0, True, {}
        
        # Test if in goal area:
        elif self.in_goal(self.x+dx, self.y+dy):
            return 0, 1, True, {}
        
        else:
            self.x, self.y = self.x+dx, self.y+dy
            return self.get_state(), 0, False, {}
        
    def reset(self):
        self.s = self.s_init
        self.x, self.y, self.vx, self.vy = self.state_to_vars(self.s_init)
        
    def getname(self):
        return "Drone"
    
    def set_state(self, s):
        self.s = s
        self.x, self.y, self.vx, self.vy = self.state_to_vars(s)
        
    def get_state_vars(self):
        return self.x, self.y, self.vx, self.vy
    
    def get_size(self):
        """Returns StateSize, ActionSize, s_init (for Runfile)"""
        return (self.vars_to_state(self.Xmax, self.Ymax, self.Vmax, self.Vmax),
                self.Anmbr*self.Anmbr,
                self.s_init)
        
env = DroneInCorridor()
print(env.get_size())
print(env.vars_to_state(10,30,5,5))
print(env.state_to_vars(70299))
print(env.state_to_vars(70300))
print(env.state_to_vars(70400))
for x in range(10):
    for y in range(30):
        for vx_ in range(11):
            for vy_ in range(11):
                vx=vx_-5; vy=vy_-5
            
                i = env.vars_to_state(x,y,vx,vy)
                x2,y2,vx2,vy2 = env.state_to_vars(i)
                i2 = env.vars_to_state(x2,y2,vx2,vy2)
                if i != i2 or (x!=x2 or y!=y2 or vx!=vx2 or vy!=vy2):
                    print(i, i2,(x,y,vx,vy), (x2,y2,vx2,vy2))

# print(env.vars_to_state(14,30,5,5))
# print(env.state_to_vars(8000))

# env.set_state(0)
# print(env.get_vars())
# print(env.state_to_vars(0))
# print(env.action_to_vars(0))
# print(env.step(0))


# print(env.state_to_vars(17))
# print(env.action_to_vars(1))
# for i in range(10):
#     print(env.step(1))
#     env.set_state(0)

