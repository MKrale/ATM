"""File containing classes for explicitely expressing models, i.e. with a full P, Q & R table"""
from abc import abstractmethod
import numpy as np
from gym import Env, spaces, utils
from AM_Gyms.AM_Env_wrapper import AM_ENV
from AM_Gyms.ModelLearner_V2 import ModelLearner
from AM_Gyms.ModelLearner_Robust import ModelLearner_Robust
import os
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    
def jsonKeys2int(x):
    if isinstance(x, dict):
        newdict = {}
        for (k,v) in x.items():
            if k.isdigit():
                newdict[int(k)] = v
            else:
                newdict[k] = v
        return newdict
    return x

class Environment_Explicit_Interface():
    # Interface class for our explicit environments, contains for importing & exporting
    
    # Basic variables
    StateSize:int
    ActionSize:int
    MeasureCost:int
    s_init:int
    isLearned = False
    
    def get_vars(self):
        """Returns (statesize, actionsize, cost, s_init)"""
        return self.StateSize, self.ActionSize, self.MeasureCost, self.s_init
    
    @abstractmethod 
    def env_to_dict(self):
        pass
    
    @abstractmethod
    def env_from_dict(self):
        pass
    
    def export_model(self, fileName, folder = None):
        """Exports model to json file"""
        if folder is None:
            folder = os.getcwd()
        fullPath = os.path.join(folder,fileName)
        with open(fullPath, 'w') as outfile:
            # print("\n\n",self.env_to_dict(),"\n\n")
            json.dump(self.env_to_dict(), outfile, cls=NumpyEncoder)

    def import_model(self, fileName, folder=None):
        """Imports model from json file"""
        if folder is None:
            folder = os.getcwd()
        fullPath = os.path.join(folder,fileName)
        with open(fullPath, 'r') as outfile:
            model = json.load(outfile, object_hook = jsonKeys2int)
        self.env_from_dict(model)
        self.isLearned = True
        
    def env_to_dict(self):
        return{
                "StateSize":    self.StateSize,
                "ActionSize":   self.ActionSize,
                "MeasureCost":  self.MeasureCost,
                "s_init":       self.s_init
                }
        
    def env_from_dict(self, dict):
        self.StateSize, self.ActionSize = dict["StateSize"], dict["ActionSize"]
        self.MeasureCost, self.s_init = dict["MeasureCost"], dict["s_init"]
        

class AM_Environment_Explicit(Environment_Explicit_Interface):
    """Class to explicitely express AM environments, i.e. with matrixes for P, R and (optionally) Q."""
    
    P:dict
    R:dict
    Q:np.ndarray
    
    StateSize:int
    ActionSize:int
    MeasureCost:int
    s_init:int
    
    isLearned = False
    
    def learn_model(self, env:Env):
        """Learns explicit model from Gym class (unimplemented!)"""
        print("to be implemented!")
        
    def learn_model_AMEnv(self, env:AM_ENV, N = 100, df = 0.8):
        """Learns explicit model from AM_ENV class"""
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()
        self.StateSize += 1
        learner                 = ModelLearner(env, df = df)
        try:
            learner.run_setStates(N)
        except AttributeError:
            learner.run_visits(N)
        print("done!")

        # TODO: add catch-except stuff
        self.P, self.R, self.Q  = learner.get_model()
        self.isLearned          = True
        
    def env_to_dict(self):
        """Returns dictiorary with all environment variables"""
        dict = {    "P":            self.P,
                    "R":            self.R,
                    "Q":            self.Q}
        dict.update(super().env_to_dict())
        return dict
        
    def env_from_dict(self, dict):
        """Changes class variables to those specified in dict"""
        super().env_from_dict(dict)
        self.P, self.R, self.Q = dict["P"], dict["R"], np.array(dict["Q"])

    def get_tables(self):
        """Returns (P, R, Q)"""
        return self.P, self.R, self.Q

    def get_vars(self):
        "Returns StateSize, ActionSize, MeasureCost, s_init "
        return self.StateSize, self.ActionSize, self.MeasureCost, self.s_init
        
class RAM_Environment_Explicit(Environment_Explicit_Interface):
    """Class to explicitely express uncertain AM environments, i.e. with matrixes for uP, R and Q. 
    Additionally contains an explicit copy of an \'average\' AM environment to be used by some functions."""
    
    # Uncertain dynamics
    Pmin:dict
    Pmax:dict
    R:dict
    
    # Average-case dynamics
    Pavg:dict
    Qavg:dict
    
    # Worst-case dynamics assuming full observability
    PrMdp:dict
    QrMdp:np.ndarray
    
    def learn_robust_model_Env_alpha(self, env: Env, alpha:float, N_standard=None, N_robust=None, df = 0.95):
        """Learn robust model from AM_Env class, assuming uncertainty is equal for all transitions and given by parameter alpha."""
        
        self.set_constants_env(env)
        if N_robust is None:
            N_robust = 200
        self.uP_from_alpha(alpha)
        self.learn_RMDP(N_robust, df)
        
    def set_constants_env(self, env):
        """Reads constants from AM_environment"""
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()
        self.StateSize += 1
    
    
    def learn_MDP_env(self, env, N_standard, df):
        """Learn the MDP-model from an AM environment (using ModelLearner module)"""
        env_expl = AM_Environment_Explicit()
        env_expl.learn_model_AMEnv(env, N_standard, df = df)
        self.Pavg, self.R, self.Qavg = env_expl.get_tables()
    
    def import_MDP_env(self, fileName, folder):
        env_expl = AM_Environment_Explicit()
        env_expl.import_model(fileName, folder)
        self.Pavg, self.R, self.Qavg = env_expl.get_tables()
        self.set_constants_env(env_expl)
        
    def learn_RMDP(self, N_robust, df):
        """Learn the worst-case transition and Q-function (using ModelLearner_Robust module), given the uMDP is already initialised in this class."""
        robustLearner = ModelLearner_Robust(self, df = df)
        robustLearner.run(updates=N_robust)
        self.PrMdp, self.QrMdp = robustLearner.get_model()
        
    def uP_from_alpha(self, alpha):
        """Set Pmin and Pmax, according to self.P and alpha"""
        self.Pmin, self.Pmax = {}, {}
        for s in range(self.StateSize):
            self.Pmin[s], self.Pmax[s] = {}, {}
            for a in range(self.ActionSize):
                self.Pmin[s][a], self.Pmax[s][a] = {}, {}
                for (snext, prob) in self.Pavg[s][a].items():
                    self.Pmin[s][a][snext], self.Pmax[s][a][snext] = 0, np.min([prob*1/alpha, 1])
        
    def env_to_dict(self):
        """Returns dictiorary with all environment variables"""
        dict_robust =   {
                            "PrMdp":   self.PrMdp,
                            "QrMdp":   self.QrMdp,
                            "Pmin":    self.Pmin,
                            "Pmax":    self.Pmax,
                            "Pavg":    self.Pavg,
                            "Qavg":    self.Qavg,
                            "R":       self.R
                        }
        dict_robust.update(super().env_to_dict())
        return dict_robust
    
    def env_from_dict(self, dict):
        """Changes class variables to those specified in dict"""
        super().env_from_dict(dict)
        self.Pavg, self.Qavg, self.R = dict["Pavg"], np.array(dict["Qavg"]), dict["R"]
        self.Pmin, self.Pmax         = dict["Pmin"] , dict["Pmax"]
        self.PrMdp, self.QrMdp       = dict["PrMdp"], np.array(dict["QrMdp"])
    
    def get_avg_tables(self):
        """Returns (Pavg, R, Qavg)"""
        return self.Pavg, self.R, self.Qavg
    
    def get_uncertain_tables(self):
        """Returns (Pmin, Pmax, R)"""
        return self.Pmin, self.Pmax, self.R
        
    def get_robust_tables(self):
        "returns Pr, Qr, R for robust MDP"
        return self.PrMdp, self.QrMdp, self.R
    
    def randomize(self, beta, nmbr_tries = 25):
        """Randomizes PrMdp by shifting all probs by at p ~ [-beta, beta]."""
        for s in range(self.StateSize):
            for a in range(self.ActionSize):
                valid=False
                l = len(self.PrMdp[s][a])
                if l > 1:
                    i=0
                    while not valid and i<nmbr_tries:
                        valid=True
                        shift = np.random.random(size=len(self.PrMdp[s][a])-1)
                        shift = list(2*beta*shift - beta)
                        shift.append(-sum(shift))
                        for (i,snext) in enumerate(self.PrMdp[s][a].keys()):
                            valid = valid and ( self.PrMdp[s][a][snext] + shift[i] < self.Pmax[s][a][snext] and
                                                self.PrMdp[s][a][snext] + shift[i] > self.Pmin[s][a][snext])
                        if i>= 25:
                            shift = np.zeros(l)
                            print(f"WARNING: randomisation did not work for state-action pair {s},{a}")
                    
                    for (i,snext) in enumerate(self.PrMdp[s][a].keys()):
                        self.PrMdp[s][a][snext] += shift[i]

                    
                


class OptAM_Environment_Explicit(RAM_Environment_Explicit):
    """Class to explicitely express uncertain AM environments, i.e. with matrixes for uP, R and Q. 
    Additionally contains an explicit copy of an \'average\' AM environment to be used by some functions."""
    
    def learn_RMDP(self, N_robust, df):
        """Learn the worst-case transition and Q-function (using ModelLearner_Robust module), given the uMDP is already initialised in this class."""
        robustLearner = ModelLearner_Robust(self, df = df, optimistic = True)
        robustLearner.run(updates=N_robust)
        self.PrMdp, self.QrMdp = robustLearner.get_model()
    
    
    
def make_negative_recursively(dict:dict, depth:int):
    if depth == 1:
        for (key, val) in dict.items():
            dict[key] = -val
    elif depth > 1:
        for (key, val) in dict.items():
            dict[key] = make_negative_recursively(val, depth-1)
    return dict
            
class IntKeyDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(int(key), value)