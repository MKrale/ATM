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
    R:np.ndarray
    Q:np.ndarray
    
    StateSize:int
    ActionSize:int
    MeasureCost:int
    s_init:int
    
    isLearned = False
    
    def learn_model(self, env:Env):
        """Learns explicit model from Gym class (unimplemented!)"""
        print("to be implemented!")
        
    def learn_model_AMEnv(self, env:AM_ENV, N = None, df = 0.8):
        """Learns explicit model from AM_ENV class"""
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()
        self.StateSize += 1
        if N == None:
            N = self.StateSize * self.ActionSize * 50   # just guessing how many are required...
        learner                 = ModelLearner(env, df = df)
        learner.run_visits()
        self.P, self.R, self.Q = learner.get_model()
        self.isLearned          = True
        
    def env_to_dict(self):
        """Returns dictiorary with all environment variables"""
        dict = {   "P":            self.P,
                    "R":            self.R,
                    "Q":            self.Q} 
        return super().env_to_dict() | dict
        
    def env_from_dict(self, dict):
        """Changes class variables to those specified in dict"""
        super().env_from_dict()
        self.P, self.R, self.Q = dict["P"], np.array(dict["R"]), np.array(dict["Q"])

    def get_tables(self):
        """Returns (P, R, Q)"""
        return self.P, self.R, self.Q
        
class RAM_Environment_Explicit(Environment_Explicit_Interface):
    """Class to explicitely express uncertain AM environments, i.e. with matrixes for uP, R and Q. 
    Additionally contains an explicit copy of an \'average\' AM environment to be used by some functions."""
    
    # Uncertain dynamics
    Pmin:dict
    Pmax:dict
    R:np.ndarray
    
    # Average-case dynamics
    Pavg:dict
    Qavg:dict
    
    # Worst-case dynamics assuming full observability
    PrMdp:dict
    QrMdp:np.ndarray
    
    def learn_robust_model_Env_alpha(self, env: Env, alpha:float, N_standard=None, N_robust=None, df = 0.95):
        """Learn robust model from AM_Env class, assuming uncertainty is equal for all transitions and given by parameter alpha."""
        
        # Set variables
        self.StateSize, self.ActionSize, self.MeasureCost, self.s_init = env.get_vars()
        self.StateSize += 1
        # NOTE: these numbers are just randomly chosen, I should investigate this further/maybe do some check?
        if N_robust is None:
            N_robust = np.min([self.StateSize * self.ActionSize, self.ActionSize * 100])
        if N_standard is None:
            N_standard = self.StateSize * self.ActionSize * 1000
        
        # Learn model from environment & create tables
        env_expl = AM_Environment_Explicit()
        env_expl.learn_model_AMEnv(env, N_standard, df = df)
        self.Pavg, self.R, self.Qavg = env_expl.get_tables()
        self.uP_from_alpha(alpha)
        
        # Learn worst-case model
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
                    self.Pmin[s][a][snext], self.Pmax[s][a][snext] = np.max([prob-alpha, 0]), np.min([prob+alpha, 1])
        
        
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
        return dict_robust | super().env_to_dict()
    
    def env_from_dict(self, dict):
        """Changes class variables to those specified in dict"""
        super().env_from_dict(dict)
        self.Pavg, self.Qavg, self.R = dict["Pavg"], np.array(dict["Qavg"]), np.array(dict["R"])
        self.Pmin, self.Pmax         = dict["Pmin"] , dict["Pmax"]
        self.PrMdp, self.QrMdp       = dict["PrMdp"], np.array(dict["QrMdp"])
    
    def get_avg_tables(self):
        """Returns (Pavg, R, Qavg)"""
        return self.Pavg, self.R, self.Qavg
    
    def get_robust_tables(self):
        """Returns (Pmin, Pmax, R)"""
        return self.Pmin, self.Pmax, self.R
        
    def get_worstcase_MDP_tables(self):
        "returns P, Q, R, Pmin & Pmax for robust MDP"
        return self.PrMdp, self.QrMdp

class IntKeyDict(dict):
    def __setitem__(self, key, value):
        super().__setitem__(int(key), value)

# Code for learning models:

# directoryPath = os.path.join(os.getcwd(), "AM_Gyms", "Learned_Models")
# alpha = 0.3

# from AM_Gyms.MachineMaintenance import Machine_Maintenance_Env
# from AM_Gyms.Loss_Env import Measure_Loss_Env
# # from MachineMaintenance import Machine_Maintenance_Env

# env_names           = ["Machine_Maintenance_a03", "Loss_a03"]

# envs                = [Machine_Maintenance_Env(N=8), Measure_Loss_Env()]
# env_stateSize       = [11,4]
# env_actionSize      = [2,2]
# env_sInit           = [0,0]

# for (i,env) in enumerate(envs):
#     AM_env = AM_ENV(env, env_stateSize[i], env_actionSize[i], 0, env_sInit[i])
#     modelLearner = RAM_Environment_tables()
#     modelLearner.learn_model_AMEnv_alpha(AM_env, alpha)
#     modelLearner.export_model(env_names[i], directoryPath)
