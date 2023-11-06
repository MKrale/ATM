"""
Maze environment
"""
import os
import gym
import numpy as np
from typing import List, Tuple, Dict
from gym import spaces

from AM_Gyms.Maze.maze_view_2d import MazeView2D

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "samples")


class MazeEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array"]
    }

    # Define constants for clearer code
    actions: List[str] = ["N", "S", "W", "E"]

    def __init__(self, screen_size: Tuple[int, int] = (500, 500),
                 maze_file_path: str = None, maze_size: Tuple[int, int] = (10, 10),
                 breakChance = 0 ):
        """Initializes maze environment.

        Args:
            screen_size (tuple): screen size
            maze_size (tuple): maze size
            maze_file_path (str): maze file path
        """
        self.maze_size: Tuple[int, int] = maze_size

        # Creates a new maze view for the environment
        self.maze_view: MazeView2D = MazeView2D(caption="OpenAI Gym - Maze (%d x %d)" % maze_size,
                                                screen_size=screen_size, maze_size=maze_size,
                                                maze_file_path=maze_file_path)

        # Defines action space
        # They must be gym.spaces objects
        self.action_space: spaces = spaces.Discrete(2 * len(self.maze_size))

        # Defines observation space
        # The observation will be the coordinate of the agent
        low: np.ndarray = np.zeros(len(self.maze_size), dtype=int)
        high: np.ndarray = np.array(self.maze_size, dtype=int) - np.ones(len(self.maze_size), dtype=int)
        self.observation_space: spaces.Box = spaces.Box(low, high, dtype=np.int64)
        self.breakChance = breakChance

    def step(self, action: int or str) -> Tuple[np.array, float, bool, Dict]:
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (Union[int, str]): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)

        """
        if np.random.random() > self.breakChance:
            self.maze_view.move_robot(self.actions[action])
            
        if np.array_equal(self.maze_view.robot, self.maze_view.goal):
            reward: float = 1
            done: bool = True
        else:
            reward: float = -0.01
            done: bool = False

        # Optionally we can pass additional info, we are not using that for now
        info: Dict = {}

        ######
        # Get unraveled state
        ######
        
        s = int(np.ravel_multi_index(self.maze_view.robot, self.maze_size))
        return s, reward, done, info
    
    def set_state(self, state):
        pos = np.unravel_index(state, self.maze_size)
        self.maze_view.robot = pos

    def reset(self) -> np.array:
        """Resets the state of the environment and returns an initial observation.

        Returns:
            observation (object): the initial observation.

        """
        self.maze_view.reset_game()
        return self.maze_view.robot

    def render(self, mode: str = 'human'):
        """Renders the environment.

        Args:
            mode (str): the mode to render with

        """
        # Handles the user input
        # self.maze_view.process_input()

        # if mode in ['human', 'rgb_array']:
        #     return self.maze_view.render(mode)
        # else:
        #     super(MazeEnv, self).render(mode=mode)
    
    def getname(self):
        return "ZMaze_{}_p{}".format(self.maze_size[0], self.breakChance)


class MazeEnvSample5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvSample5x5, self).__init__(screen_size=(500, 500), maze_size=(5, 5),
                                               maze_file_path=os.path.join(SAMPLES_DIR, "maze2d_5x5.npy"))


class MazeEnvRandom5x5(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom5x5, self).__init__(screen_size=(500, 500), maze_size=(5, 5),
                                               maze_file_path=None)


class MazeEnvSample10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvSample10x10, self).__init__(screen_size=(500, 500), maze_size=(10, 10),
                                                 maze_file_path=os.path.join(SAMPLES_DIR, "maze2d_10x10.npy"))


class MazeEnvRandom10x10(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom10x10, self).__init__(screen_size=(500, 500), maze_size=(10, 10),
                                                 maze_file_path=None)


class MazeEnvSample25x25(MazeEnv):

    def __init__(self):
        super(MazeEnvSample25x25, self).__init__(screen_size=(500, 500), maze_size=(25, 25),
                                                 maze_file_path=os.path.join(SAMPLES_DIR, "maze2d_25x25.npy"))


class MazeEnvRandom25x25(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom25x25, self).__init__(screen_size=(500, 500), maze_size=(25, 25),
                                                 maze_file_path=None)


class MazeEnvSample50x50(MazeEnv):

    def __init__(self):
        super(MazeEnvSample50x50, self).__init__(screen_size=(600, 600), maze_size=(50, 50),
                                                 maze_file_path=os.path.join(SAMPLES_DIR, "maze2d_50x50.npy"))


class MazeEnvRandom50x50(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom50x50, self).__init__(screen_size=(600, 600), maze_size=(50, 50),
                                                 maze_file_path=None)


class MazeEnvSample100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvSample100x100, self).__init__(screen_size=(700, 700), maze_size=(100, 100),
                                                   maze_file_path=os.path.join(SAMPLES_DIR, "maze2d_100x100.npy"))


class MazeEnvRandom100x100(MazeEnv):

    def __init__(self):
        super(MazeEnvRandom100x100, self).__init__(screen_size=(700, 700), maze_size=(100, 100),
                                                   maze_file_path=None)
