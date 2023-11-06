"""
Maze view 2d
"""

import numpy as np
import pygame
from typing import Tuple

from AM_Gyms.Maze.maze import Maze, Cell


class MazeView2D:
    # Defines colors palette
    BACKGROUND_COLOR = (255, 255, 255, 255)
    GAME_SURFACE_COLOR = (0, 0, 0, 0)
    LINE_COLOR = (0, 0, 0, 255)
    WALL_COLOR = (0, 0, 255, 15)

    def __init__(self, caption: str = "OpenAI Gym - Maze", screen_size: Tuple[int, int] = (500, 500),
                 maze_size: Tuple[int, int] = (5, 5), maze_file_path: str = None):
        """Initializes maze view.

        Args:
            caption (str): game window display name
            screen_size (tuple): screen size
            maze_size (tuple): maze size
            maze_file_path (str): maze file path

        """
        # Initializes game
        pygame.init()
        pygame.display.set_caption(caption)
    
        # # Defines display screen size
        # self.__screen_width: int = screen_size[0]
        # self.__screen_height: int = screen_size[1]
        # self.__screen: pygame.Surface = pygame.display.set_mode((self.__screen_width + 1, self.__screen_height + 1))

        # Defines the maze size
        self.__maze_size: Tuple[int, int] = maze_size

        # Calculates cell size
        # self.__cell_width: int = self.__screen_width // maze_size[0]
        # self.__cell_height: int = self.__screen_height // maze_size[1]

        # Creates a new maze
        self.__maze: Maze = Maze(maze_size=maze_size, maze_file_path=maze_file_path)

        # Sets maze entrance and goal
        self.__entrance: np.ndarray = np.zeros(2, dtype=int)
        self.__goal: np.ndarray = np.array(maze_size) - np.array((1, 1))

        # Sets the robot coordinates
        self.robot: np.ndarray = np.zeros(2, dtype=int)

        # Creates background
        # self.__background: pygame.Surface = pygame.Surface(self.__screen.get_size()).convert()
        # self.__background.fill(self.BACKGROUND_COLOR)

        # Creates the game layer
        # self.__game_surface: pygame.Surface = pygame.Surface(self.__screen.get_size()).convert_alpha(self.__screen)
        # self.__game_surface.fill(self.GAME_SURFACE_COLOR)

        # Draws game objects
        # self.__draw_maze()
        # self.__color_entrance()
        # self.__color_goal()
        # self.__color_robot()

    def __draw_maze(self):
        """Draws maze horizontal and vertical lines."""
        for y in range(self.maze_height + 1):
            pygame.draw.line(self.__game_surface,
                             self.LINE_COLOR,
                             (0, y * self.__cell_height),
                             (self.screen_width, y * self.cell_height))

        for x in range(self.maze_width + 1):
            pygame.draw.line(self.__game_surface,
                             self.LINE_COLOR,
                             (x * self.cell_width, 0),
                             (x * self.cell_width, self.screen_height))

        # Break maze walls
        for x in range(self.maze_width):
            for y in range(self.maze_height):
                for direction in ["N", "S", "E", "W"]:
                    if not self.__maze.cells[x][y].walls[direction]:
                        self.__color_wall(x, y, direction, self.WALL_COLOR)

    def __color_wall(self, x: int, y: int, direction: str, color: Tuple[int, int, int, int] = (0, 0, 255, 15)):
        """Colors the wall between cells with a given color.

        Args:
            x (int): horizontal coordinate
            y (int): vertical coordinate
            direction (str): navigation direction (N,S,E,W)
            color (tuple): an (RGB) triplet

        """
        return
        dx: int = x * self.cell_width
        dy: int = y * self.cell_height

        if direction == 'N':
            pygame.draw.line(self.__game_surface, color,
                             (dx + 1, dy),
                             (dx + self.cell_width - 1, dy))
        elif direction == 'S':
            pygame.draw.line(self.__game_surface, color,
                             (dx + 1, dy + self.cell_width),
                             (dx + self.cell_width - 1, dy + self.cell_height))
        elif direction == 'E':
            pygame.draw.line(self.__game_surface, color,
                             (dx + self.cell_width, dy + 1),
                             (dx + self.cell_width, dy + self.cell_height - 1))
        elif direction == 'W':
            pygame.draw.line(self.__game_surface, color,
                             (dx, dy + 1),
                             (dx, dy + self.cell_height - 1))
        else:
            raise ValueError("Only N,S,E,W values are allowed")

    def __color_cell(self, x: int, y: int, color: Tuple[int, int, int, int] = (255, 0, 0, 255)):
        """Colors the cell with a given color.

        Args:
            x (int): cell horizontal coordinate
            y (int): cell vertical coordinate
            color (tuple): an (RGB) triplet

        """
        return
        x0: int = x * self.cell_width + 1
        y0: int = y * self.cell_height + 1
        w0: int = self.cell_width - 1
        h0: int = self.cell_height - 1

        pygame.draw.rect(self.__game_surface, color, (x0, y0, w0, h0))

    def __color_entrance(self, color: Tuple[int, int, int, int] = (0, 0, 255, 150)):
        """Colors the maze entrance.

        Args:
            color (tuple): an (RGB) triplet

        """
        self.__color_cell(self.entrance[0], self.entrance[1], color)

    def __color_goal(self, color: Tuple[int, int, int, int] = (255, 0, 0, 150)):
        """Colors the maze goal.

        Args:
            color (tuple): an (RGB) triplet

        """
        self.__color_cell(self.goal[0], self.goal[1], color)

    def __color_robot(self, color: Tuple[int, int, int] = (0, 0, 0), transparency: int = 255):
        """Colors the maze robot.

        Args:
            color (tuple): an (RGB) triplet
            transparency (str): transparency value

        """
        return
        x0: int = self.robot[0] * self.cell_width + self.cell_width // 2
        y0: int = self.robot[1] * self.cell_height + self.cell_height // 2
        r0: int = min(self.cell_width, self.cell_height) // 5

        pygame.draw.circle(self.__game_surface, color + (transparency,), (x0, y0), r0)

    @staticmethod
    def process_input():
        """Handles the user input."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)

    def render(self, mode="human") -> np.flipud:
        """Renders the game objects.

        Args:
            mode (str): the mode to render with
een.blit(self.__background, (0, 0))
        # self.__screen.blit(self.__game_surface, (0, 0))

        # # Draws game objects
        # self.__draw_maze()
        # self.__color_entrance()
        # self.__color_goal()
        # self.__color_robot()

        # if mode == "human":
        #     pygame.display.flip()

        # return np.flipu
        """
        # self.__screen.blit(self.__background, (0, 0))
        # self.__screen.blit(self.__game_surface, (0, 0))

        # # Draws game objects
        # self.__draw_maze()
        # self.__color_entrance()
        # self.__color_goal()
        # self.__color_robot()

        # if mode == "human":
        #     pygame.display.flip()

        # return np.flipud(np.rot90(pygame.surfarray.array3d(pygame.display.get_surface())))

    def move_robot(self, direction: str):
        """Moves robot from a cell to another cell.

        Args:
            direction (str): navigation direction (N,S,E,W)

        """
        current_cell: Cell = self.__maze.cells[self.robot[0]][self.robot[1]]

        if not current_cell.walls[direction]:
            # self.__color_robot(transparency=0)
            self.robot = self.robot + np.array(Maze.compass[direction])

        # self.__color_robot()

    def reset_game(self):
        """Resets the game and place the robot back to the entrance."""
        # self.__color_robot(transparency=0)
        self.robot = np.zeros(2, dtype=int)

    @property
    def screen_width(self) -> int:
        return self.__screen_width

    @property
    def screen_height(self) -> int:
        return self.__screen_height

    @property
    def maze_width(self) -> int:
        return self.__maze_size[0]

    @property
    def maze_height(self) -> int:
        return self.__maze_size[1]

    @property
    def cell_width(self) -> int:
        return self.__cell_width

    @property
    def cell_height(self) -> int:
        return self.__cell_height

    @property
    def entrance(self) -> np.array:
        return self.__entrance

    @property
    def goal(self) -> np.array:
        return self.__goal

    # @property
    # def robot(self) -> np.array:
    #     return self.__robot
