"""
Maze definition
"""
import random
import numpy as np
from typing import Dict, List, Tuple


class Cell:
    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs: Dict[str, str] = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x: int, y: int):
        """Initializes the cell at (x,y), surrounded by walls.

        Args:
            x (int): horizontal coordinate
            y (int): vertical coordinate

        """
        self.x: int = x
        self.y: int = y

        # A cell have all the walls at the beginning
        self.walls: Dict[str, bool] = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self) -> bool:
        """Checks if cell have all the walls.

        Returns:
            bool: True if cell have all the walls, False otherwise.

        """
        return all(self.walls.values())

    def break_wall(self, other: 'Cell', direction: str):
        """Breaks down the wall between cells (self and other).

        Args:
            other (Cell): cell object
            direction (str): navigation direction (N,S,E,W)

        """
        self.walls[direction] = False
        other.walls[Cell.wall_pairs[direction]] = False


class Maze:
    # A maze compass used for navigation between cells
    compass: Dict[str, Tuple[int, int]] = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}

    def __init__(self, maze_size: Tuple[int, int] = (5, 5), maze_file_path: str = None):
        """Initializes the maze grid, consists of (nx,ny) cells.

        Args:
            maze_size (tuple): maze size
            maze_file_path (str): file path

        """
        self.nx: int = maze_size[0]
        self.ny: int = maze_size[1]

        # Initializes all the maze cells, where cell have all the walls at the beginning
        self.cells: List[List['Cell']] = [[Cell(x, y) for y in range(self.ny)] for x in range(self.nx)]

        if maze_file_path:
            print("Maze loaded!")
            self.load_maze(maze_file_path)
        else:
            self.generate_maze()

    def cell_at(self, x: int, y: int) -> 'Cell':
        """Gets the cell object at (x,y) coordinates.

        Args:
            x (int): horizontal coordinate
            y (int): vertical coordinate

        Returns:
            Cell: cell object at (x,y) coordinates.

        """
        return self.cells[x][y]

    def find_valid_neighbours(self, cell: 'Cell') -> List[Tuple[str, Cell]]:
        """Gets a list of unvisited neighbors to the cell.

        Args:
            cell (Cell): cell object

        Returns:
            list: a list of unvisited neighbours

        """
        neighbours = []

        for direction, (dx, dy) in Maze.compass.items():
            x2, y2 = cell.x + dx, cell.y + dy
            if 0 <= x2 < self.nx and 0 <= y2 < self.ny:
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def generate_maze(self):
        """Generates maze using the Depth-first search algorithm."""
        # 1. Choose initial cell, mark it as visited and push it to the stack
        current_cell: 'Cell' = self.cell_at(0, 0)
        cell_stack: List['Cell'] = [current_cell]

        # 2. If the stack is not empty
        while cell_stack:
            # 2.1 Pop a cell from the stack and make it a current cell
            current_cell = cell_stack.pop()
            unvisited_neighbours: List[Tuple[str, Cell]] = self.find_valid_neighbours(current_cell)

            # 2.2 If the current cell has any neighbours which have not been visited
            if unvisited_neighbours:
                # 2.2.1 Push the current cell to the stack
                cell_stack.append(current_cell)
                # 2.2.2 Choose one of the unvisited neighbours
                wall_direction, next_cell = random.choice(unvisited_neighbours)
                # 2.2.3 Remove the wall between the current cell and the chosen cell
                current_cell.break_wall(next_cell, wall_direction)
                # 2.2.4 Mark the chosen cell as visited and push it to the stack
                cell_stack.append(next_cell)

    def save_maze(self, maze_file_path: str):
        """Saves the current generated maze to a file.

        Args:
            maze_file_path (str): file path

        """
        np_cells: np.ndarray = np.zeros((self.nx, self.ny), dtype=int)

        for x in range(self.nx):
            for y in range(self.ny):
                for i, direction in enumerate(self.compass.keys()):
                    if self.cells[x][y].walls[direction]:
                        np_cells[x][y] |= 2 ** i

        np.save(maze_file_path, np_cells, allow_pickle=False, fix_imports=True)

    def load_maze(self, maze_file_path: str):
        """Loads a previous generated maze from a file.

        Args:
            maze_file_path (str): file path

        """
        np_cells: np.ndarray = np.load(maze_file_path, allow_pickle=False, fix_imports=True)

        for x in range(self.nx):
            for y in range(self.ny):
                for i, direction in enumerate(self.compass.keys()):
                    if np_cells[x, y] & 2 ** i == 0:
                        self.cells[x][y].walls[direction] = False

# # 0:N, S:1, E:2, W:3
# # for (i, direction) in enumerate(Maze.compass.keys()):
# #     print(i, direction)

# for x in range(5):


#     for j in range(10)

# np.save("samples/maze2d_snake_10x210.npy", allow_pickle = False, fix_imports=True)