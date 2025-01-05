from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove, \
    SudokuSettings, allowed_squares
import copy
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from typing import Optional
from typing import List
from team05_A3.solver import solve_sudoku, str2grid, grid2str
from team05_A3.Sudoku import SudokuException


# print("Hello")

class SudokuEnv(gym.Env):
    
    def __init__(self, m, n):
        
        super(SudokuEnv, self).__init__()
        initial_board = SudokuBoard(m, n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])
        
        self.board = np.zeros((m*n, m*n), dtype=int)
        
        self.action_space = spaces.Discrete(1 + (m*n * m*n * m*n))
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(m*n, m*n, m*n), dtype=np.uint8)
        
        self.done = False
    
    def convert_to_move(self, action):
        action_adjusted = action - 1
        # print("Hello1")
        # print("Action is: ", action)
        # move = Move((action[0], action[1]), action[2] + 1)
        # print("Hello2")
        # print("Action is: ", action)
        # print("Move is: ", move)
        # Extracting the individual components from the flattened space
        val = action_adjusted // (self.game_state.board.N * self.game_state.board.N)
        i = (action_adjusted % ((self.game_state.board.N)**2)) // (self.game_state.board.N)
        j = (action_adjusted % ((self.game_state.board.N)**2)) % (self.game_state.board.N) # action -> move

        # Now use these components to create the move
        move = Move((i, j), val + 1)
        
        # print("Action is: ", action)
        # print("Move created is: ", move)
        
        return move
    
    def _get_obs(self):
        board_interim = np.zeros((self.game_state.board.N, self.game_state.board.N, self.game_state.board.N), dtype=np.uint8)
        for i in range(self.game_state.board.N):
            for j in range(self.game_state.board.N):
                # print("Index is!!: ", (i,j))
                # print(self.game_state)
                if self.board[i,j] != 0:
                    board_interim[self.board[i,j] - 1, i,j] = 1
        # print(self.game_state)
        # print(board_interim)
        return board_interim
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        # print("Final game state before reset is: ")
        '''calculcate filled% of board before reseting it'''
        num = self.game_state.board.N**2 - np.sum(self.board==0)
        perc = (num / self.game_state.board.N**2) * 100
        print("Percentage filled is: ", perc, "%")
        print(self.game_state.board)
        print(self.game_state.scores)
        
        initial_board = SudokuBoard(self.game_state.board.m, self.game_state.board.n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])
        
        self.board = np.zeros((self.game_state.board.N, self.game_state.board.N), dtype=int)
        self.done = False
        return self._get_obs(), {}
    
    def action_masks(self):
        
        action_mask_interim = np.full((self.game_state.board.N, self.game_state.board.N, self.game_state.board.N, 1), False)
        # print("Getting legal moves: ")
        for move in self.get_legal_moves():
            # print(move)
            action_mask_interim[move.value-1, move.square[0], move.square[1]] = True
        action_mask_interim = action_mask_interim.flatten()
        action_mask_interim = action_mask_interim.tolist()
        if True not in action_mask_interim:
            action_mask_interim = [True] + action_mask_interim
        else:
            action_mask_interim = [False] + action_mask_interim
        return action_mask_interim
    
    def step(self, action):
        # print('hell naw')
        
        # print("Action is: ", action)
        # print("Move is: ", move)

        if action == 0:
            reward = -5
            self.game_state.current_player = 3 - self.game_state.current_player
            if not self.get_legal_moves():
                self.done = True
        else:
            move = self.convert_to_move(action)
            reward = self.add_to_game_state(move)
            reward += 0.2

        # print(self.game_state)
        # print(f"{self.done = }")

        if (np.sum(self.board == 0) == 0):  # Board is full
            # print("Number of occupied squares is: ", len(self.game_state.occupied_squares()))
            self.done = True
        
        return self._get_obs(), reward, self.done, False, {}
    
    def get_constraints(self, move: Move):

        N = self.game_state.board.m * self.game_state.board.n
        row_values = [self.game_state.board.get((move.square[0], j)) for j in range(
            N) if self.game_state.board.get((move.square[0], j)) != SudokuBoard.empty]
        col_values = [self.game_state.board.get((i, move.square[1])) for i in range(
            N) if self.game_state.board.get((i, move.square[1])) != SudokuBoard.empty]
        block_i = (move.square[0] // self.game_state.board.m) * self.game_state.board.m
        block_j = (move.square[1] // self.game_state.board.n) * self.game_state.board.n
        block_values = [
            self.game_state.board.get((i, j))
            for i in range(block_i, block_i + self.game_state.board.m)
            for j in range(block_j, block_j + self.game_state.board.n)
            if self.game_state.board.get((i, j)) != SudokuBoard.empty
        ]
        return row_values, col_values, block_values
    
    def add_to_game_state(self, move):

        reward = self.evaluate_score(move)
        self.game_state.scores[self.game_state.current_player - 1] += reward
        self.game_state.board.put(move.square, move.value)
        self.game_state.moves.append(move)
        self.game_state.occupied_squares().append(move.square)
        self.board[move.square[0], move.square[1]] = move.value
        self.game_state.current_player = 3 - self.game_state.current_player  # Toggle between player 1 and 2
        return reward
    
    def evaluate_score(self, move):

        row_values, col_values, block_values = self.get_constraints(move)
        solves_row = len(row_values) == self.game_state.board.N - 1
        solves_col = len(col_values) == self.game_state.board.N - 1
        solves_block = len(block_values) == self.game_state.board.N - 1
        sum = solves_row + solves_col + solves_block
        score = 0
        if sum == 0:
            score = 0
        elif sum == 1:
            score = 1
        elif sum == 2:
            score = 3
        elif sum == 3:
            score = 7
        return score
    
    # Check if a move is valid
    def is_valid_move(self, i, j, value):
        return (
            self.game_state.board.get((i, j)) == SudokuBoard.empty
            and TabooMove((i, j), value) not in self.game_state.taboo_moves
            and (i, j) in self.game_state.player_squares()
        )
        
    # Check allowed moves to generate legal_moves
    def get_legal_moves(self):
        moves = []
        for i in self.game_state.player_squares():
            for value in range(1, self.game_state.board.N + 1):
                if self.is_valid_move(i[0], i[1], value):
                    row_values, col_values, block_values = self.get_constraints(Move(i, value))
                    if value not in row_values + col_values + block_values:
                        moves.append(Move(i, value))

        
        num = self.game_state.board.N**2 - np.sum(self.board == 0)
        if num > 23:
            moves_solvable = []
            for move in moves:
                
                board_copy = copy.deepcopy(self.board)
                # print("Initial Board is: ")
                # print(board_copy)
                board_copy[move.square[0], move.square[1]] = move.value
                # print("New board is: ")
                # print(board_copy)

                '''Solving Sudoku'''
                
                flattened_string = ''.join(map(str, board_copy.flatten()))
                # puzzle   = '400009200000010080005400006004200001050030060700005300500007600090060000002800007'
                grid1 = str2grid(flattened_string)
                # grid2 = str2grid(puzzle)
                # string1 = grid2str(grid1)
                # print("String is: ", string1)
                # print(grid1)
                # print("\n")
                # print(grid2)
                # print("Type is: ", type(grid1))
                grid_tuple = tuple(tuple(inner_list) for inner_list in grid1)
                # print(grid_tuple)
                check = 0       # Checking current move in legal moves
                
                try:
                    solution_set, done, info = solve_sudoku(grid_tuple)
                    if(len(solution_set) == 0):
                        check = 1
                except SudokuException as e:                    
                    check = 1

                # print("Sudoku Solver")
                if(check == 0):
                    moves_solvable.append(move)

                # print("Checking move, board copy is: ")
                # print(board_copy)
                # if solve_sudoku(board_copy):
                moves_solvable.append(move)
            # return moves
            return moves_solvable
        
        return moves
    
    def split(self, num_envs):
        # Return num_envs copies of the environment
        return [SudokuEnv() for _ in range(num_envs)]
    
# def unflatten(arr: List[int], n=9):
#     grid = []
#     for i in range(0, len(arr), n):
#         grid.append(arr[i:i+n])
#     return grid

# def str2arr(string: str, blank:str = '.'):
#     arr = []
#     end = string.find('-')
#     end = len(string) if end == -1 else end
#     for c in string[0:end]:
#         if c == blank:
#             arr.append(0)
#         else:
#             arr.append(int(c))
#     return arr  # [int(c) for c in string]

# def str2grid(string: str) -> List[List[int]]:
#     return unflatten(str2arr(string))

# def grid2str(grid: List[List[int]]) -> str:
#     return arr2str(flatten(grid))

# def flatten(grid: List[List[int]]):
#     arr = []
#     for row in grid:
#         arr.extend(row)
#     return arr

# def arr2str(arr: List[int]):
#     string = ''
#     for digit in arr:
#         string += str(digit)
#     return string

'''
def solve_sudoku(board):
    # Prepare bitmasks for each row, column, and 3x3 box
    row_mask = np.zeros((9, 9), dtype=bool)
    col_mask = np.zeros((9, 9), dtype=bool)
    box_mask = np.zeros((9, 9), dtype=bool)
    # Initialize the masks based on the given board
    for r in range(9):
        for c in range(9):
            if board[r, c] != 0:
                num = board[r, c] - 1
                row_mask[r, num] = True
                col_mask[c, num] = True
                box_mask[(r // 3) * 3 + (c // 3), num] = True
    # Function to perform the solve using bitmasking
    def backtrack(cell_idx=0):
        if cell_idx == 81:  # All cells filled
            return True
        r, c = divmod(cell_idx, 9)
        if board[r, c] != 0:  # Skip filled cells
            return backtrack(cell_idx + 1)
        # Try numbers 1 through 9
        for num in range(9):
            if not row_mask[r, num] and not col_mask[c, num] and not box_mask[(r // 3) * 3 + (c // 3), num]:
                # Place the number
                board[r, c] = num + 1
                row_mask[r, num] = col_mask[c, num] = box_mask[(r // 3) * 3 + (c // 3), num] = True
                # Recurse to the next cell
                if backtrack(cell_idx + 1):
                    return True
                # Backtrack
                board[r, c] = 0
                row_mask[r, num] = col_mask[c, num] = box_mask[(r // 3) * 3 + (c // 3), num] = False
        return False
    # Start backtracking from the first empty cell
    return backtrack()
        
    

    # Random move fallback
    def move_opposing_player(self):
        if self.get_legal_moves():
            move = random.choice(self.get_legal_moves())
            self.add_to_game_state(move)
            greedy_score = self.evaluate_score(move)
        
        # Greedy fallback
        for move in self.get_legal_moves():
            reward = self.evaluate_score(move)
            if greedy_score < reward:
                greedy_score = reward
                self.add_to_game_state(move)
    '''


if __name__ == "__main__":
    print('hell naw')