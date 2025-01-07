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
from team05_A3.PythonSolverUnique import numpy_to_sudoku_format, SudokuPuzzle, depth_first_solve
from stable_baselines3.common.callbacks import BaseCallback


# print("Hello")

class SudokuEnv(gym.Env):
    
    def __init__(self, m, n, eval_env: bool = False):
        
        super(SudokuEnv, self).__init__()
        initial_board = SudokuBoard(m, n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])
        
        self.board = np.zeros((m*n, m*n), dtype=int)
        
        self.action_space = spaces.Discrete(1 + (m*n * m*n * m*n))
        
        self.observation_space = spaces.Box(low=0, high=1, shape=(m*n, m*n, m*n), dtype=np.uint8)
        self.rewards = [0,0]
        self.infos = {}
        self.win = 0
        self.eval_env = eval_env
        self.done = False
        self.get_logables()

    def get_logables(self):
        self.infos = {}
        self.infos['player1_score'], self.infos['player2_score']= self.game_state.scores
        self.infos['done'] = self.done
        self.infos['fill_percentage'] = (1 - (np.sum(self.board == 0) / self.game_state.board.N**2)) * 100
        self.infos['reward_1'] = self.rewards[0]
        self.infos['reward_2'] = self.rewards[1]
    
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

        self.get_logables()
        # print("Reset player is: ", self.game_state.current_player)
        # print("Final game state before reset is: ")
        '''calculcate filled% of board before reseting it'''
        num = self.game_state.board.N**2 - np.sum(self.board==0)
        perc = (num / self.game_state.board.N**2) * 100
        
        if self.eval_env == False:
            print("Percentage filled is: ", perc, "%")
            print(self.game_state.board)
            print(self.game_state.scores)
        
        initial_board = SudokuBoard(self.game_state.board.m, self.game_state.board.n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])

        self.win = 0
        self.rewards = [0,0]
        
        self.board = np.zeros((self.game_state.board.N, self.game_state.board.N), dtype=int)
        self.done = False
        return self._get_obs(), {}
    
    def action_masks(self):
        
        action_mask_interim = np.full((self.game_state.board.N, self.game_state.board.N, self.game_state.board.N), False)
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

        self.get_logables()

        # print('hell naw')
        # print("Action is: ", action)
        # print("Move is: ", move)
        # print("Current player is: ", self.game_state.current_player)
        
        '''Game State Encoded Reward'''

        reward = 0
        
        if (np.sum(self.board == 0) != 0):
            if action == 0:

                self.game_state.current_player = 3 - self.game_state.current_player
                if not self.get_legal_moves():
                    reward -= 20
                    self.done = True
                self.game_state.current_player = 3 - self.game_state.current_player

            else:
                move = self.convert_to_move(action)
                self.add_to_game_state(move)
        
            reward += (self.game_state.scores[self.game_state.current_player - 1] - self.game_state.scores[2 - self.game_state.current_player]) / 10

        # Board is full
        if (np.sum(self.board == 0) == 0):
            
            # Game is tied
            if self.game_state.scores[0] == self.game_state.scores[1]:
                self.done = True
            
            # Rewards are updated to reflect win
            else:
                if self.win < 2:    # Winner/loser scores are not updated

                    winner = self.game_state.scores.index(max(self.game_state.scores)) + 1
                    # diff = abs(self.game_state.scores[self.game_state.current_player - 1] - self.game_state.scores[2 - self.game_state.current_player])

                    if winner == self.game_state.current_player:
                        reward += 2 * self.game_state.scores[winner - 1]
                        self.win += 1
                    else:
                        reward -= 2 * self.game_state.scores[2 - winner]
                        self.win += 1
            
                else:
                    self.done = True

        self.rewards[self.game_state.current_player - 1] += reward
        self.game_state.current_player = 3 - self.game_state.current_player
        
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
        # self.game_state.current_player = 3 - self.game_state.current_player  # Toggle between player 1 and 2
        # return reward
    
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
        if num > 16:

            moves_solvable = []
            for move in moves:
                
                board_copy = copy.deepcopy(self.board)
                # print("Initial Board is: ")
                # print(board_copy)
                board_copy[move.square[0], move.square[1]] = move.value
                # board_copy = board_copy.tolist()
                # print("New board is: ")
                # print(board_copy)

                '''Solving Sudoku'''

                N = len(board_copy)
                symbol_set = {str(i) for i in range(1, N + 1)}
                board_copy = numpy_to_sudoku_format(board_copy)
                puzzle = SudokuPuzzle(n = N, symbols=board_copy, symbol_set=symbol_set)
                solution = depth_first_solve(puzzle)
                
                if solution:
                    moves_solvable.append(move)


                # flattened_string = ''.join(map(str, board_copy.flatten()))
                # # puzzle   = '400009200000010080005400006004200001050030060700005300500007600090060000002800007'
                # grid1 = str2grid(flattened_string)
                # # grid2 = str2grid(puzzle)
                # # string1 = grid2str(grid1)
                # # print("String is: ", string1)
                # # print(grid1)
                # # print("\n")
                # # print(grid2)
                # # print("Type is: ", type(grid1))
                # grid_tuple = tuple(tuple(inner_list) for inner_list in grid1)
                # # print(grid_tuple)
                # check = 0       # Checking current move in legal moves
                
                # try:
                #     solution_set, done, info = solve_sudoku(grid_tuple)
                #     if(len(solution_set) == 0):
                #         check = 1
                # except SudokuException as e:                    
                #     check = 1

                # # print("Sudoku Solver")
                # if(check == 0):
                #     moves_solvable.append(move)

                # # print("Checking move, board copy is: ")
                # # print(board_copy)
                # # if solve_sudoku(board_copy):
                # moves_solvable.append(move)
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

# def is_valid_check(board, num, row, col):
#     # Check the row
#     if num in board[row, :]:
#         return False

#     # Check the column
#     if num in board[:, col]:
#         return False

#     # Check the 3x3 box
#     start_row, start_col = 3 * (row // 3), 3 * (col // 3)
#     if num in board[start_row:start_row + 3, start_col:start_col + 3]:
#         return False

#     return True

# def solve_sudoku_check(board):
#     # Find an empty cell (represented by 0)
#     for row in range(9):
#         for col in range(9):
#             if board[row, col] == 0:
#                 # Try numbers from 1 to 9
#                 for num in range(1, 10):
#                     if is_valid_check(board, num, row, col):
#                         board[row, col] = num
                        
#                         # Recursively solve the rest of the board
#                         if solve_sudoku_check(board):
#                             return True
                        
#                         # If it doesn't work, backtrack and try the next number
#                         board[row, col] = 0
                        
#                 return False  # If no valid number is found, return False

#     return True  # All cells are filled correctly

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.game_scores = [0, 0]

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # game_scores = [0,0]
        # game_scores[0] += self.locals['game_state.scores'][0]
        # game_scores[1] += self.locals['game_state.scores'][1]
        # a = self.logger.record("player1_score", self.training_env.get_attr("infos")[0]['player1_score'])
        # b = self.logger.record("player2_score", self.training_env.get_attr("infos")[0]['player2_score'])
        
        if (self.training_env.get_attr("infos")[0]['done']):
            
            # game_scores_average = game_scores
            # game_scores_average[0] = game_scores_average[0]/1000
            # game_scores_average[1] = game_scores_average[1]/1000
            # self.logger.dump(self.num_timesteps)
            # game_scores = self.locals['game_state.scores'][0]/1000
            # game_scores = self.locals
            # print(self.training_env.get_attr("infos")[0])
            # print(self.locals['infos']['player2_score'])
            # print(self.training_env.get_attr("infos")[0]['player1_score'])
            print(self.locals['rewards'])
            self.logger.record("player1_score", self.training_env.get_attr("infos")[0]['player1_score'])
            self.logger.record("player2_score", self.training_env.get_attr("infos")[0]['player2_score'])
            self.logger.record("fill percentage", self.training_env.get_attr("infos")[0]['fill_percentage'])
            self.logger.record("reward_1", self.training_env.get_attr("infos")[0]['reward_1'])
            self.logger.record("reward_2", self.training_env.get_attr("infos")[0]['reward_2'])
            
            self.logger.dump(self.num_timesteps)
        return True

if __name__ == "__main__":
    print('hell naw')