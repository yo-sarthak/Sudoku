import random
from competitive_sudoku.sudoku import GameState, Move, SudokuBoard, TabooMove, \
    SudokuSettings, allowed_squares
import copy
# import competitive_sudoku.sudokuai
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from typing import Optional

print("Hello")

class SudokuEnv(gym.Env):
    
    def __init__(self, m, n):
        
        super(SudokuEnv, self).__init__()
        
        initial_board = SudokuBoard(m, n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])
        
        self.board = np.zeros((m*n, m*n), dtype=int)
        
        self.action_space = spaces.MultiDiscrete([m*n, m*n, m*n])
        self.observation_space = spaces.Box(
            low=0, high=m*n, shape=(m*n, m*n), dtype=np.int32
        )
        self.done = False
    
    def convert_to_move(self, action):
        # print("Hello1")
        # print("Action is: ", action)
        move = Move((action[0], action[1]), action[2] + 1)
        # print("Hello2")
        # print("Action is: ", action)
        # print("Move is: ", move)
        return move
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        
        initial_board = SudokuBoard(self.game_state.board.m, self.game_state.board.n)
        allowed_squares1, allowed_squares2 = allowed_squares(initial_board, playmode='rows')
        self.game_state = GameState(initial_board=initial_board, allowed_squares1=allowed_squares1, occupied_squares1=[], allowed_squares2=allowed_squares2, occupied_squares2=[])
        
        self.board = np.zeros((self.game_state.board.N, self.game_state.board.N), dtype=int)
        self.done = False
        return self.board, {}
    
    def _get_obs(self):
        return
    
    def step(self, action):
        
        move = self.convert_to_move(action)
        
        if move in self.get_legal_moves():
            reward = self.add_to_game_state(move)
        else:
            reward = -10
            self.done = True

        # print(f"{self.done = }")

        if np.all(self.board):  # Board is full
            self.done = True
        
        return self.board, reward, self.done, False, {}
    
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
        # return moves
        return moves
    
    '''
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
        
# Create the environment
env = SudokuEnv(3, 3)


def main():
    # Train a PPO agent
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)
    model.save("sudoku_rl_agent")

if __name__ == '__main__':
    main()