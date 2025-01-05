from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sudoku import Sudoku
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

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


print("Hello")

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

        print("Final game state before reset is: ")
        print(self.game_state)
        
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

        # print(self.game_state)
        # print(f"{self.done = }")

        if len(self.game_state.occupied_squares()) == self.game_state.board.N**2:  # Board is full
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

        if len(self.game_state.occupied_squares()) > 16:
            moves_solvable = []
            for move in moves:
                board_copy = copy.deepcopy(self.board)
                board_copy[move.square[0], move.square[1]] = move.value
                board_copy = board_copy.tolist()
                sudoku = Sudoku(self.game_state.board.m, self.game_state.board.n, board=board_copy)
                if sudoku.solve():
                    moves_solvable.append(move)
            # return moves
            return moves_solvable

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

# env = DummyVecEnv([lambda: env])

# Now wrap the environment with VecNormalize, setting normalize_images=False
# env = VecNormalize(env, norm_obs=False, norm_reward=False, normalize_images=False)


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor class for processing image observations.
    
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of units for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Extract the shape of the input image from observation_space
        channels, height, width = observation_space.shape
        
        # Dynamically set the kernel size and number of channels based on m and n
        self.m = 9  # Assuming m*n is height and width, and m*n is channels
        
        # Define CNN layers (adjust based on m and n)
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=self.m // 1, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=self.m // 2, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the output size after convolution layers to determine the input size for the linear layer
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
        
        # Define a fully connected layer after convolutional layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pass observations through the CNN layers and then the linear layers
        return self.linear(self.cnn(observations))
    


def main():
    # Train a PPO agent
    action_masks = get_action_masks(env)
    # print(action_masks)
    # print("Shape is: ", action_masks.shape)
    policy_kwargs = dict(
        normalize_images = True,
    #features_extractor_class=CustomCNN,
    #features_extractor_kwargs=dict(features_dim=128),
    )
    model = MaskablePPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=100_000, use_masking=True)
    model.save("sudoku_rl_agent")

if __name__ == '__main__':
    main()