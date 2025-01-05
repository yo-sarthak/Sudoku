from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from team05_A3.sudoku_gym import SudokuEnv
from team05_A3.SudokuSolver import solver


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
    
# Create the environment

env = make_vec_env(lambda: SudokuEnv(3,3), n_envs=4)

 # Train a PPO agent
# action_masks = get_action_masks(env)
# print(action_masks)
# print(action_masks)
# print("Shape is: ", action_masks.shape)
policy_kwargs = dict(
    normalize_images = True,
    #features_extractor_class=CustomCNN,
    #features_extractor_kwargs=dict(features_dim=128),
    )
model = MaskablePPO("MlpPolicy", env=env, verbose=1, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=25_000, use_masking=True)
model.save("sudoku_rl_agent")