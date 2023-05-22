from stable_baselines3.common.type_aliases import Schedule
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp
from torch.distributions import Categorical
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from torch import Tensor
from stable_baselines3.common.utils import constant_fn
import torch.nn.init as init

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch import nn
import gym


class DQNConvBlock(th.nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 kernel_size: int, 
                 stride: int = 1, 
                 dropout: float = 0.2):
        super(DQNConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x

class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN policy that accepts 128-channel input.
    """

    def __init__(self, observation_space, features_dim=512, *args, **kwargs):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = th.nn.Sequential(
            DQNConvBlock(observation_space.shape[0], 32, 8, 4),
            DQNConvBlock(32, 64, 4, 2),
            DQNConvBlock(64, 64, 3, 1),
            DQNConvBlock(64, 64, 3, 1),
        )
        lin_size = self._compute_linear_input_size(observation_space)
        self.fc = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(lin_size, features_dim),
            th.nn.ReLU(),
        )

    def _compute_linear_input_size(self, observation_space: gym.spaces.Box) -> int:
        dummy = th.zeros(observation_space.shape).unsqueeze(0)
        dummy = self.cnn(dummy)
        return dummy.view(dummy.size(0), -1).size(1)
    
    def forward(self, observations: Tensor, **kwargs) -> Tensor:
        responses = self.cnn(observations)
        return self.fc(responses)
    
