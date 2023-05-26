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
import PIL
from settings import FRAME_RATE
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from torch import nn
import gym
import numpy as np

from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


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
    
class LightCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN policy that accepts 128-channel input.
    """

    def __init__(self, observation_space, features_dim=512, *args, **kwargs):
        super(LightCNNFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = th.nn.Sequential(
            nn.MaxPool2d(2, 2),
            DQNConvBlock(observation_space.shape[0], 16, 3, 2),
            nn.MaxPool2d(4),
            DQNConvBlock(16, 8, 4, 2),
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
        print(responses.shape)
        return self.fc(responses)
    
class ResNetFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__(observation_space, features_dim)
        self.n_frames = 16

        # pretrained resenet50
        self.resnet = th.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        self.resnet.fc = nn.Identity()
        for param in self.resnet.parameters():
            param.requires_grad = False
        # make last couple layers trainable
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.avgpool.parameters():
            param.requires_grad = True
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

        # add a new layer that outputs features_dim
        self.fc = nn.Sequential(
            nn.Linear(2048, features_dim),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_dim * self.n_frames, features_dim),
            nn.ReLU(),
        )

    def forward_channel_component(self, observations: Tensor, w, **kwargs) -> Tensor:
        responses = observations * w
        responses = responses.sum(dim=1, keepdim=True) / w.sum()
        return responses

    def forward(self, observations: Tensor, **kwargs) -> Tensor:
        frames = []
        for i in range(self.n_frames):
            start = i * 3
            end = start + 3
            features = self.resnet(observations[:, start:end, :, :])
            features = self.fc(features)
            features = features.unsqueeze(1)
            frames.append(features)
        responses = th.cat(frames, dim=1)
        responses = self.out(responses)

        return responses            
    

class SemanticSegFormerFeatureExtractor(BaseFeaturesExtractor):
        
        def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
            super().__init__(observation_space, features_dim)
            self.n_frames = 8
            self.feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
            self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
            self.out = nn.Sequential(
                nn.MaxPool2d(4),
                DQNConvBlock(150, 64, 1, 1),
                nn.Flatten(),
                nn.Linear(64*32*32, features_dim)
            )
            # to go from (batch, n_frames, embedding_dim) to (batch, 1, embedding_dim)
            self.out2 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.n_frames*features_dim, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: Tensor, **kwargs) -> Tensor:
            frames = []
            for i in range(self.n_frames):
                start = i*3
                end = start+3
                current_frame = observations[:, start:end, :, :]

                # resize to 1024x1024
                current_frame = th.nn.functional.interpolate(current_frame, size=(512, 512), mode='bilinear', align_corners=False)
                assert current_frame.min() >= 0 and current_frame.max() <= 1
                inputs = self.feature_extractor(current_frame, return_tensors="pt")
                inputs["pixel_values"] = inputs["pixel_values"].to(self.model.device)
                outputs = self.model(**inputs)
            
                responses = outputs.logits
                responses = self.out(responses)
                responses = responses.unsqueeze(1)
                frames.append(responses)
            
            responses = th.stack(frames, dim=1)
            responses = self.out2(responses)

            # vizualize the segmentation
            im = outputs.logits[0].detach().cpu().numpy()
            im = im.argmax(axis=0)
            im = 255*im / 150
            im = im.astype(np.uint8)
            im = PIL.Image.fromarray(im)
            im.save('obs.png')

            return responses
class MobileNetV2FeatureExtractor(BaseFeaturesExtractor):
    """
    CNN policy that accepts 128-channel input.
    """

    def __init__(self, observation_space, features_dim=512, num_heads=8, *args, **kwargs):
        super(MobileNetV2FeatureExtractor, self).__init__(observation_space, features_dim)
        self.n_frames = FRAME_RATE

        # load mobilenetv2
        self.mobilenet = th.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        for param in self.mobilenet.parameters():
            param.requires_grad = False
        # make last couple layers trainable
        for param in self.mobilenet.features[-1].parameters():
            param.requires_grad = True
        for param in self.mobilenet.features[-2].parameters():
            param.requires_grad = True
        for param in self.mobilenet.features[-3].parameters():
            param.requires_grad = True
            
        # add a new layer that outputs features_dim
        self.mobilenet_out = nn.Sequential(
            nn.Linear(1280, features_dim),
            nn.ReLU(),
        )
        self.mha = nn.MultiheadAttention(features_dim*self.n_frames, num_heads=num_heads)
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features_dim * self.n_frames, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Tensor, **kwargs) -> Tensor:
        frames = []
        for i in range(self.n_frames):
            start = i * 3
            end = start + 3
            features = self.mobilenet(observations[:, start:end, :, :])
            features = self.mobilenet_out(features)
            frames.append(features)
        responses = th.cat(frames, dim=1)
        responses = self.mha(responses, responses, responses)
        responses = self.out(responses[0])
        return responses
