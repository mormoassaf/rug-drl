import torch as th
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import create_mlp
from torch.distributions import Categorical
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from torch import Tensor
from stable_baselines3.common.utils import constant_fn


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    CNN policy that accepts 128-channel input.
    """

    def __init__(self, observation_space, features_dim=512, *args, **kwargs):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(128, 32, kernel_size=8, stride=4),
            th.nn.ReLU(),
            th.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            th.nn.ReLU(),
            th.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            th.nn.ReLU(),
            th.nn.Flatten(),
            th.nn.Linear(25088, features_dim),
            th.nn.ReLU()
        )
    
    def forward(self, observations: Tensor) -> Tensor:
        return self.cnn(observations)
    
    
# policy that accepts 128-channel input
class CustomPolicy(BasePolicy):

    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 net_arch,
                 activation_fn,
                 ortho_init,
                 *args,
                 **kwargs):
        super(CustomPolicy, self).__init__(observation_space, action_space, lr_schedule)
        self.features_dim = 512
        self.features_extractor = CNNFeatureExtractor(observation_space, features_dim=512)
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init
        self._build(lr_schedule, net_arch, activation_fn, ortho_init)
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr_schedule(1), eps=1e-5)

    def _build(self, lr_schedule, net_arch, activation_fn, ortho_init):
        self.mlp_extractor = th.nn.Sequential(*create_mlp(self.features_dim, self.features_dim, net_arch, activation_fn))
        self.action_net = th.nn.Sequential(*create_mlp(self.features_dim, self.action_space.n, net_arch, activation_fn))
        self.value_net = th.nn.Sequential(*create_mlp(self.features_dim, 1, net_arch, activation_fn))

    def _predict(self, observation, deterministic=False):
        return super()._predict(observation, deterministic)
    
    def predict_values(self, observation: Tensor, deterministic: bool = False) -> Tensor:
        features = self.extract_features(observation)
        values = self.value_net(features)
        return values

    def evaluate_actions(self, observation: Tensor, action: Tensor) -> Tensor:
        features = self.extract_features(observation)
        action_logits = self.action_net(features)
        dist = Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        values = self.value_net(features)
        return values, action_log_probs, dist_entropy

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        action_logits = self.action_net(features)
        dist = Categorical(logits=action_logits)
        if deterministic:
            action = th.argmax(action_logits, dim=1)
        else:
            action = dist.sample()
        
        values = self.value_net(features)

        return action, values, dist.log_prob(action)
    
    def extract_features(self, obs):
        features = self.features_extractor(obs)
        features = self.mlp_extractor(features)
        return features

    def _build_lr_schedule(self, lr_schedule):
        if isinstance(lr_schedule, float):
            lr_schedule = constant_fn(lr_schedule)
        elif callable(lr_schedule):
            lr_schedule = lr_schedule
        else:
            raise ValueError(f"Invalid lr_schedule: {lr_schedule}. "
                            "Expected float or callable.")
        self.lr_schedule = lr_schedule