import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(torch.nn.Module):

    def __init__(self, input_size, output_size, hidden_size):
        super(QNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class QNetworkConv(torch.nn.Module):

    def __init__(self, in_channels: int, spatial_size: tuple, output_size: int):
        super(QNetworkConv, self).__init__()
        self.in_channels = in_channels
        self.spatial_size = spatial_size
        self.output_size = output_size
    
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)

        conv_out_size = self._get_conv_out_size()
        # output should be (batch, output_size)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, self.output_size)

    def _get_conv_out_size(self):
        x = torch.zeros(1, self.in_channels, *self.spatial_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.numel()
    
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)