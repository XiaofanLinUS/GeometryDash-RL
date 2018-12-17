import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

class Dasher(nn.Module):
    def __init__(self):
        super(Dasher, self).__init__()
        # expect input size: 400x400x1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=12, kernel_size=7, stride=1)
        self.pool_half = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=144*144*12, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        # state(frame of game)
        self.states = []
        # gradient(history of decisions made upon sampling)
        self.probs = []
        # rewards(history of rewards check whether current move is good or bad)
        self.rewards = []

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.pool_half(x)
        x = F.elu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        return x

    def makeMove(self, x):
        self.states.append(x)
        probability = self.forward(x)
        probability = Categorical(probability)
        move = probability.sample()
        self.probs.append(-probability.log_prob(move))
        return move.item()

