import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Bernoulli

class Dasher(nn.Module):
    def __init__(self):
        super(Dasher, self).__init__()
        # expect input size: 400x400x1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=5, out_channels=5, kernel_size=7, stride=1)
        self.pool_half = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=188180, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        self.reward_decay = 0.8

        # state(frame of game)
        # self.states = []

        # gradient(history of decisions made upon sampling)
        self.log_probs = []
        # rewards(history of rewards check whether current move is good or bad)
        self.rewards = []

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = self.pool_half(x)
        x = x.view(x.numel())
        x = F.elu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        
        return x

    def conclude_loss(self):
        # lost decay calculation
        reward_so_far = 0
        # corrected reward
        corrected_reward = torch.empty(len(self.rewards))
        for idx, reward in enumerate(reversed(self.rewards)):
            reward_so_far = self.reward_decay * reward_so_far + reward
            corrected_reward[idx] = reward_so_far

        # normalized
        corrected_reward = (corrected_reward - corrected_reward.mean()) / corrected_reward.std()

        corrected_reward = corrected_reward.cuda()
        log_probs = torch.tensor(self.log_probs).cuda()

        final_loss = -log_probs * corrected_reward

        # reset for next game
        self.reset()

        return final_loss

    def reset(self):
        # self.states = []
        self.log_probs = []
        self.rewards = []

    def save_reward(self, r):
        self.rewards.append(r)

    def make_move(self, x):
        # self.states.append(x)
        probability = self.forward(x)
        probability = Bernoulli(probability)
        move = probability.sample()
        # self.log_probs.append(probability.log_prob(move))
        return move.item()

