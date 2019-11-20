import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------------------------------------------------------

class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, action_size)

        nn.init.xavier_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.fill_(0.0)  

    def forward(self, state):
        x = self.bn0(state)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))

#------------------------------------------------------------------------------

class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()

        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128 + action_size, 128)
        self.fc3 = nn.Linear(128, 1)

        nn.init.xavier_normal_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        nn.init.xavier_normal_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.fill_(0.0)  


    def forward(self, state, action):
        state = self.bn0(state)
        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


#------------------------------------------------------------------------------
