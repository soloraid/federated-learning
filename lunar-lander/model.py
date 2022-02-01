import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.output = nn.Linear(128, action_size)

    def forward(self, state):
        out = F.relu(self.fc1(state))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.output(out)

        return out

    # def __init__(self, state_size, action_size, seed):
    #     super(QNetwork, self).__init__()
    #     self.seed = torch.manual_seed(seed)
    #     self.fc1 = nn.Linear(state_size, 32)
    #     self.output = nn.Linear(32, action_size)
    #
    # def forward(self, state):
    #     out = F.relu(self.fc1(state))
    #     out = self.output(out)
    #
    #     return out
