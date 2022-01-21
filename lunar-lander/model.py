import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.Linear(state_size, 32)  # 64)
        self.output = nn.Linear(32, action_size)

    def forward(self, state):
        out = F.relu(self.fc(state))
        out = self.output(out)

        return out
