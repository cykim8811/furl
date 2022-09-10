
import torch
import torch.nn as nn
import torch.nn.functional as F


class CartpoleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 128)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return self.actor(x), self.critic(x)
