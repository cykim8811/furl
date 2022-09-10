
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class BreakoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(3136, 4), std=0.01)
        self.critic = layer_init(nn.Linear(3136, 1), std=1)

    def forward(self, x):
        batched = x.dim() == 4
        if not batched: x = x.unsqueeze(0)
        x = x.float() / 255
        x1 = self.conv(x)
        x2 = self.conv(x)
        x1 = x1.view(x1.size(0), -1)
        if not batched: x1 = x1.squeeze(0)
        x2 = x2.view(x2.size(0), -1)
        if not batched: x2 = x2.squeeze(0)
        return self.actor(x1), self.critic(x2)
