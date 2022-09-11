
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

class BreakoutLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )
        self.lstm = nn.GRU(3136, 512, batch_first=True)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(512, 4), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, x, h):
        dims = x.shape[:-3]
        if len(dims) == 0:
            x = x.unsqueeze(0)
        x = x.view(-1, *x.shape[-3:])
        x = x.float() / 255
        x = self.conv(x)
        x = x.view(dims + (-1,))
        if len(dims) == 0:
            x = x.unsqueeze(0)
        if not h.is_contiguous():
            h = h.contiguous()
        x, h = self.lstm(x, h)
        return self.actor(x), self.critic(x), h

    def init_hidden(self, device="cpu"):
        return torch.zeros(1, 512).to(device)