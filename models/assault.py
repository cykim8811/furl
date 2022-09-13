
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class AssaultModel(nn.Module):
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
        self.actor = layer_init(nn.Linear(3136, 7), std=0.01)
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


class AssaultICMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 32, 6, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(32 * 32, 7), std=0.01)
        self.critic = layer_init(nn.Linear(32 * 32, 1), std=1)

        self.icm_feature = nn.Sequential(
            layer_init(nn.Linear(32 * 32, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
        self.icm_inverse = layer_init(nn.Linear(1024, 7), std=0.01)
        self.icm_forward = nn.Sequential(
            layer_init(nn.Linear(512 + 7, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )


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

    def icm(self, s0, a, s1):
        s0 = s0.float() / 255
        s1 = s1.float() / 255
        s0 = self.conv(s0)
        s1 = self.conv(s1)
        s0 = s0.view(s0.size(0), -1)
        s1 = s1.view(s1.size(0), -1)
        s0 = self.icm_feature(s0)
        s1 = self.icm_feature(s1)
        a_onehot = F.one_hot(a, 7).float()
        s1_pred = self.icm_forward(torch.cat([s0, a_onehot], dim=1))
        action_pred = self.icm_inverse(torch.cat([s0, s1], dim=1))

        pred_error = (s1_pred - s1).pow(2).mean(dim=1)
        action_loss = F.cross_entropy(action_pred, a)
        return pred_error, action_loss