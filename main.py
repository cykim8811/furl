
import gym
import furl

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

from models.cartpole import CartpoleModel
from models.breakout import BreakoutModel, BreakoutICMModel
from models.assault import AssaultICMModel
from models.gym import AtariICMModel, AtariModel

import wandb

if __name__ == "__main__":
    model = AtariModel(7).cuda()
    strategy_param = {
        'gamma': 0.99,
        'normalize_advantage': True,
        'use_gae': True,
        'gae_lambda': 0.95,
        'clip_eps': 0.1,
    }
    param={
        'lr': 0.001,
        'update_interval': 256,
        'epochs': 4,
        'num_processes': 2,
        'wandb_project': 'assault-icm',
        'forward_loss_coef': 3.0,
        'inverse_loss_coef': 0.3,
    }
    strategy = furl.algorithms.PPOStrategy(**strategy_param)
    trainer = furl.Trainer(param, strategy)
    trainer.fit(model, furl.gym.make_atari_state("AssaultNoFrameskip-v4"), total_steps=None)
