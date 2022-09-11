
import gym
import furl

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cartpole import CartpoleModel
from models.breakout import BreakoutModel, BreakoutLSTMModel

import wandb

from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv

def make_breakout_state():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)
    return furl.gym.GymState(env)

def make_breakout_nostack_state():
    env = gym.make("BreakoutNoFrameskip-v4")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 1)
    return furl.gym.GymLSTMState(env)

def make_cartpole_state():
    env = gym.make("CartPole-v1")
    return furl.gym.GymState(env)

if __name__ == "__main__":
    model = BreakoutLSTMModel().cuda()
    strategy_param = {
        'gamma': 0.99,
        'normalize_advantage': True,
        'use_gae': True,
        'gae_lambda': 0.95,
        'clip_eps': 0.15,
    }
    param={
        'lr': 0.002,
        'update_interval': 384,
        'epochs': 4,
        'num_processes': 4,
        'wandb_project': 'breakout-ppo'
    }
    strategy = furl.algorithms.PPOLSTMStrategy(**strategy_param)
    trainer = furl.Trainer(param, strategy)
    trainer.fit(model, make_breakout_nostack_state, total_steps=None)
