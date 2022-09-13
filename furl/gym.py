
import gym
import wandb
import furl.main
import torch
import numpy as np

from stable_baselines3.common.atari_wrappers import ClipRewardEnv, EpisodicLifeEnv, FireResetEnv, MaxAndSkipEnv, NoopResetEnv

class GymState(furl.main.State):
    def __init__(self, env: gym.Env or str):
        self.env = gym.make(env) if type(env) is str else env
        self.obs = None
        self.score = 0

        self.last_scores = []  # Temporary implementation for score logging
    
    def step(self, action):
        self.obs, reward, done, info = self.env.step(action)
        self.score += reward
        if done:
            self.reset()
        return {
            'reward': torch.tensor(reward),
            'done': torch.tensor(done).float(),
        }
    
    def reset(self):
        self.obs = self.env.reset()
        done = False

        self.last_scores.append(self.score) # Temporary implementation for score logging

        self.score = 0
        
    def to_tensor(self):
        return torch.tensor(np.array(self.obs), dtype=torch.float)

class AtariState(GymState):
    def __init__(self, env_name):
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        super().__init__(env)

def make_atari_state(env_name):
    return lambda: AtariState(env_name)
    
class AssaultState(GymState):
    def __init__(self, env_name):
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        super().__init__(env)

def make_assault_state():
    return lambda: AssaultState("AssaultNoFrameskip-v4")