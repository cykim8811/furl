
import gym
import wandb
import furl.main
import torch
import numpy as np

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
