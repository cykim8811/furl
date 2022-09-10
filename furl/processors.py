
import furl.main

import torch
import numpy as np

from typing import List, Tuple

class Flatten(furl.main.Processor):
    def __call__(self, memory, model):
        additional_dim = len(memory['reward'].shape)
        for key in memory.data:
            print('\nkey', key)
            print('from', memory[key].shape)
            memory[key] = memory[key].view(-1, *memory[key].shape[additional_dim:])
            print('to  ', memory[key].shape)


class GAE(furl.main.Processor):
    def __init__(self, gamma: float, lam: float, delta_key: str = 'delta', done_key: str = 'done'):
        self.gamma = gamma
        self.lam = lam
        self.delta_key = delta_key
        self.done_key = done_key

    def __call__(self, memory, model):
        gae = 0
        advantage = torch.zeros_like(memory[self.delta_key])
        for i in reversed(range(len(memory[self.delta_key]))):
            gae = memory[self.delta_key][i] + self.gamma * self.lam * gae * (1 - memory[self.done_key][i].float())
            advantage[i] = gae
        return advantage
    

class Delta(furl.main.Processor):
    def __init__(self, gamma: float=0.99, current_value: str='value', reward: str='reward', next_value: str='next_value', done: str='done'):
        self.gamma = gamma

        self.reward = reward
        self.next_value = next_value
        self.done = done
        self.current_value = current_value

    def __call__(self, memory, model):
        return memory[self.reward] + self.gamma * memory[self.next_value].detach() * (1 - memory[self.done].float()) - memory[self.current_value].squeeze(-1)

class Normalize(furl.main.Processor):
    def __init__(self, key: str, eps: float=1e-8):
        self.key = key
        self.eps = eps

    def __call__(self, memory, model):
        normalized = (memory[self.key] - memory[self.key].mean())
        if (memory[self.key].std() < self.eps).any():
            print("Warning: {} not being normalized. Standard deviation is too small".format(self.key))
        else:
            normalized = memory[self.key] / (memory[self.key].std() + self.eps)
        return normalized

class Minibatch(furl.main.Processor):
    def __init__(self,target: str, batch_size: int, shuffle: bool=True):
        self.target = target
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, memory, model):
        target = memory[self.target]
        indices = torch.randperm(target.shape[0]) if self.shuffle else torch.arange(target.shape[0])
        return torch.stack(target[indices].split(self.batch_size))
        
