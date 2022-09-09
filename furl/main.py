
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time

from typing import Dict, List, Tuple

def dict_to_device(data: Dict[str, torch.Tensor], device: torch.device):
    for key in data:
        data[key] = data[key].to(device)

class Memory:
    def __init__(self, size: int, device: torch.device = torch.device('cpu')):
        self.data = None
        self.size = size
        self.index = 0
        self.device = device

    def __getitem__(self, key: str):
        return self.data[key]
    
    def __setitem__(self, key: str, value: torch.Tensor):
        if self.data is None:
            self.data = {}
        self.data[key] = value

    def add(self, data: dict, force: bool = False):
        if self.data is None:
            self.data = {key: torch.zeros((self.size, *value.shape), dtype=value.dtype, device=value.device) for key, value in data.items()}
        
        if self.index >= self.size:
            if force:
                for key in data:
                    self.data[key] = torch.cat([self.data[key], data[key].unsqueeze(0)])
                self.index += 1
                return
            else:
                raise IndexError(f"Memory is full. Use force=True to add more data. Current size: {self.size}")
        
        for key in data:
            self.data[key][self.index] = data[key]

        self.index += 1
    
    def stack(data: List['Memory'])->'Memory':
        new_memory = Memory(len(data), data[0][list(data[0].data.keys())[0]].device)
        for key in data[0].data:
            new_memory[key] = torch.stack([memory[key] for memory in data])
        new_memory.size = new_memory[key].shape[0]
        new_memory.index = new_memory.size
        return new_memory
    
    def permute(self, *args):
        for key in self.data:
            self.data[key] = self.data[key].permute(*args)
        return self

    def swapaxes(self, axis1, axis2):
        for key in self.data:
            self.data[key] = self.data[key].swapaxes(axis1, axis2)
        return self
    
    def clear(self):
        self.data = None
        self.index = 0
    
    def to(self, device: torch.device or str):
        self.device = device if type(device) is torch.device else torch.device(device)
        if self.data is not None:
            for key in self.data:
                self.data[key] = self.data[key].to(device)
        return self
    
    def cpu(self):
        return self.to(torch.device('cpu'))

    def cuda(self):
        return self.to(torch.device('cuda'))

class State:
    def step(self, action: int)->Dict[str, torch.Tensor]:
        raise NotImplementedError

    def reset(self)->None:
        raise NotImplementedError

    # Convert state to tensor. returns a tensor or tuple of tensors
    def to_tensor(self)->torch.Tensor or Tuple['torch.Tensor']:
        raise NotImplementedError


class Strategy:
    def __init__(self):
        self.include_last = False

    def act(self, model: nn.Module, state_tensor: torch.Tensor)->Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def learn(self, model: nn.Module, memory: Memory, optimizer: optim.Optimizer):
        raise NotImplementedError

class Processor:
    def __call__(self, memory: Memory, model: nn.Module):
        raise NotImplementedError

class Episode:
    def __init__(self, rank: int, model: nn.Module, state: State, strategy: Strategy, param: dict):
        self.memory = Memory(param['update_interval']).to(next(model.parameters()).device)
        self.model = model
        self.state = state
        self.strategy = strategy
        self.param = param
        self.rank = rank
    
    def step(self):
        model_device = next(self.model.parameters()).device
        state_tensor = self.state.to_tensor()
        if type(state_tensor) is torch.Tensor:
            state_tensor = state_tensor.to(model_device)
            act_dict = self.strategy.act(self.model, state_tensor)
            state_tensor = {'state': state_tensor}
        else:
            dict_to_device(state_tensor, model_device)
            act_dict = self.strategy.act(self.model, state_tensor)
        dict_to_device(act_dict, model_device)
        step_dict = self.state.step(act_dict['action'].item())
        dict_to_device(step_dict, model_device)
        self.memory.add({**state_tensor, **act_dict, **step_dict})

        if self.rank == 0 and step_dict['done']: self.state.log() # TODO: Change logging implementation

    def reset(self):
        self.state.reset()
        

class Trainer:
    def __init__(self, param, strategy):
        self.param = {
            'lr': 0.0001,
            'update_interval': 100,
            'epochs': 3,
            'num_processes': 1,
        }
        self.param.update(param)
        self.strategy = strategy
    
    def update(self, model: nn.Module, optimizer: optim.Optimizer, memory: Memory):
        for epoch in range(self.param['epochs']):
            self.strategy.learn(model, memory, optimizer)

    def fit(self, model, state_class, total_steps=None, total_time=None):
        optimizer = optim.Adam(model.parameters(), lr=self.param['lr'], eps=1e-5)
        episodes = [Episode(i, model, state_class(), self.strategy, self.param) for i in range(self.param['num_processes'])]
        for episode in episodes: episode.reset()
        step = 0
        start_time = time.time()
        while (total_steps and step < total_steps) or (total_time and (time.time() - start_time) < total_time) or (not total_steps and not total_time):
            for episode in episodes: episode.step()
            if (step+1) % self.param['update_interval'] == 0:
                if self.strategy.include_last:
                    for episode in episodes:
                        last_state = episode.state.to_tensor()
                        if type(last_state) is torch.Tensor: last_state = {'state': last_state}
                        dict_to_device(last_state, next(model.parameters()).device)
                        episode.memory.add(last_state, force=True)
                stacked_memory = Memory.stack([episode.memory for episode in episodes])
                stacked_memory.swapaxes(0, 1)
                self.update(model, optimizer, stacked_memory)
                for episode in episodes:
                    episode.memory.clear()
            step += 1
