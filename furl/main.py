
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

import time
import os
import wandb

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

    def reserve(self, data_like: dict):
        if self.data is None:
            self.data = {key: torch.zeros((self.size, *value.shape), dtype=value.dtype, device=value.device) for key, value in data_like.items()}
        else:
            raise RuntimeError("Memory is already reserved.")

    def add(self, data: dict, force: bool = False):
        if self.data is None:
            self.reserve(data)
        
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
        self.logger = None
    
    def set_logger(self, logger): # TODO: Change logging implementation
        self.logger = logger
    
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

        if step_dict['done'] and self.rank == 0: # TODO: Change logging implementation
            if self.logger is not None:
                self.logger(sum(self.state.last_scores)/len(self.state.last_scores))
                #self.logger(self.state.last_scores[-1])

    def reset(self):
        self.state.reset()

class Trainer:
    def __init__(self, param, strategy):
        mp.set_start_method('spawn')
        self.param = {
            'lr': 0.0001,
            'update_interval': 100,
            'epochs': 3,
            'num_processes': 1,
        }
        self.param.update(param)
        self.strategy = strategy

        if 'wandb_project' in param:
            wandb.init(project=param['wandb_project'], config=param)
            if 'wandb_sweeps' in param:
                self.param.update(wandb.config)

    def update(model: nn.Module, optimizer: optim.Optimizer, memory: Memory, param: dict, strategy: Strategy):
        for epoch in range(param['epochs']):
            strategy.learn(model, memory, optimizer)
    
    def run_thread(rank, total_steps, total_time, model, state_class, strategy, param, queue):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('gloo', rank=rank, world_size=param['num_processes'])

        episode = Episode(rank, model, state_class(), strategy, param)
        episode.reset()
        if rank == 0:
            optimizer = optim.Adam(model.parameters(), lr=param['lr'], eps=1e-5)
            episode.set_logger(logger=lambda score: queue.put(score))
        start_time = time.time()
        step = 0
        while (total_steps and step < total_steps) or (total_time and (time.time() - start_time) < total_time) or (not total_steps and not total_time):
            episode.step()
            if (step+1) % param['update_interval'] == 0:
                if strategy.include_last:
                    last_state = episode.state.to_tensor()
                    if type(last_state) is torch.Tensor: last_state = {'state': last_state}
                    dict_to_device(last_state, next(model.parameters()).device)
                    episode.memory.add(last_state, force=True)

                if rank == 0:
                    stacked_memory = Memory(param['num_processes'], next(model.parameters()).device)
                    stacked_memory.reserve(episode.memory.data)
                    for key in episode.memory.data:
                        dist.gather(episode.memory[key], [t.squeeze(0) for t in stacked_memory[key].split(1, dim=0)], dst=0)
                    stacked_memory.index = param['num_processes']
                    stacked_memory.swapaxes(0, 1)
                    Trainer.update(model, optimizer, stacked_memory, param, strategy)
                else:
                    for key in episode.memory.data:
                        dist.gather(episode.memory[key], dst=0)

                episode.memory.clear()
            step += 1
            dist.barrier()

    def fit(self, model, state_class, total_steps=None, total_time=None):
        if 'wandb_project' in self.param:
            wandb.watch(model)
        processes = []
        model.share_memory()
        queue = mp.Queue()
        for rank in range(self.param['num_processes']):
            process = mp.Process(target=Trainer.run_thread, args=(rank, total_steps, total_time, model, state_class, self.strategy, self.param, queue))
            process.start()
            processes.append(process)

        start_time = time.time()
        step = 0
        while (total_steps and step < total_steps) or (total_time and (time.time() - start_time) < total_time) or (not total_steps and not total_time):
            if queue.empty():
                time.sleep(1)
                continue
            score = queue.get()
            wandb.log({'score': score})
            print(f'Step {step:7d}: {score:6.2f}')
            step += 1
        
        for process in processes: process.join()
