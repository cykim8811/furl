
import furl.main
from furl.processors import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class A2CStrategy:
    def __init__(self, gamma: float = 0.99, normalize_advantage: bool = True, norm_eps: float = 1e-8, use_gae: bool = False, gae_lambda: float = 0.95):
        self.include_last = True
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.norm_eps = 1e-8
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda

    def act(self, model, state):
        state_tensor = state.to_tensor().to(next(model.parameters()).device)
        logits, value = model(state_tensor)
        logprob = F.log_softmax(logits, dim=-1)
        action = logprob.exp().multinomial(1)[0].long()
        return {
            'action': action,
            'logprob': logprob,
            'value': value.squeeze(0),
        }
    
    def learn(self, model, memory, optimizer):
        memory['current_logit'], memory['current_value'] = model(memory['state'][:-1])
        memory['current_value'] = memory['current_value'].squeeze(-1)

        t_logprob = F.log_softmax(memory['current_logit'], dim=-1)
        memory['logprob_action'] = t_logprob.gather(-1, memory['action'].unsqueeze(-1)).squeeze(-1)

        last_value = model(memory['state'][-1])[1].squeeze(-1)
        memory['next_value'] = torch.cat([memory['value'][1:], last_value.unsqueeze(0)])


        memory['delta'] = furl.processors.Delta(gamma=self.gamma)(memory, model)
        memory['advantage'] = memory['delta']
        if self.use_gae:
            memory['advantage'] = furl.processors.GAE(self.gamma, self.gae_lambda)(memory, model)
            
        memory['return'] = (memory['advantage'] + memory['current_value'].squeeze(-1)).detach()

        if self.normalize_advantage:
            memory['advantage'] = furl.processors.Normalize('advantage', self.norm_eps)(memory, model)


        actor_loss = -(memory['logprob_action'] * memory['advantage'].detach()).mean()
        critic_loss = (memory['return'] - memory['current_value'].squeeze(-1)).pow(2).mean()
        loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class PPOStrategy:
    def __init__(self, gamma: float = 0.99, normalize_advantage: bool = True, norm_eps: float = 1e-8, use_gae: bool = False, gae_lambda: float = 0.95, clip_eps: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5, max_grad_norm: float = 0.5):
        self.include_last = True
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.norm_eps = 1e-8
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def act(self, model, state):
        state_tensor = state.to_tensor().to(next(model.parameters()).device)
        with torch.no_grad():
            logits, value = model(state_tensor)
            logprob = F.log_softmax(logits, dim=-1)
            action = logprob.exp().multinomial(1)[0].long()
            return {
                'action': action,
                'logprob': logprob,
                'value': value.squeeze(0),
            }
    
    def learn(self, model, memory, optimizer):
        additional_dims = memory['reward'].shape
        memory['value'] = memory['value'].view(*additional_dims)
        memory['old_logprob_action'] = memory['logprob'].gather(-1, memory['action'].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            last_state_feed = memory['state'][-1].flatten(0, len(additional_dims)-2) if len(additional_dims) > 1 else memory['state'][-1].unsqueeze(0)
            last_value = model(last_state_feed)[1].squeeze(-1).reshape(1, *additional_dims[1:])
            memory['next_value'] = torch.cat([memory['value'][1:], last_value])

            memory['delta'] = furl.processors.Delta(gamma=self.gamma)(memory, model)
            memory['advantage'] = memory['delta']
            if self.use_gae:
                memory['advantage'] = furl.processors.GAE(self.gamma, self.gae_lambda)(memory, model)
                
            memory['return'] = (memory['advantage'] + memory['value'].squeeze(-1)).detach()

            if self.normalize_advantage:
                memory['advantage'] = furl.processors.Normalize('advantage', self.norm_eps)(memory, model)


        state_feed = memory['state'][:-1].flatten(0, len(additional_dims)-1)
        hidden_feed = memory['hidden'][:-1].flatten(0, len(additional_dims)-1)

        memory['current_logit'], memory['current_value'] = model(state_feed, hidden_feed[0])
        
        memory['current_logit'] = memory['current_logit'].reshape(*additional_dims, -1)
        memory['current_value'] = memory['current_value'].squeeze(-1).reshape(*additional_dims)

        t_logprob = F.log_softmax(memory['current_logit'], dim=-1)
        memory['logprob_action'] = t_logprob.gather(-1, memory['action'].unsqueeze(-1)).squeeze(-1)

        ratio = torch.exp(memory['logprob_action'] - memory['old_logprob_action'].detach())    
        actor_loss = -torch.min(
            ratio * memory['advantage'].detach(),
            torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * memory['advantage'].detach()
        ).mean()
        critic_loss = (memory['return'] - memory['current_value'].squeeze(-1)).pow(2).mean()
        entropy = -(t_logprob.exp() * t_logprob).sum(-1).mean()
        loss = actor_loss + critic_loss * self.vf_coef - entropy * self.ent_coef
        
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None: nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()


class PPOLSTMStrategy:
    def __init__(self, gamma: float = 0.99, normalize_advantage: bool = True, norm_eps: float = 1e-8, use_gae: bool = False, gae_lambda: float = 0.95, clip_eps: float = 0.2, ent_coef: float = 0.01, vf_coef: float = 0.5, max_grad_norm: float = 0.5):
        self.include_last = True
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.norm_eps = 1e-8
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

    def act(self, model, state):
        device = next(model.parameters()).device
        state_tensor = state.to_tensor()
        state_x = state_tensor['state'].to(device)
        state_h = state_tensor['hidden'].to(device)
        if state.hidden is None:
            state.hidden = model.init_hidden(device=device)
        initial_hidden = state.hidden
        with torch.no_grad():
            logits, value, hidden = model(state_x, state_h)
            logprob = F.log_softmax(logits, dim=-1)
            action = logprob.exp().multinomial(1)[0].long()
            return {
                'action': action,
                'logprob': logprob,
                'value': value.squeeze(0),
                'initial_hidden': initial_hidden,
            }
    
    def learn(self, model, memory, optimizer):
        additional_dims = memory['reward'].shape
        memory['value'] = memory['value'].view(*additional_dims)
        memory['old_logprob_action'] = memory['logprob'].gather(-1, memory['action'].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            last_x_feed = memory['state'][-1].flatten(0, len(additional_dims)-2) if len(additional_dims) > 1 else memory['state'][-1].unsqueeze(0)
            last_h_feed = memory['hidden'][-1].flatten(0, len(additional_dims)-2) if len(additional_dims) > 1 else memory['state'][-1].unsqueeze(0)
            last_value = model(last_x_feed.unsqueeze(1), last_h_feed.swapaxes(0, 1))[1].squeeze(-1).reshape(1, *additional_dims[1:])
            memory['next_value'] = torch.cat([memory['value'][1:], last_value])

            memory['delta'] = furl.processors.Delta(gamma=self.gamma)(memory, model)
            memory['advantage'] = memory['delta']
            if self.use_gae:
                memory['advantage'] = furl.processors.GAE(self.gamma, self.gae_lambda)(memory, model)
                
            memory['return'] = (memory['advantage'] + memory['value'].squeeze(-1)).detach()

            if self.normalize_advantage:
                memory['advantage'] = furl.processors.Normalize('advantage', self.norm_eps)(memory, model)


        x_feed = memory['state'][:-1].flatten(0, len(additional_dims)-1)
        h_feed = memory['hidden'][:-1].flatten(0, len(additional_dims)-1)

        dones = memory['done'].flatten(0, len(additional_dims)-1)
        
        current_logits = []
        current_values = []
        last_idx = 0

        # TODO: Fix this
        for i in range(dones.shape[0]):
            if dones[i]:
                current_logit, current_value, _ = model(x_feed[last_idx:i+1], h_feed[last_idx])
                current_logits.append(current_logit)
                current_values.append(current_value)

                last_idx = i+1

        if not dones[-1]:
            current_logit, current_value, _ = model(x_feed[last_idx:], h_feed[last_idx])
            current_logits.append(current_logit)
            current_values.append(current_value)
        
        
        memory['current_logit'] = torch.cat(current_logits)
        memory['current_value'] = torch.cat(current_values)
        
        memory['current_logit'] = memory['current_logit'].reshape(*additional_dims, -1)
        memory['current_value'] = memory['current_value'].squeeze(-1).reshape(*additional_dims)

        t_logprob = F.log_softmax(memory['current_logit'], dim=-1)
        memory['logprob_action'] = t_logprob.gather(-1, memory['action']).squeeze(-1)

        ratio = torch.exp(memory['logprob_action'] - memory['old_logprob_action'].detach().squeeze(-1))
        actor_loss = -torch.min(
            ratio * memory['advantage'].detach(),
            torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * memory['advantage'].detach()
        ).mean()
        critic_loss = (memory['return'] - memory['current_value'].squeeze(-1)).pow(2).mean()
        entropy = -(t_logprob.exp() * t_logprob).sum(-1).mean()
        loss = actor_loss + critic_loss * self.vf_coef - entropy * self.ent_coef
        
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None: nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()
