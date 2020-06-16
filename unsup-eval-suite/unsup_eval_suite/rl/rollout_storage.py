"""
RolloutStorage adapted from:
https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""

import torch

class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, frame_shape,
                 action_n, action_shape, aux_sz):
        self.frames = torch.zeros(num_steps+1, num_processes, *frame_shape)
        self.aux = None
        if aux_sz>0: self.aux = torch.zeros(num_steps+1, num_processes, aux_sz)
        self.masks = torch.ones(num_steps+1, num_processes, 1)
        if action_n is not None:  # discrete actions
            self.actions = torch.zeros(num_steps, num_processes, action_n)
            self.actions = self.actions.long()
        else:
            self.actions = torch.zeros(num_steps, num_processes, *action_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.step = 0

    def to(self, device):
        self.frames = self.frames.to(device)
        if self.aux is not None: self.aux = self.aux.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)

    def insert(self, frames, aux, masks, actions, rewards):
        self.frames[self.step+1].copy_(frames)
        if self.aux is not None: self.aux[self.step+1].copy_(aux)
        self.masks[self.step+1].copy_(masks)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.step = self.step + 1

    def clone_prev(self):
        prev_frame = self.frames[-1].clone()
        prev_aux = None if self.aux is None else self.aux[-1].clone()
        prev_masks = self.masks[-1].clone()
        return prev_frame, prev_aux, prev_masks
