"""
Functionality for collecting experience from env.
"""

import time

import numpy as np
import torch

import gym
from .baselines_common.vec_env import VecEnvWrapper
from .baselines_common.vec_env.dummy_vec_env import DummyVecEnv
from .baselines_common.vec_env.shmem_vec_env import ShmemVecEnv

import gym_bullet_aux


def to_torch(data, device):
    return torch.from_numpy(data).float().to(device)


def to_uint8(data):
    return (data*255).astype(np.uint8)


def from_uint8(data):
    return data.astype(np.float32)/255.0


def get_act_sz(action_space):
    if isinstance(action_space, gym.spaces.Discrete):
        act_sz = action_space.n
    else:
        act_sz = action_space.shape[0]
    return act_sz


def make_aux_action(bsz, action_space, device):
    if isinstance(action_space, gym.spaces.Discrete):
        nan_action = np.nan*torch.ones(bsz, 1).long()
    else:
        nan_action = np.nan*torch.ones(bsz, *action_space.shape).to(device)
    return nan_action


def aux_from_infos(infos, device):
    assert('aux' in infos[0].keys())
    aux = []
    for info in infos: aux.append(torch.from_numpy(info['aux']))
    return torch.stack(aux, dim=0).to(device)


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    return _thunk


def make_vec_envs(env_name, env_args, seed, num_envs, device):
    if env_name.startswith(('Reacher','Franka','BlockOnIncline')):
        num_envs_types = 6  # we have 6 types of rearrange envs
        num_envs_for_mod = 6 if env_name.startswith('BlockOnIncline') else 4
        any_env_skin = env_args.any_env_skin
        envs = []
        for i in range(num_envs):
            v = i%num_envs_for_mod if any_env_skin else env_args.env_skin_id
            if any_env_skin>1: v += (any_env_skin-1)*num_envs_types
            full_env_name = env_name+'-v'+str(v)
            envs.append(make_env(full_env_name, seed+i, i))
    else:
        envs = [make_env(env_name, seed, i) for i in range(num_envs)]
    if len(envs) > 1:
        envs = ShmemVecEnv(envs) # removed context='fork'
    else:
        envs = DummyVecEnv(envs)
    envs = PyTorchVecEnv(envs, device)
    return envs


# This class was adapted from
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class PyTorchVecEnv(VecEnvWrapper):
    def __init__(self, venv, device, squeeze_actions=False):
        super(PyTorchVecEnv, self).__init__(venv)
        self.device = device
        self.squeeze_actions = squeeze_actions

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            actions = actions.squeeze(1)  # squeeze dim for discrete actions
        if self.squeeze_actions: actions = actions.squeeze()
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info
