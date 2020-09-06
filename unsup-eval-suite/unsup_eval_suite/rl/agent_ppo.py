"""
PPO RL Learner interface: AgentPPO.
"""

from collections import deque
from datetime import datetime
import logging
import os
import time

import numpy as np
import torch

from .ppo_torch.a2c_ppo_acktr import algo
from .ppo_torch.a2c_ppo_acktr.model import CNNBase64, MLPBase, MLPBaseLongTwin
from .ppo_torch.a2c_ppo_acktr.model import Policy as PolicyPPO
from .ppo_torch.a2c_ppo_acktr.utils import update_linear_schedule

from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from .rollout_storage import RolloutStorage
from .agent_random import AgentRandom, to_onehot


class AgentPPO(AgentRandom):
    def __init__(self, envs, latent_sz, latent_hist, aux_sz,
                 num_envs_per_worker, max_replay_rollouts, args, logfile):
        super(AgentPPO, self).__init__(
            envs, latent_sz, latent_hist, aux_sz,
            num_envs_per_worker, max_replay_rollouts, args, logfile)
        logging.info('Constructed AgentPPO')
        self.rl_lr = 1e-4  # high lr because rwds are normed
        self.max_epochs = args.max_epochs
        self.decay_f = lambda epoch, max_epochs: 1.0 - (epoch-1.0)/max_epochs
        self.mdp_gamma = 0.99
        self.clip_param = 0.2
        self.ppo_epoch = 3; self.num_mini_batch = 8
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.optimizer_epsilon = 1e-5
        self.use_clipped_value_loss = True  # key parameter absent from RLLib
        self.use_gae = False #True
        self.gae_lambda = 0.95
        self.rank = args.rank
        # Create an instance of PPO policy and learning agent.
        # PPO will be learning from low-dim observations for now.
        base = MLPBase; base_kwargs={'recurrent': None}
        if 'CartPole' not in args.env_name and 'Pendulum' not in args.env_name:
            base = MLPBase; hsz = 256  # MLPBaseLongTwin
            base_kwargs['hidden_size'] = hsz
            #base_kwargs['num_act_outputs'] = hsz  # TODO: clarify this
            self.ppo_epoch = args.unsup_num_sub_epochs; self.num_mini_batch = 4
        rl_state_sz = self.aux_sz if self.latent_sz is None else self.latent_sz
        self.policy = PolicyPPO((rl_state_sz,), envs.action_space,
                                base=base, base_kwargs=base_kwargs)
        self.policy.to(self.device)
        self.learner = algo.PPO(
            self.policy, self.clip_param, self.ppo_epoch,
            self.num_mini_batch, self.value_loss_coef, self.entropy_coef,
            lr=self.rl_lr, eps=self.optimizer_epsilon,
            max_grad_norm=self.max_grad_norm,
            use_clipped_value_loss=self.use_clipped_value_loss)
        if self.latent_sz is not None:
            self.prev_convfeat_seq = deque(maxlen=self.latent_hist-1)
            self.prev_act_seq = deque(maxlen=self.latent_hist-1)
            self.prev_done_seq = deque(maxlen=self.latent_hist-1)

    def make_rollouts(self, envs, rollout_len):
        rlts = RolloutStoragePPO(
            rollout_len, self.num_envs_per_worker, self.frames_shape,
            self.action_n, self.action_shape, self.aux_sz, self.latent_sz)
        rlts.to(self.device)
        rlts.frames[0] = self.prev_frame; rlts.aux[0] = self.prev_aux
        if self.prev_masks is not None: rlts.masks[0] = self.prev_masks
        if self.prev_latent is not None: rlts.latents[0] = self.prev_latent
        return rlts

    def rollout_step(self, envs, step, rlts, unsup):
        prev_obs = rlts.aux[step] if self.latent_sz is None else rlts.latents[step]
        prev_masks = rlts.masks[step]  # t is dim0
        with torch.no_grad():
            res = self.policy.act(prev_obs, None, prev_masks)
        value, act, action_log_prob, recurrent_hidden_states = res
        next_frame, next_aux, masks, reward, done = self.make_env_step(envs, act)
        # PPO's actor returns unnormed actions for contunuous domains already.
        # So here we just need to compute onehot repr for discrete actions.
        action = act if self.action_n is None else to_onehot(act, self.action_n)
        next_latents = None
        if self.latent_sz is not None:
            assert(unsup is not None)
            next_latents, next_convfeats = self.get_latent_code(
                unsup, next_frame, self.prev_convfeat_seq,
                self.prev_act_seq, self.prev_done_seq)
            self.prev_convfeat_seq.append(next_convfeats)
        rlts.insert(next_frame, next_aux, masks, action, reward,
                    action_log_prob, value, act, next_latents)
        if self.latent_sz is not None:
            self.prev_act_seq.append(action)
            self.prev_done_seq.append(done)  # done is not a tensor here

    def get_latent_code(self, unsup, x_T,
                        prev_convfeat_seq, prev_act_seq, prev_done_seq):
        assert((type(x_T) == torch.Tensor) and (x_T.dim() == 4))
        assert(len(prev_convfeat_seq)==len(prev_act_seq)==len(prev_done_seq))
        batch_size, chnls, data_h, data_w = x_T.size()
        # Dummy current action (we don't yet know what the policy will choose).
        a_1toT = torch.zeros(
            [batch_size, 1, *self.action_shape],device=self.device)
        x_T_feats = unsup.conv_stack(x_T.unsqueeze(dim=1))
        x_1toT_feats = x_T_feats  # already sequential
        if len(prev_convfeat_seq)>0:  # extract history information.
            done_accum = torch.tensor(
                prev_done_seq, device=self.device).float().transpose(0,1)
            done_accum = done_accum.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
            not_done = torch.abs(1-done_accum.clamp(0,1))
            not_done = not_done.reshape(batch_size, -1, 1)
            prev_act = torch.cat(list(prev_act_seq), dim=1)
            prev_act = prev_act.reshape(batch_size, not_done.size(1), -1)
            prev_act = prev_act*not_done
            a_1toT = torch.cat([prev_act, a_1toT], dim=1)
            prev_convfeat = torch.cat(list(prev_convfeat_seq), dim=1)
            prev_convfeat = prev_convfeat.reshape(batch_size, not_done.size(1), -1)
            x_1toT_feats = torch.cat([prev_convfeat*not_done, x_1toT_feats], dim=1)
        # Pad obs, acts if hist is short.
        num_pads = self.latent_hist - x_1toT_feats.size(1)
        if num_pads > 0:  # pad timesteps with 0th frame, noop action
            x_feats_pads = x_1toT_feats[:,0:1,:].repeat(1,num_pads,1)
            x_1toT_feats = torch.cat([x_feats_pads, x_1toT_feats], dim=1)
            # TODO: make sure this is noop in all envs
            act_pads = torch.zeros(
                [batch_size, num_pads, a_1toT.size(-1)], device=self.device)
            a_1toT = torch.cat([act_pads, a_1toT], dim=1)
        assert(x_1toT_feats.size(1) == a_1toT.size(1) == self.latent_hist)
        # Get latent code from unsupervised learner (unsup detaches, no grads)
        latent_code = unsup.latent_code(None, a_1toT, x_1toT_feats)
        return latent_code.detach(), x_T_feats.detach()  # detach to be sure

    def process_rollouts(self, epoch, rlts, unsup):
        with torch.no_grad():
            prev_obs = rlts.aux[-1] if self.latent_sz is None else rlts.latents[-1]
            next_value, _, _, _ = self.policy.act(
                prev_obs,
                None,  # no recurrence
                rlts.masks[-1])
        rlts.compute_returns(
            next_value, self.use_gae, self.mdp_gamma, self.gae_lambda)
        advantages = rlts.returns[:-1] - rlts.value_preds[:-1]
        advantages = (advantages - advantages.mean())/(advantages.std() + 1e-5)
        gen_fn = rlts.feed_forward_generator
        update_linear_schedule(  # update optimizer lr
            self.learner.optimizer, epoch, self.max_epochs, self.rl_lr)
        self.learner.frac_train_remaining = self.decay_f(epoch, self.max_epochs)
        return advantages, gen_fn

    def init_prev_obs(self, prev_frame, prev_aux, prev_masks=None, unsup=None):
        self.prev_frame = prev_frame
        self.prev_aux = prev_aux
        if prev_masks is not None: self.prev_masks = prev_masks
        if self.latent_sz is not None:
            assert(unsup is not None)
            latent, convfeats = self.get_latent_code(
                unsup, prev_frame, self.prev_convfeat_seq,
                self.prev_act_seq, self.prev_done_seq)
            self.prev_convfeat_seq.append(convfeats)
            self.prev_latent = latent

    def get_optimizer(self):
        return self.learner.optimizer

    def compute_loss(self, sample):
        return self.learner.compute_loss(sample)

    def get_optimizer(self):
        return self.learner.optimizer

    def get_nn_parameters(self):
        return self.policy.parameters()

    def reset_optimizer(self):
        self.learner.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.rl_lr, eps=self.optimizer_epsilon)
        return self.learner.optimizer

    def get_play_action(self, single_obs):
        obs = torch.from_numpy(single_obs).float().unsqueeze(0).to(self.device)
        masks = torch.ones(1,1).float().to(self.device)
        with torch.no_grad():
            res = self.policy.act(obs, None, masks)
        value, act, action_log_prob, _ = res
        return act[0].detach().cpu().numpy()


# This class closely follows
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
class RolloutStoragePPO(RolloutStorage):
    def __init__(self, num_steps, num_processes, frame_shape,
                 action_n, action_shape, aux_sz, latent_sz):
        super(RolloutStoragePPO, self).__init__(
            num_steps, num_processes, frame_shape,
            action_n, action_shape, aux_sz)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        if action_n is not None:
            self.acts = torch.zeros(num_steps, num_processes, 1)  # discrete act
        if latent_sz is not None:
            self.latents = torch.zeros(num_steps+1, num_processes, latent_sz)

    def to(self, device):
        super(RolloutStoragePPO, self).to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        if hasattr(self, 'acts'): self.acts = self.acts.to(device)
        if hasattr(self, 'latents'): self.latents = self.latents.to(device)

    def insert(self, frames, aux, masks, actions, rewards,
               action_log_probs, value_preds, acts, latents=None):
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        if hasattr(self, 'acts'): self.acts[self.step].copy_(acts)
        if hasattr(self, 'latents'): self.latents[self.step].copy_(latents)
        # Note: super called at the end because it increments self.step
        super(RolloutStoragePPO, self).insert(frames, aux, masks, actions, rewards)

    # Code below is from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    # but without bad_masks and recurrency.
    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[
                    step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages,
                               num_mini_batch=None, mini_batch_size=None):
        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),
                               mini_batch_size, drop_last=True)
        for indices in sampler:
            if  hasattr(self, 'latents'):
                obs_batch = self.latents[:-1].view(-1, *self.latents.size()[2:])[indices]
            else:
                obs_batch = self.aux[:-1].view(-1, *self.aux.size()[2:])[indices]
            if  hasattr(self, 'acts'):  # discrete
                actions_batch = self.acts.view(-1, self.acts.size(-1))[indices]
            else:  # continuous
                actions_batch = self.actions.view(-1, self.actions.size(-1))[indices]
            value_preds_batch = self.value_preds[:-1].view(-1, 1)[indices]
            return_batch = self.returns[:-1].view(-1, 1)[indices]
            masks_batch = self.masks[:-1].view(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.view(-1, 1)[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.view(-1, 1)[indices]

            yield obs_batch, None, actions_batch, \
                value_preds_batch, return_batch, masks_batch, \
                old_action_log_probs_batch, adv_targ
