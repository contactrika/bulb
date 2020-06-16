"""
RL Learner interface: AgentRandom.
"""

from collections import deque
from datetime import datetime
import logging
import os
import time

import numpy as np
import torch

import gym

from .rollout_storage import RolloutStorage
from ..utils.env_utils import make_aux_action


def to_onehot(action, action_n):
    action_onehot = torch.zeros(
        [action.size(0), action_n], device=action.device)
    # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_
    action_onehot.scatter_(1, action, 1.0)  # set 1.0 for idx of a at dim=1
    return action_onehot


class AgentRandom:
    def __init__(self, envs, latent_sz, latent_hist, aux_sz,
                 num_envs_per_worker, max_replay_rollouts, args, logfile):
        logging.info('Constructed AgentRandom')
        self.num_envs_per_worker = num_envs_per_worker
        self.max_replay_rollouts = max_replay_rollouts
        self.device = args.agent_device
        if self.device is None: self.device = args.device
        self.replay_device = args.agent_replay_device
        if self.replay_device is None: self.replay_device = args.device
        self.frames_shape = envs.observation_space.shape
        self.action_shape = envs.action_space.shape
        self.action_n = None
        if isinstance(envs.action_space, gym.spaces.Discrete):
            self.action_n = envs.action_space.n
            self.action_shape = (envs.action_space.n,)  # store onehot
        else:
            self.action_low = torch.from_numpy(envs.action_space.low).to(self.device)
            rng = envs.action_space.high - envs.action_space.low
            self.action_rng =  torch.from_numpy(rng).to(self.device)
        self.latent_sz = latent_sz; self.latent_hist = latent_hist
        self.aux_sz = aux_sz
        self.aux_action = make_aux_action(
            self.num_envs_per_worker, envs.action_space, self.device)
        self.prev_frame = None
        self.prev_aux = None
        self.prev_masks = None
        self.prev_latent = None
        self.replay_rollouts = None
        if max_replay_rollouts>0:
            self.replay_offset = \
                int(max_replay_rollouts * args.agent_replay_rnd_frac)
            assert(num_envs_per_worker <=
                   (self.max_replay_rollouts-self.replay_offset))
            self.replay_rollouts = RolloutStorage(
                args.agent_rollout_len, self.max_replay_rollouts,
                envs.observation_space.shape, self.action_n, self.action_shape,
                self.aux_sz)
            self.replay_rollouts.to(self.replay_device)
        self.num_replay_rollouts = 0
        self.episode_rewards = deque(maxlen=10)
        if args.rank==0:
            self.start_time = time.time()
            self.log_interval = args.log_interval
            self.logfile = logfile
            self.log(str(args))

    def get_optimizer(self):
        return None  # not optimizing anything for this random agent

    def process_rollouts(self, epoch):
        pass

    def init_prev_obs(self, prev_frame, prev_aux, prev_masks=None, unsup=None):
        self.prev_frame = prev_frame
        self.prev_aux = prev_aux
        if prev_masks is not None: self.prev_masks = prev_masks

    def make_rollouts(self, envs, rollout_len):
        rlts = RolloutStorage(
            rollout_len, self.num_envs_per_worker, self.frames_shape,
            self.action_n, self.action_shape, self.aux_sz)
        rlts.to(self.device)
        rlts.frames[0] = self.prev_frame; rlts.aux[0] = self.prev_aux
        if self.prev_masks is not None: rlts.masks[0] = self.prev_masks
        return rlts

    def fill_rollouts(self, envs, rollout_len, unsup=None):
        rlts = self.make_rollouts(envs, rollout_len)
        for step in range(rollout_len):
            self.rollout_step(envs, step, rlts, unsup)  # step envs
        self.prev_frame, self.prev_aux, self.prev_masks = rlts.clone_prev()
        return rlts

    def update_replay(self, rlts):
        # Assumes self.replay_rollouts has been created and initialized.
        assert(self.max_replay_rollouts > 0)
        rlt_bids = np.arange(rlts.frames.size(1))
        num_new_frames = rlt_bids.shape[0]
        num_free = self.max_replay_rollouts - self.num_replay_rollouts
        num_fill = min(num_new_frames, num_free)
        if num_fill>0:  # just add new frames into replay until full
            rpl_ids = np.arange(self.num_replay_rollouts,
                                self.num_replay_rollouts+num_fill)
            self.num_replay_rollouts += num_fill
            self.replace_in_replay(rlts, rpl_ids, rlt_bids[0:num_fill])
        num_left = num_new_frames - num_fill
        if num_left>0: # replace frames randomly
            assert(self.num_replay_rollouts==self.max_replay_rollouts)
            rpl_ids = np.random.permutation(np.arange(
                self.replay_offset, self.max_replay_rollouts))[0:num_left]
            self.replace_in_replay(rlts, rpl_ids, rlt_bids[num_fill:])


    def replace_in_replay(self, rlts, rpl_ids, bids):
        assert(min(rpl_ids)>=0 and max(rpl_ids)<self.num_replay_rollouts)
        assert(rpl_ids.shape==bids.shape)
        rpl_dvc = self.replay_rollouts.frames.device
        self.replay_rollouts.frames[:,rpl_ids] = rlts.frames[:,bids].to(rpl_dvc)
        self.replay_rollouts.actions[:,rpl_ids] = rlts.actions[:,bids].to(rpl_dvc)
        if self.replay_rollouts.aux is not None:
            self.replay_rollouts.aux[:,rpl_ids] = rlts.aux[:,bids].to(rpl_dvc)
        self.replay_rollouts.masks[:,rpl_ids] = rlts.masks[:,bids].to(rpl_dvc)
        self.replay_rollouts.rewards[:,rpl_ids] = rlts.rewards[:,bids].to(rpl_dvc)


    def rollout_step(self, envs, step, rlts, unsup):
        action, act = self.get_random_action()
        next_frame, next_aux, masks, reward, _ = self.make_env_step(envs, act)
        rlts.insert(next_frame, next_aux, masks, action, reward)

    def get_play_action(self, single_obs):
        action, act = self.get_random_action()
        return act[0]  # return a single action for play

    def get_random_action(self):
        # a basic random action; subclasses should override this function
        if self.action_n is None:
            action = torch.rand([self.num_envs_per_worker, *self.action_shape])
            action = action.to(self.device)*self.action_rng + self.action_low
            act = action
        else:
            act = torch.randint(self.action_n, (self.num_envs_per_worker,1))
            action = to_onehot(act, self.action_n)
        return action, act

    def make_env_step(self, envs, act):
        next_frame, reward, done, infos = envs.step(act.to(self.device))
        next_aux = []
        for i, info in enumerate(infos):
            if 'episode' in info.keys():
                self.episode_rewards.append(info['episode']['r'])
            if self.aux_sz>0: next_aux.append(torch.from_numpy(info['aux']))
        if self.aux_sz>0: next_aux = torch.stack(next_aux, dim=0).to(self.device)
        masks = [[0.0] if done_ else [1.0] for done_ in done]
        masks = torch.FloatTensor(masks)
        return next_frame, next_aux, masks, reward, done

    def fill_seq_bufs_from_rollouts(self, rlts, batch_size, seq_len, device):
        num_rlts = rlts.frames.size(1)  # in rls storage: dim0 is t, dim1 is bid
        frame_size =  rlts.frames.size()[2:]
        frames_1toL = torch.zeros(batch_size, seq_len, *frame_size).to(self.device)
        act_1toL = torch.zeros(batch_size, seq_len, *self.action_shape).to(self.device)
        if self.aux_sz>0:
            aux_size = rlts.aux.size()[2:]
            aux_1toL = torch.zeros(batch_size, seq_len, *aux_size).to(self.device)
        # TODO: maybe do a smart selection later
        bids = torch.randperm(num_rlts)[0:batch_size]
        for i, bid in enumerate(bids):
            tid = torch.randint(rlts.frames.size(0)-seq_len, (1,))[0]
            currbid_masks_1toL = rlts.masks[tid:tid+seq_len,bid].squeeze(-1)
            assert(currbid_masks_1toL.size(0)==seq_len)
            frames_1toL[i,:,:] = rlts.frames[tid:tid+seq_len,bid,:]
            if self.aux_sz>0: aux_1toL[i,:,:] = rlts.aux[tid:tid+seq_len,bid,:]
            action = rlts.actions[tid:tid+seq_len,bid,:]
            act_1toL[i,:,:] = action[:,:]
            # Mask should be 0 only at the start of the episode.
            # The offset works out to be one less than the id of the first
            # zero mask (one less because we want to replicate the frame just
            # before the 1st occurrence of mask==0).
            last_tid = None  # replicate last frame until the end
            if (currbid_masks_1toL[1:]<1).any():
                tmp_done_2toL = torch.abs(1-currbid_masks_1toL[1:])
                res = tmp_done_2toL.nonzero(as_tuple=True)[0]
                next_episode_tid = 1 + res[0].item()
                last_tid = next_episode_tid - 1
            if last_tid is not None:
                frames_1toL[i,last_tid:,:] = rlts.frames[tid+last_tid,bid,:]
                if self.aux_sz>0:
                    aux_1toL[i,last_tid:,:] = rlts.aux[tid+last_tid,bid,:]
                act_1toL[i,last_tid:,:] = rlts.actions[tid+last_tid-1,bid,:]
        """
        if debug:
            import imageio
            print('frames_1toL', frames_1toL.size())
            for tmp_bid in range(frames_1toL.size(0)):
                for tmp_t in range(frames_1toL.size(1)):
                    rgb_pix = frames_1toL[tmp_bid,tmp_t].numpy().swapaxes(0,2).swapaxes(0,1)
                    tmp_pfx = '/tmp/tmp_frombuf_bid'+str(tmp_bid)+'_t'+str(tmp_t)
                    imageio.imwrite(tmp_pfx+'.png', rgb_pix)
                    with open(tmp_pfx+'.txt', 'w') as tmp_aux_file:
                        tmp_aux_file.write(str(aux_1toL[tmp_bid, tmp_t]))
        input('Continue fill_seq_bufs_from_rollouts')
        """
        aux_1toL = None if self.aux_sz==0 else aux_1toL.to(device)
        return frames_1toL.to(device), act_1toL.to(device), aux_1toL

    def do_logging(self, epoch, rollout_len, tb_writer):
        if self.logfile is not None and epoch%self.log_interval==0:
            total_num_steps = (epoch+1)*self.num_envs_per_worker*rollout_len
            end = time.time()
            print_str = '{} {} epoch {} num timesteps {} FPS {}'.format(
                datetime.now().strftime('%H:%M:%S'), self.__class__.__name__,
                epoch, total_num_steps, int(total_num_steps/(end-self.start_time)))
            self.log(print_str)
            if len(self.episode_rewards)>1:
                eprews = self.episode_rewards
                msg = 'Last {} train epds: rwd mean/median {:.3f}/{:.3f}'
                msg += ' min/max {:.3f}/{:.3f}'
                print_str = msg.format(
                    len(eprews), np.mean(eprews), np.median(eprews),
                    np.min(eprews), np.max(eprews))
                self.log(print_str)
                tb_writer.add_scalar('mean_agent_rwd', np.mean(eprews), epoch)
                tb_writer.add_scalar('min_agent_rwd', np.min(eprews), epoch)
                tb_writer.add_scalar('max_agent_rwd', np.max(eprews), epoch)

    def log(self, print_str):
        print(print_str)
        if self.logfile is not None: self.logfile.write(print_str+'\n')
