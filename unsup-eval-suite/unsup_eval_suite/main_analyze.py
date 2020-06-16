"""
Main for evaluating the alignment of trained unsupervised learners.
"""
import argparse
from collections import deque
import re
import os
from sys import platform
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time

import numpy as np
np.set_printoptions(precision=2, linewidth=150, threshold=None, suppress=True)
import torch
torch.set_printoptions(precision=4, linewidth=150, threshold=500000)

import gym

import gym_bullet_aux  # need explicit import to register gym envs correctly
from gym_bullet_aux.envs.aux_bullet_env import AuxBulletEnv
from gym_bullet_aux.envs.block_on_incline_env import BlockOnInclineEnv

from .rl.rollout_storage import RolloutStorage
from .utils.env_utils import (
    make_vec_envs, aux_from_infos, make_aux_action, get_act_sz)
from .utils.load_log_save import (
    load_checkpoint, init_logging, log_info, do_logging, do_saving)
from .utils.analyze_utils import analyze_alignment
from .utils.viz import viz_samples


def get_args():
    parser = argparse.ArgumentParser(description="Analyze")
    parser.add_argument('--load_checkpt', type=str, default=None,
                        help='Load checkpoint to play or analyze')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize in PyBullet simulator')
    parser.add_argument('--play', type=int, default=10,
                        help='Number of episodes to play loaded RL policy')
    parser.add_argument('--analyze_dataset_size', type=int, default=1000,
                        help='Get this many points for analysis training')
    parser.add_argument('--env_name', type=str, default=None, help='Env name')
    args = parser.parse_args()
    return args


def guess_env(checkpt_path, env_name, obs_resolution, do_viz):
    if env_name is None:
        env_parts = checkpt_path.split('BulletEnv-v')
        guessed_env_v = int(env_parts[1].split('_')[0])
        env_parts = env_parts[0].split('Aux')
        guessed_base_env_name = env_parts[-1]
        print('Guessed base_env_name', guessed_base_env_name,
              'env_v', guessed_env_v)
        random_colors = False
        if guessed_base_env_name.endswith('Clr'):
            random_colors = True
            guessed_base_env_name = guessed_base_env_name.rstrip('Clr')
        env = AuxBulletEnv(
            base_env_name=guessed_base_env_name, env_v=guessed_env_v,
            obs_resolution=obs_resolution, random_colors=random_colors,
            visualize=do_viz, debug_level = 10 if do_viz else 0)
    else:
        assert('BlockOnIncline' in env_name)
        guessed_base_env_name = 'BlockOnIncline'
        env_parts = env_name.split('-v')
        guessed_env_v = int(env_parts[1])
        variant = 'Ycb' if 'Ycb' in env_name else 'Geom'
        env = BlockOnInclineEnv(
            version=guessed_env_v, variant=variant,
            obs_resolution=obs_resolution,
            visualize=do_viz, debug_level = 10 if do_viz else 0)
    return env, guessed_base_env_name, guessed_env_v


def play(args):
    env, guessed_base_env_name, guessed_env_v = guess_env(
        args.load_checkpt, args.env_name, obs_resolution=None, do_viz=args.viz)
    unsup, unsup_optim, agent, epoch, loaded_args = load_checkpoint(
        args.load_checkpt, 'cpu', env, args, tb_writer=None, logfile=None)
    if args.env_name is None:
        env_parts = loaded_args.env_name.split('BulletEnv-v')
        base_env_name = env_parts[0][len('Aux'):]; env_v = int(env_parts[1])
        if base_env_name.endswith('Clr'):
            base_env_name  = base_env_name.rstrip('Clr')
        if guessed_base_env_name != base_env_name:
            print('guessed_base_env_name', guessed_base_env_name)
            print('base_env_name', base_env_name)
            assert(guessed_base_env_name == base_env_name)
        assert(guessed_env_v == env_v)
    num_episodes = args.play
    noop_action = np.nan * np.zeros(env.action_space.shape)
    for epsd in range(args.play):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        obs, _, _, info = env.step(noop_action)
        step = 0
        input('Reset done; press enter to start episode')
        while True:
            act = agent.get_play_action(obs)
            next_obs, rwd, done, info = env.step(act)
            if platform.startswith('linux'): time.sleep(0.01)
            if done:
                if 'episode' in info.keys():
                    print('Episode reward {:0.4f}'.format(info['episode']['r']))
                break
            obs = next_obs
            step += 1
        if args.viz: input('Episode ended; press enter to go on')


def eval(load_checkpt, analyze_dataset_size, gpu):
    use_cuda = (gpu is not None) and torch.cuda.is_available()
    device = 'cuda:'+str(gpu) if use_cuda else 'cpu'
    # Load unsup and RL checkpoint.
    env = guess_env(load_checkpt, obs_resolution=64, do_viz=False)
    unsup, unsup_optim, agent, epoch, args = load_checkpoint(
        load_checkpt, device, env, args=None, logfile=None)
    # Set up tbwriter and logging.
    args.save_path, args.checkpt_path = init_logging('~/Desktop/tmp/')
    from tensorboardX import SummaryWriter
    tb_writer = SummaryWriter(args.save_path)
    logfname = os.path.join(args.save_path, 'log.txt')
    logfile = open(logfname, 'w', buffering=1)
    # Create parallel envs for analysis and a single env for viz (DummyVecEnv).
    envs = make_vec_envs(args.env_name, args, args.seed,
                         args.num_envs_per_worker, args.device)
    viz_env = make_vec_envs(args.env_name, args, args.seed, 1, args.device)
    # Init env frames.
    prev_frame = envs.reset()
    aux_action = make_aux_action(
        args.num_envs_per_worker, envs.action_space, args.device)
    _, _, _, infos = envs.step(aux_action)
    prev_aux = aux_from_infos(infos, args.device)
    prev_masks = None
    aux_nms = infos[0]['aux_nms']
    clr_chn, im_h, im_w = envs.observation_space.shape
    args.im_sz = im_h
    args.act_sz = get_act_sz(envs.action_space)
    args.aux_sz = prev_aux.size(-1)
    accum_sz = int(analyze_dataset_size/args.unsup_batch_size)
    print('Set max_num_replay_rollouts to', analyze_dataset_size)
    agent.max_num_replay_rollouts = analyze_dataset_size
    assert(agent.max_num_replay_rollouts%agent.num_envs_per_worker==0)
    agent.replay_device = 'cpu'
    agent.replay_rollouts = RolloutStorage(
        agent.rollout_len, agent.max_num_replay_rollouts,
        envs.observation_space.shape, agent.action_n, agent.action_shape,
        agent.aux_sz)
    print('Fill replay...')
    while agent.num_replay_rollouts<agent.max_num_replay_rollouts:
        agent.rollouts = agent.make_rollouts(
            envs, prev_frame, prev_aux, prev_masks)
        agent.fill_rollouts(envs)
        if agent.num_replay_rollouts%10==0:
            print('num replay', agent.num_replay_rollouts)
        prev_frame, prev_aux, prev_masks = agent.rollouts.clone_prev()
    num_eval_runs = 10
    for rn in range(num_eval_runs):
            do_eval_run(envs, viz_env, agent, unsup, args, epoch+rn, accum_sz,
                        aux_nms, logfile, tb_writer)
    log_info(logfile, ['Eval done!'])
    env.close()
    envs.close()
    viz_env.close()


def dump_obs(obs, fname):
    import imageio
    img = obs.detach().cpu().numpy().swapaxes(0,2).swapaxes(0,1)
    img = (img*255).astype(np.uint8)
    imageio.imwrite(fname, img)


def do_eval_run(envs, viz_env, agent, unsup, args, epoch, accum_sz,
                aux_nms, logfile, tb_writer):
    latents_accum = deque(maxlen=accum_sz)
    lowdims_accum = deque(maxlen=accum_sz)
    for i in range(accum_sz):
        obs_1toL, act_1toL, aux_1toL = agent.fill_seq_bufs_from_rollouts(
            agent.replay_rollouts, args.unsup_batch_size,
            unsup.pr.hist+unsup.pr.pred, unsup.device)
        bsz, seq_len, *_ = obs_1toL.size()
        recon_x, _, z_distr = unsup.recon(
            obs_1toL, act_1toL, epoch, debug=i%10==0)
        recon_x = recon_x.detach()
        # Note: using mu instead of smpl, since we applied sigmoid to smpl.
        z_mu = z_distr.mu.view(bsz, seq_len, -1).detach()
        z_std = torch.exp(0.5*z_distr.logvar).view(bsz, seq_len, -1).detach()
        latents_accum.append(z_mu)
        lowdims_accum.append(aux_1toL)
        tmp_lowdim = aux_1toL
        tmp = tmp_lowdim.view(bsz*seq_len, -1)[:,0]
        print('Neg theta\n', tmp_lowdim.view(bsz*seq_len, -1)[tmp<-3.0].cpu().numpy())
        print('z_mu\n', z_mu.view(bsz*seq_len, -1)[tmp<-3.0].cpu().numpy())
        print('Pos theta\n', tmp_lowdim.view(bsz*seq_len, -1)[tmp>=3.0].cpu().numpy())
        print('z_mu\n', z_mu.view(bsz*seq_len, -1)[tmp>=3.0].cpu().numpy())
        if False:  # debug
            print('nms', aux_nms)
            for bid in range(bsz):
                for tid in range(seq_len):
                    #if np.abs(lowdim[bid,tid,0])<1.0: continue
                    print('lowdim', lowdim[bid,tid].cpu().numpy())
                    print('latent', z_mu[bid,tid].cpu().numpy())
                    dump_obs(obs_1toL[bid,tid,:], fname='/tmp/debug_obs.png')
                    dump_obs(recon_x[bid,tid,:], fname='/tmp/debug_recon.png')
                    input('continue')
        viz_samples(unsup, obs_1toL, act_1toL, aux_1toL,
                    epoch, logfile, tb_writer, viz_env, 'anlz')
    # Analyze alignment with lowdim.
    args.analyze_max_iter = int(1e7)
    anlz_dict, anlz_dict_hist = analyze_alignment(
        latents_accum, lowdims_accum, aux_nms, args, use_nn=True, use_gp=False)
    do_logging(epoch, anlz_dict, anlz_dict_hist, logfile, tb_writer)


def main(args):
    if args.play>0:
        play(args)
    else:
        eval(args.load_checkpt, args.analyze_dataset_size, args.gpu)


if __name__ == '__main__':
    main(get_args())
