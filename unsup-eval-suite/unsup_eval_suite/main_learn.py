"""
Main for training unsupervised learners and evaluating alignment.
"""

from collections import deque
from datetime import datetime
from glob import glob
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time

import numpy as np
np.set_printoptions(precision=2, linewidth=150, threshold=None, suppress=True)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
torch.set_printoptions(precision=4, linewidth=150, threshold=500000)

import gym

from .svae.svae_core import SVAE, DSA

from .all_args import get_all_args
from .rl.agent_random import AgentRandom
from .rl.agent_ppo import AgentPPO
from .utils.data_utils import extract_tgts
from .utils.env_utils import (
    make_vec_envs, aux_from_infos, make_aux_action, get_act_sz)
from .utils.load_log_save import (
    load_checkpoint, init_logging, log_info, do_logging, do_saving)
from .utils.analyze_utils import analyze_alignment
from .utils.viz import viz_samples
from .utils.prob import get_log_lik

import gym_bullet_aux  # need explicit import to register gym envs correctly


def multiproc_grads(num_grad_workers, parameters):
    if num_grad_workers==1: return  # not multiproc
    # Multiproc/MPI code: collect all gradients from workers.
    # https://pytorch.org/tutorials/intermediate/dist_tuto.html
    distributed_size = float(dist.get_world_size())
    for param in parameters:
        if param.grad is None: continue
        dist.all_reduce(param.grad.data,
                        op=torch.distributed.ReduceOp.SUM)
        param.grad.data /= distributed_size


def optimize_rl(rl_learner, epoch, num_grad_workers, rlts, unsup):
    rl_optimizer = rl_learner.get_optimizer()  # RL parameters
    if rl_optimizer is None: return  # nothing to optimize
    # Compute RL learning rates, returns, advantages for this epoch.
    # Bind rollouts to batch generator gen_fn.
    rl_gen_time = rl_loss_time = rl_update_time = 0
    advantages, gen_fn = rl_learner.process_rollouts(epoch, rlts, unsup)
    for curr_ppo_epoch in range(rl_learner.ppo_epoch):  # train PPO
        start = time.time()
        data_generator = gen_fn(advantages, rl_learner.num_mini_batch)
        rl_gen_time += time.time()-start
        for sample_id, sample in enumerate(data_generator):
            last_sub_iter = (curr_ppo_epoch+1==rl_learner.ppo_epoch and
                             sample_id+1==rl_learner.num_mini_batch)
            debug = epoch%args.log_interval==0 and last_sub_iter
            start = time.time()
            rl_loss = rl_learner.compute_loss(sample)  # compute main rl loss
            rl_loss_time += time.time()-start
            start = time.time()
            rl_optimizer.zero_grad()  # clear grads
            rl_loss.backward()        # prop RL grads
            # Clip gradients for RL loss only (part of PPO algo).
            torch.nn.utils.clip_grad_norm_(
                rl_learner.get_nn_parameters(), rl_learner.max_grad_norm)
            multiproc_grads(args.num_grad_workers, rl_learner.get_nn_parameters())
            rl_optimizer.step()  # optimize using RL gradients
            rl_update_time += time.time()-start
            if debug and args.rank==0:  # log, save
                msg = 'rl_gen_time {:0.6f} rl_loss_time {:0.6f} '
                msg += ' rl_update_time {:0.6f}'
                print(msg.format(rl_gen_time, rl_loss_time, rl_update_time))


def train_unsup_from_replay(agent, unsup, unsup_optim, args, epoch, debug,
                            logfile, tb_writer, viz_env):
    unsup.train()  # set internal torch flags
    start = time.time()
    obs_1toL, act_1toL, aux_1toL = agent.fill_seq_bufs_from_rollouts(
        agent.replay_rollouts, args.unsup_batch_size,
        unsup.pr.hist+unsup.pr.pred, unsup.device)
    unsup_gen_time = time.time()-start; start = time.time()
    kl_beta = 1.0
    if hasattr(args, 'svae_kl_beta'): kl_beta = args.svae_kl_beta
    unsup_loss, debug_dict = unsup.loss(
        obs_1toL, act_1toL, kl_beta,
        0 if args.ulo_loss_fxn == 'ground_truth_loss' else args.ulo_beta,
        epoch, debug)
    if (unsup_loss!=unsup_loss).any(): raise ValueError('NaN in unsup_loss')
    unsup_loss_time = time.time()-start
    #
    # TODO: remove after debugging
    extra_loss_time = 0
    use_gt = args.ulo_loss_fxn == 'ground_truth_loss'
    if use_gt or args.svae_decoder_extra_epochs>0:
        tmp_res = extract_tgts(obs_1toL, act_1toL, aux_1toL,
                               unsup.pr.hist, unsup.pr.past, unsup.pr.pred)
        tmp_auxs_tgt = tmp_res[-1]
        tmp_x_1toT_feats = unsup.conv_stack(tmp_res[0])
        z_smpls, _ = unsup.encoder(tmp_x_1toT_feats, tmp_res[1])
        # Train decoder more.
        start = time.time()
        mbsz = max(z_smpls.size(0)//4, 2)
        z_smpls_dtch = z_smpls.detach(); tmp_obs_tgt = tmp_res[-3]
        for tmpi in range(args.svae_decoder_extra_epochs):
            tmp_ids = torch.randperm(z_smpls_dtch.size(0))[0:mbsz]
            tmp_recon_xs = unsup.decoder(z_smpls_dtch[tmp_ids])
            tmp_recon_log_lik = get_log_lik(
                tmp_obs_tgt[tmp_ids], tmp_recon_xs, lp=2)
            tmp_dec_loss = tmp_recon_log_lik.mean().mul(-1)
            unsup_optim.zero_grad()
            tmp_dec_loss.backward()
            multiproc_grads(args.num_grad_workers, unsup.parameters())
            unsup_optim.step()
        extra_loss_time = time.time()-start
        # Align dynamic part of the latent space.
        if use_gt:
            #assert((tmp_auxs_tgt>=0).all() and (tmp_auxs_tgt<=1).all())
            gt_diff = torch.abs(z_smpls[:,:,unsup.pr.static_sz:] -
                                tmp_auxs_tgt[:,:,unsup.pr.static_sz:])
            gt_loss = args.ulo_beta*gt_diff.mean()
            unsup_loss = unsup_loss + gt_loss
            if debug and (epoch%args.log_interval)==0:
                for tmpd in range(unsup.pr.dynamic_sz):
                    debug_dict['train_gt_dim'+str(tmpd)+'_diff'] = \
                        gt_diff[:,:,tmpd].mean()
                    print('z_smpls\n', z_smpls[0:5,0,:])
                    print('tmp_auxs_tgt\n', tmp_auxs_tgt[0:5,0,:])
    # END: remove after debugging
    #
    start = time.time()
    unsup_optim.zero_grad()
    unsup_loss.backward()
    multiproc_grads(args.num_grad_workers, unsup.parameters())
    unsup_optim.step()
    unsup_update_time = time.time()-start
    if debug and args.rank==0:  # log, viz, save
        msg = 'unsup_gen_time {:0.4f} unsup_loss_time {:0.4f}'
        msg += ' unsup_update_time {:0.4f} extra_loss_time {:0.4f}'
        print(msg.format(unsup_gen_time, unsup_loss_time, unsup_update_time,
                         extra_loss_time))
        do_logging(epoch, debug_dict, {}, logfile, tb_writer)
        if epoch%args.viz_interval==0:
            viz_samples(unsup, obs_1toL, act_1toL, aux_1toL,
                        epoch, logfile, tb_writer, viz_env, 'train')


def get_test_obs_for_analyze(tot_sz, bsz, envs, agent, unsup, device):
    print('Collect', tot_sz, 'analyze test samples...')
    hist = unsup.pr.hist; past = unsup.pr.past; pred = unsup.pr.pred
    obs_1toL = []; act_1toL = []; aux_1toL = []; curr_sz = 0
    while curr_sz < tot_sz:
        rlts = agent.fill_rollouts(envs, hist+pred, unsup)
        res = agent.fill_seq_bufs_from_rollouts(rlts, bsz, hist+pred, device)
        tmp_obs_1toL, tmp_act_1toL, tmp_aux_1toL = res
        obs_1toL.append(tmp_obs_1toL); act_1toL.append(tmp_act_1toL)
        aux_1toL.append(tmp_aux_1toL)
        curr_sz = len(obs_1toL)*bsz
        if curr_sz%100==0: print('n anlz test', curr_sz)
    obs_1toL = torch.cat(obs_1toL, dim=0); act_1toL = torch.cat(act_1toL, dim=0)
    aux_1toL = torch.cat(aux_1toL, dim=0)
    assert(obs_1toL.size(0) == curr_sz)
    print('Collected', obs_1toL.size(), 'analyze test samples')
    return obs_1toL, act_1toL, aux_1toL


def get_train_data_for_analyze(tot_sz, bsz, rlts, agent, unsup,
                               epoch, logfile, tb_writer, viz_env, viz_title):
    print('Collect', tot_sz, 'analyze train samples...')
    latents = []; lowdims = []; curr_sz = 0
    while curr_sz < tot_sz:
        obs_1toL, act_1toL, aux_1toL = agent.fill_seq_bufs_from_rollouts(
            rlts, bsz, unsup.pr.hist+unsup.pr.pred, unsup.device)
        obs_1toT, act_1toT, aux_1toT, xs_tgt, acts_tgt, auxs_tgt = extract_tgts(
            obs_1toL, act_1toL, aux_1toL,
            unsup.pr.hist, unsup.pr.past, unsup.pr.pred)
        latent_code = unsup.latent_code(obs_1toT, act_1toT)
        lowdim_tgt = aux_1toT[:,-1] # low dim tgt: current (last past) frame
        latents.append(latent_code); lowdims.append(lowdim_tgt)
        curr_sz = len(lowdims)*bsz
        if curr_sz%100==0: print('n anlz train', curr_sz)
        if curr_sz >= tot_sz:
            viz_samples(unsup, obs_1toL, act_1toL, None, epoch, logfile,
                        tb_writer, viz_env, title_prefix=viz_title)
    latents = torch.cat(latents, dim=0)
    lowdims = torch.cat(lowdims, dim=0)
    assert(latents.size(0) == curr_sz)
    assert(lowdims.size(0) == curr_sz)
    print('Collected', lowdims.size(0), 'analyze train samples')
    return latents, lowdims


def compute_test_latents(unsup, obs_1toL, act_1toL, aux_1toL,
                         args, epoch, logfile, tb_writer, viz_env, viz_title):
    hist = unsup.pr.hist; past = unsup.pr.past; pred = unsup.pr.pred
    res = extract_tgts(obs_1toL, act_1toL, aux_1toL, hist, past, pred)
    obs_1toT, act_1toT, aux_1toT, _, _, _ = res
    # Low dim tgt: current (last past) frame
    lowdims = aux_1toT[:,-1,:].to(args.device)
    # Run unsup (VAE, SVAE) on the test frames.
    latents = []; strt = 0; tot_obs = obs_1toL.size(0)
    while strt < tot_obs:  # eval in pieces, since unsup is on GPU
        fnsh = min(strt+args.unsup_batch_size, tot_obs)
        if fnsh<=strt:
            print('compute_test_latents(): strt', strt, 'fnsh', fnsh)
            print('tot_obs', tot_obs, 'unsup_batch_size', args.unsup_batch_size)
            assert(fnsh>strt)
        ltnt_code = unsup.latent_code(obs_1toT[strt:fnsh].to(args.device),
                                      act_1toT[strt:fnsh].to(args.device))
        latents.append(ltnt_code)
        if fnsh>=tot_obs:
            viz_samples(unsup, obs_1toL[strt:fnsh].to(args.device),
                        act_1toL[strt:fnsh].to(args.device), None,
                        epoch, logfile, tb_writer, viz_env, viz_title)
        strt = fnsh
    latents = torch.cat(latents, dim=0)
    assert(strt==tot_obs)
    assert(latents.size(0)==lowdims.size(0))
    return latents, lowdims


def analyze(envs, replay_rlts, agent, unsup, test_obs_act_lowdim_1toL,
            aux_nms, args, epoch, logfile, tb_writer, viz_env):
    # Construct test set with frames from random/initial policy.
    test_latents, test_lowdims = compute_test_latents(
        unsup, *test_obs_act_lowdim_1toL, args, epoch,
        logfile, tb_writer, viz_env, 'test')
    # Construct test set with frames from the current policy.
    test_curr_latents = None; test_curr_lowdims = None
    if args.agent_algo != 'Random':
        test_curr_obs_act_lowdim_1toL = get_test_obs_for_analyze(
            args.analyze_dataset_size//4, args.num_envs_per_worker,
            envs, agent, unsup, args.device)
        test_curr_latents, test_curr_lowdims = compute_test_latents(
            unsup, *test_curr_obs_act_lowdim_1toL,
            args, epoch, logfile, tb_writer, viz_env, 'test_curr')
        assert(test_latents.size(1)==test_curr_latents.size(1))
    train_latents, train_lowdims = get_train_data_for_analyze(
        args.analyze_dataset_size, args.unsup_batch_size,
        replay_rlts, agent, unsup,
        epoch, logfile, tb_writer, viz_env, 'train_anlz')
    log_info(logfile, ['Analyze alignment for epoch {:d}'.format(epoch)])
    anlz_dict, anlz_dict_hist = analyze_alignment(
        train_latents, train_lowdims, test_latents, test_lowdims,
        test_curr_latents, test_curr_lowdims, aux_nms, args)
    # TODO: move to fxn
    # Plot
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize = (10,7))
    fig.tight_layout(); plt.tight_layout()
    ax = fig.add_subplot(111)
    irreg_ratios = anlz_dict_hist['anlz_test_irreg_ratios']
    irreg_dists = anlz_dict_hist['anlz_test_irreg_dists']
    print('irreg_ratios', irreg_ratios.size())
    print('irreg_dists', irreg_dists.size())
    ax.scatter(irreg_dists[:20000].detach().cpu().numpy(),
               irreg_ratios[:20000].detach().cpu().numpy(), s=1)
    ax.set_xlabel('lowdim dists')
    ax.set_ylabel('log(latentout/lodwim)')
    tb_writer.add_figure('anlz_test_irreg', fig, epoch)
    # end TODO
    if args.ulo_loss_fxn!='':
        res = extract_tgts(*test_obs_act_lowdim_1toL,
                           unsup.pr.hist, unsup.pr.past, unsup.pr.pred)
        tmp_auxs_tgt = res[-1]; tmp_acts_tgt = res[-2]
        print('test_latents', test_latents.size())
        print('tmp_auxs_tgt', tmp_auxs_tgt.size())
        tmp_test_latents = test_latents.view(*(tmp_auxs_tgt.size()))
        if hasattr(unsup, 'ulo'):
            assert(args.ulo_loss_fxn == 'latent_transfer_loss')
            unsup.ulo.latent_transfer_loss(
                tmp_test_latents, tmp_acts_tgt, unsup.pr.static_sz,
                unsup.pr.ulo_static_sz, anlz_dict, 'test_latents_')
            unsup.ulo.latent_transfer_loss(
                tmp_auxs_tgt, tmp_acts_tgt, unsup.pr.static_sz,
                unsup.pr.ulo_static_sz, anlz_dict, 'test_lodims_')
        else:
            assert(args.ulo_loss_fxn == 'ground_truth_loss')
            gt_diff = torch.abs(tmp_test_latents-tmp_auxs_tgt)
            for tmpd in range(tmp_test_latents.size(-1)):  # hist for curr state
                nm = str(aux_nms[tmpd])
                anlz_dict['test_gt_dim_'+nm+'_diff'] = gt_diff[:,:,tmpd].mean()
                anlz_dict_hist['test_latents_'+nm] = tmp_test_latents[:,0,tmpd]
                anlz_dict_hist['test_lowdims_'+nm] = tmp_auxs_tgt[:,0,tmpd]
    log_info(logfile, ['{:s}={:0.4f}'.format(k,v) for k,v in anlz_dict.items()]);
    do_logging(epoch, anlz_dict, anlz_dict_hist, logfile, tb_writer)


def analyze_checkpts(envs, replay_agent, unsup, test_obs_act_lowdim_1toL,
                     args, logfile, tb_writer, viz_env, aux_nms):
    hist = unsup.pr.hist; past = unsup.pr.past; pred = unsup.pr.pred
    assert(hist+pred <= args.agent_rollout_len)
    while replay_agent.num_replay_rollouts < replay_agent.max_replay_rollouts:
        rlts = replay_agent.fill_rollouts(envs, args.agent_rollout_len, unsup)
        replay_agent.update_replay(rlts)
        if replay_agent.num_replay_rollouts%10==0:
            print('replay sz', replay_agent.num_replay_rollouts)
    agent = replay_agent  # initial policy (random)
    for epoch in range(args.max_epochs+1):
        if (epoch%args.analyze_interval)!=0: continue
        prfr = agent.prev_frame; prau = agent.prev_aux; prmsk = agent.prev_masks
        agent = None; unsup = None  # try to free up memory
        unsup, _, agent, _, _ = load_checkpoint(
            args.analyze_path+str(epoch)+'.pt', args.device,
            envs, args, logfile, tb_writer, load_optim=False)
        assert(unsup is not None)
        agent.init_prev_obs(prfr, prau, prmsk, unsup)
        rlts = agent.fill_rollouts(envs, args.agent_rollout_len, unsup)
        replay_agent.update_replay(rlts)  # store results for curr policy in rpl
        obs_1toL, act_1toL, aux_1toL = agent.fill_seq_bufs_from_rollouts(
            rlts, args.num_envs_per_worker, hist+pred, unsup.device)
        viz_samples(unsup, obs_1toL, act_1toL, aux_1toL,
                    epoch, logfile, tb_writer, viz_env, 'streaming')
        analyze(envs, replay_agent.replay_rollouts, agent, unsup,
                test_obs_act_lowdim_1toL, aux_nms,
                args, epoch, logfile, tb_writer, viz_env)
        agent.do_logging(epoch, args.agent_rollout_len, tb_writer)
        debug = (epoch%args.log_interval)==0
        obs_1toL, act_1toL, aux_1toL = replay_agent.fill_seq_bufs_from_rollouts(
            replay_agent.replay_rollouts, args.unsup_batch_size,
            hist+pred, unsup.device)
        unsup_loss, debug_dict = unsup.loss(
            obs_1toL, act_1toL, args.svae_kl_beta, args.ulo_beta, epoch, debug)
        do_logging(epoch, debug_dict, {}, logfile, tb_writer)
        viz_samples(unsup, obs_1toL, act_1toL, aux_1toL,
                    epoch, logfile, tb_writer, viz_env, 'train')
    log_info(logfile, ['analyze_checkpts() done!'])


def train(envs, agent, unsup, unsup_optim, test_obs_act_lowdim_1toL,
          args, start_epoch, logfile, tb_writer, viz_env, aux_nms):
    #
    # Main loop for train and analyze.
    #
    if args.rank==0: log_info(logfile, ['Training started'])
    epoch = start_epoch
    while epoch <= args.max_epochs:
        rlts = agent.fill_rollouts(envs, args.agent_rollout_len, unsup)
        if args.unsup_num_sub_epochs>0:
            agent.update_replay(rlts)
            if agent.num_replay_rollouts < agent.max_replay_rollouts:
                if agent.num_replay_rollouts%10==0:
                    print('replay sz', agent.num_replay_rollouts)
                continue  # not ready to train yet, fill replay first
        if args.rank==0 and (epoch%args.viz_interval)==0:
            # Visualize new rollouts (before using them for training).
            obs_1toL, act_1toL, aux_1toL = agent.fill_seq_bufs_from_rollouts(
                rlts, args.num_envs_per_worker,
                unsup.pr.hist+unsup.pr.pred, unsup.device)
            viz_samples(unsup, obs_1toL, act_1toL, aux_1toL,
                        epoch, logfile, tb_writer, viz_env, 'streaming')
        if (args.rank==0 and (args.analyze_interval>0) and
            (epoch%args.analyze_interval)==0):
            analyze(envs, agent.replay_rollouts, agent, unsup,
                    test_obs_act_lowdim_1toL, aux_nms,
                    args, epoch, logfile, tb_writer, viz_env)
        # Train RL and unsupervised representation learner.
        optimize_rl(agent, epoch, args.num_grad_workers, rlts, unsup)
        if args.rank==0: agent.do_logging(epoch, args.agent_rollout_len, tb_writer)
        for sub_epoch in range(args.unsup_num_sub_epochs):  # train unsup
            last_sub_epoch = sub_epoch+1==args.unsup_num_sub_epochs
            debug = (epoch%args.log_interval)==0 and last_sub_epoch
            train_unsup_from_replay(
                agent, unsup, unsup_optim, args, epoch,
                debug, logfile, tb_writer, viz_env)
        if args.rank==0 and (epoch%args.viz_interval)==0:
            do_saving(agent, unsup, unsup_optim, args, epoch, logfile)
        epoch += 1  # advance epoch
    if args.rank==0: log_info(logfile, ['Training done!'])


def main(rank, args):
    args.rank = rank  # set local rank param
    logfile = None; tb_writer = None
    if args.rank==0:
        # tensorboardX import should be done after init_logging().
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(args.save_path)
        logfname = os.path.join(args.save_path, 'log.txt')
        logfile = open(logfname, 'w', buffering=1)
        if rank==0: log_info(logfile, [datetime.now().strftime('%Y-%m-%d'), args])
    if args.num_grad_workers>1:
        # https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html
        # https://github.com/pytorch/pytorch/issues/20037
        # Note: overall DataParallel does not seem to speed up training :-(
        # https://discuss.pytorch.org/t/debugging-dataparallel-no-speedup-and-uneven-memory-allocation/1100/27
        # Perhaps this would provide benefit for much larger nets...
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12345'  # any usable TCP port
        os.environ['WORLD_SIZE'] = str(args.num_grad_workers)
        os.environ['RANK'] = str(rank)
        os.environ['OMP_NUM_THREADS'] = '1'
        # Initialize MPI, GLOO or other multiproc backend.
        # TODO: gloo seems to be slower than non-distributed... so use nccl
        dist.init_process_group(backend='nccl', world_size=args.num_grad_workers)
        assert(rank==dist.get_rank())
        num_grad_workers = dist.get_world_size()
        print('Starting worker {} out of {}'.format(rank, num_grad_workers))

    # Init GPUs and randomness.
    use_cuda = (args.gpu is not None) and torch.cuda.is_available()
    args.device = 'cuda:'+str(args.gpu+rank) if use_cuda else 'cpu'
    if args.agent_device is None: args.agent_device = args.device
    if args.agent_replay_device is None: args.agent_replay_device = args.device
    np.random.seed(args.seed*1000+rank)  # different numpy seeds
    torch.manual_seed(args.seed)  # same seed for CUDA to get same model weights
    if use_cuda:
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.deterministic = False   # faster, less reproducible
        torch.cuda.manual_seed_all(args.seed)

    # Create parallel envs for training and a single env for viz (DummyVecEnv).
    envs = make_vec_envs(args.env_name, args, args.seed,
                         args.num_envs_per_worker, args.device)
    viz_env = make_vec_envs(args.env_name, args, args.seed, 1, args.device)
    init_frame = envs.reset()
    clr_chn, im_h, im_w = envs.observation_space.shape
    assert(clr_chn == 3)
    assert(im_h == im_w)
    args.im_sz = im_h
    args.act_sz = get_act_sz(envs.action_space)
    aux_action = make_aux_action(
        args.num_envs_per_worker, envs.action_space, args.device)
    _, _, _, infos = envs.step(aux_action)
    init_aux = aux_from_infos(infos, args.device)
    aux_nms = infos[0]['aux_nms']
    args.aux_sz = init_aux.size(-1)

    # Create or load unsupervised learner and active agent.
    epoch = 0; all_epochs = [epoch]; all_args = [args]; latent_sz = None
    if args.load_checkpt is not None:
        unsup, unsup_optim, agent, epoch, loaded_args = load_checkpoint(
            args.load_checkpt, args.device, envs, args, logfile, tb_writer)
        all_args.append(loaded_args); all_epochs.append(epoch)
        if args.agent_use_unsup_latent: latent_sz = unsup.latent_sz()
    else:  # construct new
        # Setup unsupervised learner.
        unsup = eval(args.unsup_class)(args, tb_writer)
        unsup_optim = torch.optim.Adam(
            [{"params": unsup.parameters(), "lr": args.unsup_lr}])
        assert(unsup.pr.hist+unsup.pr.pred <= args.agent_rollout_len)
        assert(args.unsup_batch_size > unsup.pr.hist)  # for equal seq, nonseq
        args.unsup_batch_size = args.unsup_batch_size//(unsup.pr.hist+unsup.pr.pred)
        if args.agent_use_unsup_latent: latent_sz = unsup.latent_sz()
        # Setup active agent.
        if args.analyze_path is not None: args.agent_replay_rnd_frac = 0
        agent_class = 'Agent'+args.agent_algo
        agent = eval(agent_class)(
            envs, latent_sz, unsup.pr.hist, args.aux_sz,
            args.num_envs_per_worker, args.agent_max_replay_rollouts,
            args, logfile)
    if args.agent_max_replay_rollouts < args.unsup_batch_size:
        print('args.agent_max_replay_rollouts', args.agent_max_replay_rollouts,
              'vs args.unsup_batch_size', args.unsup_batch_size)
        assert(args.agent_max_replay_rollouts >= args.unsup_batch_size)
    # Load ULO relations, if given.
    if args.ulo_checkpt is not None:  # attach ULO relations to unsup
        T = unsup.pr.past+unsup.pr.pred
        g_inp_ndims = unsup.pr.ulo_static_sz+T*(unsup.pr.dynamic_sz)+args.act_sz
        unsup.ulo = ULO(g_inp_ndims)
        unsup.ulo.load(args.ulo_checkpt, args=None, device=args.device,
                       logfile=logfile)

    # Step through envs to remove transients.
    test_obs_act_lowdim_1toL = None
    if args.rank==0:
        ntransient = 119
        rnd_agent = AgentRandom(envs, latent_sz, unsup.pr.hist, args.aux_sz,
                                args.num_envs_per_worker, max_replay_rollouts=0,
                                args=args, logfile=None)
        rnd_agent.init_prev_obs(init_frame, init_aux, None, unsup)
        for tmp_i in range(ntransient):
            envs.step(rnd_agent.get_random_action()[1])
        init_frame, _, _, infos = envs.step(rnd_agent.get_random_action()[1])
        init_aux = aux_from_infos(infos, args.device)
        # Collect test set using random policy. Note: need to use AgentRandom
        # instead of initial RL policy to get some motion (other than gravity).
        test_obs_act_lowdim = (None, None, None)
        if args.analyze_interval>0:
            if args.rank==0:
                log_info(logfile, ['Collect testset with AgentRandom'])
            rnd_agent.init_prev_obs(init_frame, init_aux, None, unsup)
            test_obs_act_lowdim_1toL = get_test_obs_for_analyze(
                args.analyze_dataset_size, args.num_envs_per_worker,
                envs, rnd_agent, unsup, args.device)
            init_frame = envs.reset()
            _, _, _, infos = envs.step(aux_action)
            init_aux = aux_from_infos(infos, args.device)
            rnd_agent = None  # garbage collect

    # Prepare agent for action; finalize args for unsup (adjust batch size to
    # ensure we have use same ntrain frames for seq, and nonseq).
    agent.init_prev_obs(init_frame, init_aux, None, unsup)

    # Print command-line arg values for easy inspection
    if args.rank==0:
        for tmp_id, tmp_args in enumerate(all_args):
            args_str = ''
            for arg in vars(tmp_args):
                # Tensorboard uses markdown-like formatting, hence '  \n'.
                args_str += '  \n{:s}={:s}'.format(
                    str(arg), str(getattr(args, arg)))
            tb_writer.add_text('args', args_str, all_epochs[tmp_id])

    # Run training (or analyze).
    if args.analyze_path is None:
        train(envs, agent, unsup, unsup_optim, test_obs_act_lowdim_1toL,
              args, epoch, logfile, tb_writer, viz_env, aux_nms)
    else:
        assert(args.load_checkpt is None)
        assert(args.num_grad_workers==1)
        analyze_checkpts(envs, agent, unsup, test_obs_act_lowdim_1toL,
                         args, logfile, tb_writer, viz_env, aux_nms)

    # Cleanup
    if rank==0: logfile.close()
    if args.num_grad_workers>1 and torch.distributed.is_available():
        dist.destroy_process_group()  # cleanup


if __name__ == '__main__':
    args = get_all_args()
    if args.max_epochs is None:
        steps_per_epoch = args.agent_rollout_len * args.num_envs_per_worker
        args.max_epochs = args.total_env_steps // steps_per_epoch
    if args.env_name.startswith(('Cnplt')): args.analyze_interval = None
    short_env = ('CartPole' in args.env_name or
                 'InvertedPendulum' in args.env_name or
                 'BlockOnIncline' in args.env_name)
    if args.agent_rollout_len is None:
        args.agent_rollout_len = 32 if short_env else 64
    if args.unsup_num_sub_epochs is None:
        args.unsup_num_sub_epochs = 10 if short_env else 50
    # Rearrange has max 50 steps w/ 4 repeats: 200=50x4; Block is max 24 steps.
    if 'Rearrange' in args.env_name or 'BlockOnIncline' in args.env_name:
        args.agent_rollout_len = 10   # use short rollouts to speed up
    # Set up logging intervals based on the number of epochs.
    default_log_interval = args.max_epochs//500
    if args.log_interval is None:
        args.log_interval = min(default_log_interval, 10)
    if args.viz_interval is None:
        args.viz_interval = min(10*default_log_interval, 100)
    if args.analyze_interval is None:
        args.analyze_interval = min(50*default_log_interval, 100)
    # Set up logging to terminal and log file, then set up Tensorboard.
    # Note: init_logging has to be called before tensorboardX import.
    args.save_path, args.checkpt_path = init_logging(args)
    # Start distributed main.
    if torch.distributed.is_available() and args.num_grad_workers>1:
        mp.spawn(main, args=(args,),  # rank passed as 0th arg by distr
                 nprocs=args.num_grad_workers, join=True)
    else:
        main(rank=0, args=args)
