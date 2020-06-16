"""
Utils for logging and for loading/saving checkpoints.
"""

from datetime import datetime
import logging
import os

import numpy as np
import torch

from ..rl.agent_random import AgentRandom
from ..rl.agent_ppo import AgentPPO
from ..rl.agent_rllib import AgentRLLib

# imports used dynamically
from ..svae.svae_core import SVAE, DSA


def load_checkpoint(checkpt_file, device, envs, args, logfile,
                    tb_writer=None, load_optim=True):
    checkpt_file = os.path.expanduser(checkpt_file)
    print("Loading chkpt {}...".format(checkpt_file))
    if not os.path.isfile(checkpt_file): return None, None, None, None, None
    logging.info("=> loading checkpoint '{}'".format(checkpt_file))
    chkpt = torch.load(checkpt_file, map_location=device)
    loaded_args = chkpt['args']; epoch = chkpt['epoch']
    loaded_args.device = device
    loaded_args.agent_device = device
    loaded_args.agent_replay_device = device
    if args is None: args = loaded_args
    optim_lst = []

    # Load unsup state and optimizer.
    use_args = args if hasattr(args, 'unsup_class') else loaded_args
    unsup = eval(use_args.unsup_class)(use_args, tb_writer)
    unsup.load_state_dict(chkpt['unsup_state_dict'], strict=False)
    unsup_optim = torch.optim.Adam(unsup.parameters(), lr=use_args.unsup_lr)
    if load_optim:
        unsup_optim.load_state_dict(chkpt['unsup_optim_state_dict'])
    else:
        epoch = 0  # re-start optimization
    optim_lst.append(unsup_optim)

    # Load RL.
    use_args = args if hasattr(args, 'agent_algo') else loaded_args
    agent_class = 'Agent'+use_args.agent_algo
    latent_sz = None
    if (hasattr(use_args, 'agent_use_unsup_latent') and
        use_args.agent_use_unsup_latent): latent_sz = unsup.latent_sz()
    agent = eval(agent_class)(
        envs, latent_sz, unsup.pr.hist, use_args.aux_sz,
        use_args.num_envs_per_worker, use_args.agent_max_replay_rollouts,
        use_args, logfile)
    if hasattr(agent, 'policy'):
        agent.policy.load_state_dict(chkpt['policy_state_dict'])
    if hasattr(agent, 'get_optimizer') and load_optim:
        agent_optim = agent.get_optimizer()
        if agent_optim is not None:
            agent_optim.load_state_dict(chkpt['agent_optim_state_dict'])
            optim_lst.append(agent_optim)

    if device != 'cpu':
        for optim in optim_lst:
            for state in unsup_optim.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v): state[k] = v.cuda()
    logging.info("Loaded chkpt '{}' (epoch {})".format(
        checkpt_file, chkpt['epoch']))
    return unsup, unsup_optim, agent, epoch, chkpt['args']


def init_logging(args):
    # ATTENTION: this function has to be called before tensorboardX import.
    env_name_parts = args.env_name.split('-v')
    env_basename =  env_name_parts[0]
    if env_basename.startswith('Aux'): env_basename = env_basename[3:]
    if env_basename.endswith('BulletEnv'):
        env_basename = env_basename[:-9]
    unsup_params_name = args.unsup_params_class
    if unsup_params_name.startswith('PARAMS_'):
        unsup_params_name = unsup_params_name[7:]
    date_str = datetime.strftime(datetime.today(), "%y%m%d_%H%M%S")
    dir_parts = [env_basename, args.agent_algo, unsup_params_name]
    if args.agent_algo != 'Random':
        dir_parts.append('r{:d}'.format(int(args.agent_replay_rnd_frac*100)))
    if hasattr(args, 'svae_kl_beta') and np.abs(args.svae_kl_beta-1.0)>0.001:
        dir_parts.append('kl{:d}'.format(int(args.svae_kl_beta)))
    if hasattr(args, 'ulo_loss_fxn') and args.ulo_loss_fxn != '':
        if args.ulo_checkpt is not None:
            assert(args.ulo_loss_fxn=='latent_transfer_loss')
            dir_parts.append('ulo{:0.2f}'.format(int(args.ulo_beta)))
        elif args.ulo_loss_fxn=='ground_truth_loss':
            dir_parts.append('ldgt{:0.2f}'.format(int(args.ulo_beta)))
        else:
            print('WARNING: unknown ulo_loss_fxn', args.ulo_loss_fxn)
            assert(False)  # unknown --ulo_loss_fxn
    dir_parts.append('subep'+str(args.unsup_num_sub_epochs))
    dir_parts.append('bsz'+str(args.unsup_batch_size))
    if args.svae_nolstm: dir_parts.append('nolstm')
    analyze=(args.analyze_path is not None)
    dir_parts.extend(
        ['seed'+str(args.seed), 'analyze' if analyze else 'output', date_str])
    save_path = '_'.join(dir_parts)
    save_path = os.path.join(os.path.expanduser(args.save_path_prefix), save_path)
    assert(not os.path.exists(save_path)); os.makedirs(save_path)
    checkpt_path = os.path.join(save_path, 'checkpt-%04d.pth' % 0)
    # python logging does not work well with torch multiprocessing
    #logging.basicConfig(
    #    level=logging.INFO, format="%(asctime)s %(message)s",
    #    handlers=[logging.FileHandler(os.path.join(save_path, 'log.txt')),
    #              logging.StreamHandler(sys.stdout)])
    return save_path, checkpt_path


def log_info(logfile, data_lst):
    tm = datetime.now().strftime('%H:%M:%S')
    for data in data_lst: print(tm, data); logfile.write(tm+' '+str(data)+'\n')


def do_logging(epoch, debug_dict, debug_hist_dict, logfile, tb_writer):
    dbg_str = 'Train epoch {:d}'.format(epoch)
    if 'recon_log_lik' in debug_dict.keys():
        dbg_str += ' recon_log_lik: {:.4f}'.format(debug_dict['recon_log_lik'])
    log_info(logfile, [dbg_str])
    if tb_writer is not None:
        for k,v in debug_dict.items():
            vv = v.mean().item() if type(v)==torch.Tensor else v
            tb_writer.add_scalar(k, vv, epoch)
        for k,v in debug_hist_dict.items():
            tb_writer.add_histogram(
                k,v.clone().cpu().data.numpy(), epoch)


def do_saving(agent, unsup, unsup_optim, args, epoch, logfile, remove_old=False):
    logging.info('Adding histograms to Tebsorboard')
    fbase = os.path.join(args.save_path, args.env_name+'_epoch')
    if remove_old:
        oldf = fbase+'{:d}.pt'.format(epoch-2*args.viz_interval)
        if os.path.exists(oldf): os.remove(oldf)
    checkpt_path = fbase+'{:d}.pt'.format(epoch)
    log_info(logfile, ['Saving {:s}'.format(checkpt_path)])
    save_dict = {'unsup_state_dict':unsup.state_dict(),
                 'unsup_optim_state_dict':unsup_optim.state_dict(),
                 'args':args, 'epoch':epoch}
    if hasattr(agent, 'policy'):
        save_dict['policy_state_dict'] = agent.policy.state_dict()
    agent_optim = agent.get_optimizer()
    if agent_optim is not None:
        save_dict['agent_optim_state_dict'] = agent_optim.state_dict()
    torch.save(save_dict, checkpt_path)
    log_info(logfile, ['do_saving() done'])


def save_png(inp, id):
    from PIL import Image
    from .env_utils import to_uint8
    if torch.is_tensor(inp): inp = inp.detach().cpu().numpy()
    print('save_png inp', inp.shape, inp[:2,:2,:2])
    if inp.shape[-1] > 4: inp = np.moveaxis(inp, 0, 2)
    if inp.dtype!=np.uint8: inp = to_uint8(inp)
    im = Image.fromarray(inp, mode='RGB')
    im.save('tmp'+str(id)+'.png')
