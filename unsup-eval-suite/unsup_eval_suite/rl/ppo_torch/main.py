#
# Code adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
#

from collections import deque
from datetime import datetime
import os
import time
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .a2c_ppo_acktr import algo, utils
from .a2c_ppo_acktr.algo import gail
from .a2c_ppo_acktr.arguments import get_args
from .a2c_ppo_acktr.envs import make_vec_envs
from .a2c_ppo_acktr.model import Policy, CNNBase64
from .a2c_ppo_acktr.storage import RolloutStorage
from .evaluation import evaluate


def main(rank, num_workers, args, save_chkpt):
    if rank==0:
        logfile = open(os.path.join(os.path.dirname(save_chkpt), 'log.txt'),
                       'w', buffering=1)
        logfile.write(str(args))
    if num_workers>1:
        # https://tutorials.pytorch.kr/intermediate/ddp_tutorial.html
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        os.environ['WORLD_SIZE'] = str(num_workers)
        os.environ['RANK'] = str(rank)
        # TODO: here we initialize MPI, GLOO or other multiproc backend.
        dist.init_process_group(backend='gloo', world_size=args.num_processes)
        rank = dist.get_rank(); num_workers = dist.get_world_size()
        print('Starting worker {} out of {}'.format(rank, num_workers))

    # Explicitly setting seed helps ensure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = 'cpu'
    if args.cuda:
        device = torch.device("cuda:{}".format(rank%torch.cuda.device_count()))

    envs = make_vec_envs(args.env_name, args, args.seed,
                         args.num_envs_per_worker, args.gamma, device, False)

    if args.load_chkpt is not None:
        load_path = os.path.expanduser(args.load_chkpt)
        if rank==0: print('Continue training from {}'.format(load_path))
        # TODO: we are not using normalization, so discard second arg
        actor_critic, _ = torch.load(load_path,
                                     map_location=lambda storage, loc: storage)
    else:
        actor_critic = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base=CNNBase64 if args.env_name.startswith('Coin') else None,
            base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            use_clipped_value_loss=True,
            worker_id=rank,num_workers=num_workers)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))

        gail_train_loader = torch.utils.data.DataLoader(
            gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20),
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=True)

    rollouts = RolloutStorage(args.num_steps, args.num_envs_per_worker,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        # TODO: making this exactly like OpenAI w/ PPO
        # nsteps = Config.NUM_STEPS = 256; total_timesteps = int(256e6)
        # nenvs = envs.num_envs = 32 (or 96)
        # nbatch = nenvs * nsteps
        # nupdates = total_timesteps//nbatch
        if hasattr(agent, 'frac_train_remaining'):
            agent.frac_train_remaining = 1.0 - (j-1.0)/num_updates

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if rank==0 and (j%args.save_interval==0 or j==num_updates-1):
            torch.save([actor_critic,
                        getattr(utils.get_vec_normalize(envs), 'ob_rms', None)],
                       save_chkpt+'_epoch'+str(j)+'.pt')

        if rank==0 and j%args.log_interval==0 and len(episode_rewards)>1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            if rank==0:
                msg = 'Worker0 updates {} num timesteps {} FPS {} \n'
                msg += 'Last {} training episodes: mean|median '
                msg += 'reward {:.1f} | {:.1f} min|max {:.1f} | {:.1f}\n'
                print_str = msg.format(
                    j, total_num_steps, int(total_num_steps/(end-start)),
                    len(episode_rewards), np.mean(episode_rewards),
                    np.median(episode_rewards), np.min(episode_rewards),
                    np.max(episode_rewards), dist_entropy, value_loss, 
                    action_loss)
                print(print_str)
                logfile.write(print_str)

        if (rank==0 and args.eval_interval is not None and
            len(episode_rewards)>1 and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms
            evaluate(actor_critic, ob_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)
    # Cleanup
    if rank==0: logfile.close()
    if torch.distributed.is_available():
        dist.destroy_process_group()  # cleanup


if __name__ == "__main__":
    args = get_args()
    save_sfx = '_' + datetime.strftime(datetime.today(), "%y%m%d_%H%M%S")
    save_dir = os.path.expanduser(args.save_dir)+save_sfx
    save_chkpt = os.path.join(save_dir, args.algo+'_'+args.env_name)
    try:
        os.makedirs(save_dir)
    except OSError:
        pass
    # Start main.
    if torch.distributed.is_available() and args.num_processes>1:
        mp.spawn(main, args=(args.num_processes,args,save_chkpt,),
                 nprocs=args.num_processes, join=True)
    else:
        main(rank=0, args=args, num_workers=1, save_chkpt=save_chkpt)
