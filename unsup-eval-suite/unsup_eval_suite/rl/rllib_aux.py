"""
A simple script to launch RLLib.

Install dependencies:
pip install ray[rllib] torch tensorflow-gpu

CD to the directory with the RLLib script:
cd unsup-eval-suite/unsup_eval_suite/rl

Run RLLib training:
python rllib_aux.py --env_name=AuxCartPoleBulletEnv-v1

Note: this script only runs low-dim version of Aux*BulletEnv-v*
But it can be easily extended to fun RGB and ptcloud versions.

"""
import argparse
import glob
import io
import os
import pickle

import torch

import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.rollout import rollout
from ray.rllib.agents import sac, ppo, impala, a3c
from ray.rllib.agents.ddpg import apex, td3
from ray.tune.suggest.variant_generator import grid_search
#from ray.rllib.models import ModelCatalog

try:
    from .rl_external_aux import get_args
except Exception as e:  # alternative loading
    from rl_external_aux import get_args

import gym_bullet_aux  # to register envs
from gym_bullet_aux.envs import AuxBulletEnv


# https://ray.readthedocs.io/en/latest/rllib-env.html
class RllibAuxBulletEnv(AuxBulletEnv):
    def __init__(self, env_config):
        super(RllibAuxBulletEnv, self).__init__(**env_config)


def env_creator(env_config):
    return RllibAuxBulletEnv(env_config)


def guess_checkpt(rl_algo, env_name):
    # Try to guess checkpoint path
    pfx = os.path.expanduser('~/ray_results/')
    if 'Viz' in env_name:
        parts = env_name.split('Viz')
        env_name = parts[0]+parts[1]
    if rl_algo=='ApexDDPG': rl_algo = 'APEX_DDPG'
    data_dir = pfx+rl_algo+'/'+rl_algo+'_'+env_name+'_*/'
    pth = data_dir+'checkpoint_*'
    options = glob.glob(pth)
    print('pth', pth, 'options', options)
    iter = max([int(res.split('_')[-1]) for res in options])
    load_pth = data_dir+'checkpoint_'+str(iter)+'/checkpoint-'+str(iter)
    print('Guessed path', load_pth)
    load_option = glob.glob(load_pth)
    assert(len(load_option)==1)
    return load_option[0]


def make_rl_config(args, num_gpus):
    # https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/walker2d-ppo.yaml
    if args.rl_algo=='ApexDDPG':
        rl_config = apex.APEX_DDPG_DEFAULT_CONFIG.copy()
    elif args.rl_algo=='TD3':
        # https://github.com/ray-project/ray/blob/master/rllib/agents/ddpg/td3.py
        rl_config = td3.TD3_DEFAULT_CONFIG.copy()
    else:
        rl_config = eval(args.rl_algo.lower()).DEFAULT_CONFIG.copy()
    env_parts = args.env_name.split('-v')
    assert(len(env_parts)==2)  # EnvName-v0
    assert(env_parts[0].startswith('Aux'))
    assert(env_parts[0].endswith('BulletEnv'))
    base_env_name = env_parts[0][len('Aux'):-len('BulletEnv')]
    rl_config['env_config'] = {
        #'max_episode_len': 200,
        'base_env_name': base_env_name,
        'env_v': int(env_parts[1]),
        'obs_resolution': None,
        'obs_torch_float_format': args.use_pytorch,
        'visualize': 'Viz' in args.env_name,
        'debug_level': 'Debug' in args.env_name
    }
    bsz = args.rollout_len*args.ncpus
    # train_batch_size is an awkward name
    # https://github.com/ray-project/ray/issues/4628
    rl_config['train_batch_size'] = bsz
    rl_config['rollout_fragment_length'] = args.rollout_len  # aka sample_batch_size
    rl_config['soft_horizon'] = True  # don't reset episode after rollout
    rl_config['num_gpus'] = num_gpus
    rl_config['num_workers'] = args.ncpus
    rl_config['num_envs_per_worker'] = 1 if args.play else args.num_envs_per_worker
    rl_config['env'] = args.env_name
    rl_config['lr'] = args.rl_lr
    rl_config['clip_rewards'] = None
    rl_config['gamma'] = 0.995
    #rl_config['horizon'] = rollout_len  # seems to break things
    # Customize NN architecture and hidden layers.
    rl_config['model']['fcnet_activation'] = 'tanh'
    rl_config['model']['fcnet_hiddens'] = [*args.hidden_layer_sizes]
    # rl_config['model']['custom_model_config'] = {
    #                 'fcnet_activation': 'ReLU',
    #                 'fcnet_hiddens':[1024,512,256],
    #                 'no_final_linear':False,
    #                 'vf_share_layers': True,
    #                 'free_log_std':False,
    #             }
    if args.use_pytorch: rl_config['framework'] = 'torch'
    if args.rl_algo=='A3C' and args.use_pytorch:
        rl_config['sample_async'] = False
    if args.rl_algo=='PPO':
        rl_config['kl_coeff'] = 1.0
        rl_config['num_sgd_iter'] = 100
        rl_config['sgd_minibatch_size'] = bsz//10
        #rl_config['vf_share_layers'] = True
        rl_config['entropy_coeff'] = 0.01   # low exploration noise
    if args.rl_algo=='Impala':
        rl_config['num_sgd_iter'] = 50
        rl_config['replay_proportion'] = 0.5  # 0.5:1 proportion
        rl_config['replay_buffer_num_slots'] = 10000
    if args.rl_algo=='ApexDDPG':
        rl_config['learning_starts'] = args.rollout_len
        rl_config['target_network_update_freq'] = 2*args.rollout_len
        rl_config['timesteps_per_iteration'] = 2*args.rollout_len
    if args.rl_algo=='SAC':
        # https://github.com/ray-project/ray/tree/master/rllib/tuned_examples/sac
        rl_config['train_batch_size'] = 32
        rl_config['no_done_at_end'] = False
        rl_config['target_network_update_freq'] = 32
        rl_config['gamma'] = 0.95
        rl_config['tau'] = 1.0
        opt = rl_config['optimization']
        opt['actor_learning_rate'] = 5e-3    # args.rl_lr
        opt['critic_learning_rate'] = 5e-3   # args.rl_lr
        opt['entropy_learning_rate'] = 1e-4  # args.rl_lr
    return rl_config


def run_with_aux(args):
    ray.tune.registry.register_env(args.env_name, env_creator)
    num_gpus = 0
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
        num_gpus = len(args.gpus.split(','))
    ray.init(num_cpus=args.ncpus+1, num_gpus=num_gpus, local_mode=args.debug)
    rl_config = make_rl_config(args, num_gpus)
    # Play if requested.
    if args.play is not None:
        rl_config['num_workers'] = 0
        if args.load_checkpt is None:
            args.load_checkpt = guess_checkpt(args.rl_algo, args.env_name)
        print('Loading checkpoint', args.load_checkpt)
        play_algo = 'APEX_DDPG' if args.rl_algo=='ApexDDPG' else args.rl_algo
        cls = get_agent_class(play_algo)
        agent = cls(env=args.env_name, config=rl_config)
        agent.restore(os.path.expanduser(args.load_checkpt))
        rollout(agent, args.env_name, num_episodes=args.play,
                num_steps=1000, no_render=True)
        return  # just playing
    # Run training.

    if args.rl_algo=='ApexDDPG':
        rl_trainer_class = apex.ApexDDPGTrainer
    elif args.rl_algo=='TD3':
        rl_trainer_class = td3.TD3Trainer
    else:
        rl_trainer_class = eval(args.rl_algo.lower()+'.'+args.rl_algo+'Trainer')
    ray.tune.run(rl_trainer_class, config=rl_config,
                 checkpoint_freq=args.save_interval,
                 restore=args.load_checkpt, reuse_actors=True)
    #rl_trainer = rl_trainer_class(
    #    env=args.env_name, config=rl_config, checkpoint_freq=args.save_interval)
    #rl_trainer.restore(args.load_checkpt)
    #while True: print(rl_trainer.train())


if __name__ == "__main__":
    run_with_aux(get_args())
