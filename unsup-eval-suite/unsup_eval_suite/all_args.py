"""
All command-line arguments.
"""

import argparse
import os

import torch

from .svae.svae_params import get_parser as svae_get_parser


def get_all_args():
    svae_prsr =  svae_get_parser()
    parser = argparse.ArgumentParser(
        description="ALL", parents=[svae_prsr])
    parser.add_argument('--save_path_prefix', type=str, default='./output/')
    parser.add_argument('--load_checkpt', type=str, default=None,
                        help='Directory with SVAE and RL checkpoint files')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (start)'
                        'Will use num_workers GPUs starting with this ID')
    parser.add_argument('--log_interval', type=int, default=None,
                        help='Interval for log messages (None for auto)')
    parser.add_argument('--viz_interval', type=int, default=None,
                        help='Interval for visualization (None for auto)')
    parser.add_argument('--total_env_steps', type=int, default=256000000,
                        help='Total env steps (256M)')
    parser.add_argument('--num_grad_workers', type=int, default=1,
                        help='Number of parallel GPU (or CPU) workers')
    parser.add_argument('--max_epochs', type=int, default=100000,  # large dflt
                        help='Maximum number of training epochs')
    parser.add_argument('--analyze_path', type=str, default=None,
                        help='Do not train, only run analysis')
    # Arguments for alignment analysis.
    parser.add_argument('--analyze_interval', type=int, default=None,
                        help='Interval for running analysis during training'
                             ' (use None to set automatically')
    parser.add_argument('--analyze_max_iter', type=int, default=10000,     # 10K
                        help='Max iters for analysis training')
    parser.add_argument('--analyze_lr', type=float, default=1e-4,
                        help='Learning rate for analyze net')
    parser.add_argument('--analyze_hsz', type=int, default=256,
                        help='Learning rate for analyze net')
    parser.add_argument('--analyze_dataset_size', type=int, default=10000, # 10K
                        help='Get this many points for analysis training')
    parser.add_argument('--ulo_checkpt', type=str, default=None,
                        help='ULO checkpoint file')
    parser.add_argument('--ulo_beta', type=float, default=1e3,
                        help='ULO loss weight (~1e3 because g outs are small)')
    parser.add_argument('--ulo_loss_fxn', type=str, default='',
                        choices=['', 'ground_truth_loss', 'latent_transfer_loss'],
                        help='Funciton name for ulo loss')
    # Arguments for unsupervised learner.
    parser.add_argument('--unsup_class', type=str, default='SVAE',
                        choices=['SVAE', 'DSA'],
                        help='Class name for the unsupervised learner class')
    parser.add_argument('--unsup_lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--unsup_batch_size', type=int, default=1024,
                        help='SVAE batch size')
    parser.add_argument('--unsup_num_sub_epochs', type=int, default=None,  # auto
                        help='SVAE number of sup epochs')
    parser.add_argument('--unsup_params_class', type=str,
                        default='PARAMS_VAE',
                        help='Class name from [unsup_algo]/*_params.py')
    # Arguments for the agent (e.g. RL actor+learner, MPC actor, random actor).
    parser.add_argument('--agent_algo', type=str, default='Random',
                        choices=['Random', 'PPO'],
                        help='Short name for RL algo')
    parser.add_argument('--agent_rollout_len', type=int, default=None,  # auto
                        help='Length of RL rollouts')
    parser.add_argument('--agent_max_replay_rollouts', type=int, default=5000,
                        help='Maximum number of replay rollouts to store')
    parser.add_argument('--agent_replay_rnd_frac', type=float, default=0.5,
                        help='Fraction of frames from random policy in replay')
    parser.add_argument('--agent_device', type=str, default=None,
                        help='Device for RL agent')
    parser.add_argument('--agent_replay_device', type=str, default='cpu',
                        help='Device for RL replay buffer')
    parser.add_argument('--agent_use_unsup_latent', action='store_true',
                        help='Whether to use output of unsup for RL state')
    # ENV arguments.
    parser.add_argument('--env_name', type=str,
                        default='AuxReacherBulletEnvViz-v0',
                        help='Environment name')
    parser.add_argument('--num_envs_per_worker', type=int, default=64,
                        help='Number envs per parallel worker')
    parser.add_argument('--env_skin_id', type=int, default=0,
                        help='E.g. statics/appearance ID for the env')
    parser.add_argument('--env_dynamics_id', type=int, default=0,
                        help='E.g. dynamics ID for the env')
    parser.add_argument('--any_env_skin', type=int, default=1,
                        help='Select any of env skins.')
    args = parser.parse_args()
    return args
