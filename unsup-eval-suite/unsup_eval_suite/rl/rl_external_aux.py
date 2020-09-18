"""
Common utilities for launching external RL training.
"""
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="RLLib")
    parser.add_argument(
        '--rl_algo', type=str, default='PPO',
        choices=[
            'PPO', 'A3C',  # on-policy
            'DDPG', 'HER', 'SAC', 'TD3', 'ApexDDPG',  # off-policy
            'Impala'  # mixed
        ])
    parser.add_argument('--env_name', type=str,
                        default='AuxCartPoleBulletEnv-v1',
                        help='Environment name')
    parser.add_argument('--gpus', type=str, default=None, help='GPU IDs')
    parser.add_argument('--ncpus', type=int, default=4, help='Number of CPUs')
    parser.add_argument('--num_envs_per_worker', type=int, default=1,
                        help='Number envs per parallel worker')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for model saving')
    parser.add_argument('--use_pytorch', action='store_true',
                        help='Whether to use pytorch models')
    parser.add_argument('--rollout_len', type=int, default=200,
                        help='Rollout length')
    parser.add_argument('--hidden_layer_sizes', type=int, default=(64, 64),
                        help='List of hidden layer sizes')
    parser.add_argument('--rl_lr', type=float, default=1e-4,
                        help='RL learning rate')
    parser.add_argument('--max_train_steps', type=int, default=int(1e8),
                        help='Maximum number of training steps')
    parser.add_argument('--play', type=int, default=None,
                        help='Play this many episodes, then exit (no training)')
    parser.add_argument('--load_checkpt', type=str, default=None,
                        help='Load checkpoint to play or continue training')
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Forces RLlib to run on single process, so IDEs can debug with breakpoints')
    args, unknown = parser.parse_known_args()
    return args
