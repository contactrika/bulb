"""
A simple example of training with model-free RL from stable baselines.

pip install torch stable-baselines3 tensorboardX
python -m bulb.rl_demo --env_name=InvertedPendulumBulletEnvLD-v0 --viz
"""

import argparse
import sys
import time

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import gym
import pybullet

from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

import bulb  # to register envs
from .utils.torch_init_and_log import (
    get_logger2_args, init_gpus_and_randomness, Logger2)


def get_args(base_parser):
    parser = argparse.ArgumentParser(
        description='RLDemo', parents=[base_parser], add_help=True)
    parser.add_argument('--env_name', type=str,
                        default='InvertedPendulumBulletEnvLD-v0',
                        help='Env name')
    parser.add_argument('--num_envs', type=int, default=4,
                        help='Number of simulation envs')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=int(10e3),
                        help='Number of steps per epoch')
    parser.add_argument('--debug', action='store_true',
                        help='Print debug messages')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize in PyBullet simulator')
    args = parser.parse_args()
    return args


def main():
    base_args, base_parser = get_logger2_args()
    args = get_args(base_parser)
    args.device = init_gpus_and_randomness(args.seed, args.gpu)
    logger = Logger2('/tmp/tmp', use_tensorboardX=True)
    logger.log_tb_object(args, 'args')
    envs = make_vec_env(args.env_name, n_envs=args.num_envs,
                        vec_env_cls=SubprocVecEnv)
    viz_env = None
    if args.visualize:
        nm_core, nm_vrsn,  = args.env_name.split('-')
        nm_core += 'Viz' if args.visualize else 'Dbg' if args.debug else ''
        viz_env = make_vec_env(nm_core+'-'+nm_vrsn, n_envs=1)
    rl_learner = PPO(
        'MlpPolicy', envs, verbose=1, seed=args.seed, device='cpu')
    for epoch in range(args.num_epochs):
        rl_learner.learn(args.steps_per_epoch)
        if args.visualize:
            obs = viz_env.reset(); done = False
            while not done:
                act, _ = rl_learner.predict(obs)
                if len(act.shape) > len(viz_env.action_space.shape):
                    act = act[0:1]  # just one viz env
                obs, rwd, done, _ = viz_env.step(act)
                time.sleep(0.01)  # to make motions visible


if __name__ == "__main__":
    main()
