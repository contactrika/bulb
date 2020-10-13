"""
A simple example of training with model-free RL from stable baselines.

pip install stable-baselines3
python -m bulb.rl_demo --env_name=AntBulletEnvLD-v0 --viz
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


def get_args():
    parser = argparse.ArgumentParser(description="EnvDemo")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--env_name', type=str,
                        default='AntBulletEnvLD-v0', help='Env name')
    parser.add_argument('--num_envs', type=int, default=4,
                        help='Number of simulation envs')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--steps_per_epoch', type=int, default=int(10e3),
                        help='Number of steps per epoch')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize in PyBullet simulator')
    args = parser.parse_args()
    return args


def main(args):
    envs = make_vec_env(args.env_name, n_envs=args.num_envs,
                        vec_env_cls=SubprocVecEnv)
    viz_env = None
    if args.viz:
        nm_core, nm_vrsn,  = args.env_name.split('-')
        nm_core += 'Viz' if args.viz else 'Dbg' if args.debug else ''
        viz_env = make_vec_env(nm_core+'-'+nm_vrsn, n_envs=1)
    rl_learner = PPO(
        'MlpPolicy', envs, verbose=1, seed=args.seed, device='cpu')
    for epoch in range(args.num_epochs):
        rl_learner.learn(args.steps_per_epoch)
        if args.viz:
            obs = viz_env.reset(); done = False
            while not done:
                act, _ = rl_learner.predict(obs)
                if len(act.shape) > len(viz_env.action_space.shape):
                    act = act[0:1]  # just one viz env
                obs, rwd, done, _ = viz_env.step(act)
                time.sleep(0.01)  # to make motions visible


if __name__ == "__main__":
    main(get_args())
