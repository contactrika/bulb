"""
A simple demo for env (with random actions).

python -m bulb.env_demo --env_name=AntBulletEnvLD-v0 --viz --debug
python -m bulb.env_demo --env_name=ReacherRearrangeGeomIM64-v0
"""

import argparse
import logging
import sys
import time

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import gym
import pybullet

import bulb  # to register envs
from .envs.rearrange_utils import test_override


def play(env, num_episodes, debug, viz):
    for epsd in range(num_episodes):
        print('------------ Play episode ', epsd, '------------------')
        obs = env.reset()
        step = 0
        #input('Reset done; press enter to start episode')
        while True:
            # Action is random in this demo.
            if isinstance(env.action_space, gym.spaces.Discrete):
                act = np.random.randint(env.action_space.n)
            else:
                act = np.random.rand(*env.action_space.shape)  # in [0,1]
                rng = env.action_space.high - env.action_space.low
                act = act*rng + env.action_space.low
            next_obs, rwd, done, info = env.step(act)
            if viz: time.sleep(0.02)
            if debug and step%(env.max_episode_steps//10)==0:
                env.render_obs(debug_out_dir='/tmp/')
            if done: break
            step += 1
        #input('Episode ended; press enter to go on')


def get_args():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)])

    parser = argparse.ArgumentParser(description="EnvDemo")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--env_name', type=str,
                        default='AntBulletEnvLD-v0', help='Env name')
    parser.add_argument('--num_episodes', type=int, default=22,
                        help='Number of episodes')
    parser.add_argument('--debug', action='store_true',
                        help='Whether to print debug info')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize in PyBullet simulator')
    args = parser.parse_args()
    return args


def main(args):
    assert('-v' in args.env_name)  # specify env version
    nm_core, nm_vrsn,  = args.env_name.split('-')
    nm_core += 'Viz' if args.viz else 'Dbg' if args.debug else ''
    env = gym.make(nm_core+'-'+nm_vrsn); env.seed(args.seed)
    print('Created ', args.env_name, 'with observation_space',
          env.observation_space.shape, 'action_space', env.action_space.shape,
          'max_episode_steps', env.max_episode_steps)
    play(env, args.num_episodes, args.debug, args.viz)
    env.close()


if __name__ == "__main__":
    main(get_args())
