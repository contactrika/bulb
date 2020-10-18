"""
A unified interface for a set of standard PyBullet gym/robotschool envs that
reports high-dimensional observations (images, point clouds) and also adds
low-dimensional state (joint angles), following AuxEnv API.
"""
import os

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import gym

import pybullet
from pybullet_envs.bullet.kukaGymEnv import KukaGymEnv

from ..utils import render_utils
from .aux_env import AuxEnv


class AuxBulletEnv(gym.Env, AuxEnv):
    PTCLOUD_BOX_SIZE = 2.0  # 2m cube cut off for point cloud observations

    def __init__(self, base_env_name, env_v=0,
                 obs_resolution=64, obs_ptcloud=False, random_colors=False,
                 debug=False, visualize=False):
        self._base_env_name = base_env_name
        self._random_colors = random_colors
        self._mobile = False
        if base_env_name.startswith(('Hopper', 'Walker2D', 'Ant', 'Humanoid',
                                     'HalfCheetah', 'InvertedPendulumSwingup')):
            self._mobile = True  # update base pos when rendering
        # If we created a bullet sim env from scratch, would init sim as:
        # conn_mode = pybullet.GUI if visualize else pybullet.DIRECT
        # self._p = bullet_client.BulletClient(connection_mode=conn_mode)
        # But all bullet gym envs init bullet client in constructor or reset.
        # Kuka creates bullet client in the constructor, so need to call
        # its constructor explicitly. All other envs derive from classes in
        # pybullet_envs.env_bases, which init bullet client in reset().
        # We set the corresponding render flag for them after their creation.
        if 'Kuka' in base_env_name:
            max_episode_steps = 1000
            self._env = KukaGymEnv(renders=visualize, maxSteps=max_episode_steps)
            self._env._max_episode_steps = max_episode_steps
            # Appropriate camera vals from KukaCamGymEnv
            self._base_pos = [0.52, -0.2, -0.33]
            self._env.unwrapped._cam_dist = 1.3
            self._env.unwrapped._cam_yaw = 180
            self._env.unwrapped._cam_pitch = -41
        else:
            self._env = gym.make(base_env_name+'BulletEnv-v'+str(env_v))
            max_episode_steps = self._env._max_episode_steps
            if 'CartPole' in base_env_name:
                self._env.render(mode='rgb_array')  # init cam dist vars
            if visualize: self._env.render(mode='human')  # turn on debug viz
        super(AuxBulletEnv, self).__init__(
            max_episode_steps, obs_resolution, obs_ptcloud, debug, visualize)
        # Zoom in camera.
        assert(hasattr(self._env.unwrapped, '_cam_dist'))
        if base_env_name.startswith('Reacher'):
            self._env.unwrapped._cam_dist = 0.55
            self._env.unwrapped._cam_pitch = -89  # -90 doesn't look ok on debug
        elif base_env_name.startswith('InvertedPendulum'):
            self._env.unwrapped._cam_dist = 2.2  # was: 1.8
        elif base_env_name.startswith('InvertedDoublePendulum'):
            self._env.unwrapped._cam_dist = 2.8
            self._env.unwrapped._cam_pitch = -1.0
            self._part_nms = ['cart', 'pole', 'pole2']
        elif base_env_name.startswith(('Hopper', 'Walker2D')):
            self._env.unwrapped._cam_dist = 1.5
            self._env.unwrapped._cam_pitch = -40
        elif base_env_name.startswith('Humanoid'):
            self._env.unwrapped._cam_dist = 1.8
            self._env.unwrapped._cam_pitch = -40
        self._env.reset()  # load URDF files, init _p sim pointer
        self._sim = self._env.unwrapped._p  # easy access to bullet client
        # Set reasonable colors and env defaults.
        if 'CartPole' in base_env_name:
            cartpole = self._env.unwrapped.cartpole
            self._sim.changeVisualShape(cartpole, 0, rgbaColor=(1,1,0,1))
            self._sim.changeVisualShape(cartpole, 1, rgbaColor=(0,0,1,1))
        # Compute camera info. This has to be done after self._env.reset()
        self._view_mat, self._proj_mat, base_pos = self.compute_cam_vals()
        # Specify camera objects for point cloud processing.
        self._cam_object_ids = None
        if self.obs_ptcloud:
            assert(self._base_env_name == 'CartPole')
            self._cam_object_ids = [self._env.unwrapped.cartpole]
            # Note: PyBullet envs with XML(MJCF)-based robots seem to have
            # issues with reporting body ids and names correctly.
            #for tmpi in range(self._sim.getNumBodies()):
            #    robot_name, scene_name = self._sim.getBodyInfo(tmpi)
            #    robot_name = robot_name.decode("utf8")
            #    body_id = self._sim.getBodyUniqueId(tmpi)
            #    print('body id:', body_id, robot_name, scene_name)
        # Specify observation and action spaces.
        self._ld_names = self.compute_low_dim_state_names()
        self._ld_names_ids_dict = \
            {k: v for v, k in enumerate(self.low_dim_state_names)}
        if obs_resolution is None:
            self.observation_space = self._env.observation_space  # low dim
        elif self.obs_ptcloud:
            state_sz = obs_resolution*3  # 3D points in point cloud
            self.observation_space = gym.spaces.Box(
                -1.0*AuxBulletEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz),
                AuxBulletEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz))
        else:  # RGB images
            self.observation_space = gym.spaces.Box(
                low=0.0, high=1.0,
                shape=[3, obs_resolution, obs_resolution],  # HxWxRGB
                dtype=np.float32)
        self.action_space = self._env.action_space
        # Turn on visualization if needed.
        if visualize:
            self._sim.resetDebugVisualizerCamera(
                self._env.unwrapped._cam_dist, self._env.unwrapped._cam_yaw,
                self._env.unwrapped._cam_pitch, base_pos)
            if self.obs_resolution is not None:
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,1)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,1)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,1)
        if self.debug:
            print('Created Aux', base_env_name, 'with observation_space',
                  self.observation_space, 'action_space', self.action_space,
                  'max_episode_steps', self.max_episode_steps)

    @property
    def low_dim_state_space(self):
        return self._env.observation_space

    @property
    def low_dim_state_names(self):
        return self._ld_names

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        self._env.unwrapped.close()

    def reset(self):
        self.reset_aggregators()
        if self._random_colors: self.reset_colors()
        state = self._env.reset()
        if self.obs_resolution is None:  # low dim state
            obs = np.clip(state, self.observation_space.low,
                          self.observation_space.high)
        else:  # pixel or ptcloud state
            obs = self.render_obs()
        # For mobile envs: could look at the robot from different angles.
        #if self._mobile:
        #    self._env.unwrapped._cam_yaw = (np.random.rand()-0.5)*360
        if self.visualize and self._mobile:
            self._sim.resetDebugVisualizerCamera(
                self._env.unwrapped._cam_dist, self._env.unwrapped._cam_yaw,
                self._env.unwrapped._cam_pitch, self.get_base_pos())
        return obs

    def step(self, action):
        state, rwd, done, info = self._env.step(action)  # apply action
        if self.obs_resolution is None:
            obs = np.clip(state, self.observation_space.low,
                          self.observation_space.high)
        else:
            obs = self.render_obs()
        done, more_info = self.update_aggregators(rwd, done)
        info.update(more_info)
        if self.debug:  # print low-dim state
            msg = f'pid {os.getpid():d} step {self.stepnum:d} act'
            msg += np.array2string(action, precision=2,
                                   formatter={'float_kind':'{:0.2f}'.format}, )
            for i in range(len(self.low_dim_state_names)):
                msg += f' {self.low_dim_state_names[i]:s} {state[i]:0.4f}'
            print(msg)
        #if self.debug:
            # sanity check: aux should be same as low-dim state reported by env
            # except for envs that modify state after applying action (Hopper).
            #low_dim_state = self.compute_low_dim_state()
            #if np.abs(state - low_dim_state).sum()>1e-6:
            #    print('        state', state)
            #    print('low_dim_state', low_dim_state)
            #    assert(False)
        if self.visualize and self._mobile:
            self._sim.resetDebugVisualizerCamera(
            self._env.unwrapped._cam_dist, self._env.unwrapped._cam_yaw,
            self._env.unwrapped._cam_pitch, self.get_base_pos())
        return obs, rwd, done, info

    def render(self, mode='rgb_array'):
        pass  #  done automatically with PyBullet visualization

    def render_obs(self, override_resolution=None, debug_out_dir=None):
        if self._mobile and not self.obs_ptcloud:
            self._view_mat, self._proj_mat, _ = self.compute_cam_vals()
        obs = render_utils.render_obs(
            self._sim, self.stepnum,
            self.obs_resolution, override_resolution,
            self._env.unwrapped._cam_dist, self.get_base_pos(),
            self._env.unwrapped._cam_pitch, self._env.unwrapped._cam_yaw,
            self.obs_ptcloud, self._cam_object_ids, self._base_env_name,
            AuxBulletEnv.PTCLOUD_BOX_SIZE,
            debug_out_dir, self._view_mat, self._proj_mat)
        return obs

    def normalize_reward(self, rwd):
        # This could be a useful function for algorithms that expect episode
        # rewards to be closer to a small range, like [0,1] or [-1,1].
        #
        # Rewards in [0,1] for CartPole, InvertedPendulum
        # Rewards close to [-?,10] for InvertedDoublePendulum
        # (bonus for 'alive', penalty for dist from vertical)
        # Rewards close to [-1,1] for InvertedPendulumSwingup
        #
        # Locomotion envs have several non-trivial components contributing to
        # the reward. It still might make sense to divide by total number of
        # steps, but the resulting range is not that simple to compute.
        #
        return rwd / self.max_episode_steps

    def compute_low_dim_state_names(self):
        if 'CartPole' in self._base_env_name:
            nms = ['theta', 'theta_vel', 'x', 'x_vel']
        elif 'InvertedPendulum' in self._base_env_name:
            nms = ['x', 'x_vel', 'theta_cos', 'theta_sin', 'theta_vel']
        elif 'InvertedDoublePendulum' in self._base_env_name:
            nms = ['x', 'x_vel', 'elbow_x',
                   'theta_cos', 'theta_sin', 'theta_vel',
                   'theta1_cos', 'theta1_sin', 'theta1_vel']
        elif self._base_env_name.startswith(
            ('Hopper', 'Walker', 'HalfCheetah', 'Ant', 'Humanoid')):
            # Note: _vel are scaled by 0.3 in these envs to be in ~[-1,1];
            # the reported state is clipped to [-5,5]
            nms = ['z_delta', 'tgt_theta_sin', 'tgt_theta_cos',
                   'x_vel', 'y_vel', 'z_vel', 'body_roll', 'body_pitch']
            njoints = len(self._env.unwrapped.robot.ordered_joints)
            for j in range(njoints):
                pfx='j'+str(j); nms.append(pfx+'_pos'); nms.append(pfx+'_vel')
            nfeet = len(self._env.unwrapped.robot.foot_list)
            for ft in range(nfeet):
                nms.append('ft'+str(ft)+'_contact')
        elif 'Reacher' in self._base_env_name:
            nms = ['tgt_x', 'tgt_y', 'to_tgt_vec_x', 'to_tgt_vec_y',
                   'theta_cos', 'theta_sin', 'theta_vel',
                   'theta1', 'theta1_vel']
        elif 'Kuka' in self._base_env_name:
            nms = ['grp_x', 'grp_y', 'grp_z', 'grp_r', 'grp_p', 'grp_y',
                   'blk_rel_x', 'blk_rel_y', 'blk_rel_eulz']
        else:
            assert(False), f'Unknown BulletEnv {self._base_env_name:s}'
        return nms

    def compute_low_dim_state(self):
        # Get low-dim state. Since there is no unified interface for bullet
        # gym envs to get the low-dim state, here handle special cases.
        # Code comes from step() method of the underlying bullet gym envs.
        # Note: some envs modify state within step() call, so this function
        # is only useful for getting initial low dim state (e.g. after reset).
        if 'CartPole' in self._base_env_name:
            # state = theta, theta_dot, x, x_dot
            cartpole = self._env.unwrapped.cartpole
            state = list(self._sim.getJointState(cartpole, 1)[0:2])
            state.extend(self._sim.getJointState(cartpole, 0)[0:2])
        elif self._base_env_name.startswith('Kuka'):
            # Kuka state = grp_pos, grp_euler, block_rel_pos
            # pos,euler are 3D; block_rel_pos is XYEulZ:
            # relative x,y position and euler angle of block in gripper space
            # Racecar state = ball_rel_x, ball_rel_y
            # Racecar env is a emulation of the MIT RC Racecar: reward is based
            # on distance to the randomly placed ball; observations are ball
            # position (x,y) in camera frame; camera follows car's body frame.
            state = self._env.unwrapped.getExtendedObservation()
        else:  # envs in pybullet_envs.gym_*_envs.py use calc_state()
            state = self._env.unwrapped.robot.calc_state()
        return np.array(state)

    def compute_cam_vals(self):
        base_pos = self.get_base_pos()
        view_matrix = self._sim.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._env.unwrapped._cam_dist,
            yaw=self._env.unwrapped._cam_yaw,
            pitch=self._env.unwrapped._cam_pitch, roll=0, upAxisIndex=2)
        proj_matrix = self._sim.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=100.0)
        return view_matrix, proj_matrix, base_pos

    def get_base_pos(self):
        pos = [0,0,0]
        if hasattr(self, '_base_pos'): return self._base_pos
        env = self._env.unwrapped
        if hasattr(env, 'robot'):
            if hasattr(env.robot, 'body_xyz'):  # for pybullet_envs/env_bases.py
                pos = env.robot.body_xyz
            elif hasattr(env.robot, 'slider'):  # for pendulum envs
                x, _ = env.robot.slider.current_position()
                pos = [x, 0, 0]
        return pos

    def reset_colors(self):
        # Randomly select a color scheme.
        base_clrs =  [(1,1,0,1),(0,1,1,1),(1,0,1,1),(0,0,1,1)]
        pole_clrs =  [(0,0,1,1),(0,0,0,1),(0,1,0,1),(1,0,0,1)]
        pole2_clrs = [(1,0,0,1),(0,0,1,1),(1,1,0,1),(0,0,0,1)]
        pole3_clrs = [(0.8,0.8,0.8,1),(0.7,0,0.7,1),(0.7,0,0,1),(0,0.8,0.8,1)]
        clr_scheme_id = np.random.randint(len(base_clrs))
        base_clr = base_clrs[clr_scheme_id]
        pole_clr = pole_clrs[clr_scheme_id]
        pole2_clr = pole2_clrs[clr_scheme_id]
        pole3_clr = pole3_clrs[clr_scheme_id]
        #print('base_clrs', base_clrs, 'pole_clr', pole_clr)
        if 'CartPole' in self._base_env_name:
            cartpole = self._env.unwrapped.cartpole
            self._sim.changeVisualShape(cartpole, 0, rgbaColor=base_clr)
            self._sim.changeVisualShape(cartpole, 1, rgbaColor=pole_clr)
        elif 'Pendulum' in self._base_env_name:
            base = self._env.unwrapped.robot.parts["cart"]
            pole = self._env.unwrapped.robot.parts["pole"]
            self._sim.changeVisualShape(base.bodies[base.bodyIndex],
                                        base.bodyPartIndex, rgbaColor=base_clr)
            self._sim.changeVisualShape(pole.bodies[pole.bodyIndex],
                                        pole.bodyPartIndex, rgbaColor=pole_clr)
            if 'Double' in self._base_env_name:
                pole2 = self._env.unwrapped.robot.parts["pole2"]
                #print('pole2_clr', pole2_clr)
                self._sim.changeVisualShape(
                    pole2.bodies[pole2.bodyIndex], pole2.bodyPartIndex,
                    rgbaColor=pole2_clr)
        elif self._base_env_name.startswith(('Walker2D', 'HalfCheetah', 'Ant')):
            if self._base_env_name.startswith('Walker2D'):
                part_nms = ['torso', 'thigh', 'leg', 'foot',
                            'thigh_left', 'leg_left', 'foot_left']
            elif self._base_env_name.startswith('HalfCheetah'):
                part_nms = ['torso', 'bthigh', 'bshin', 'bfoot',
                            'fthigh', 'fshin', 'ffoot']
            elif self._base_env_name.startswith('Ant'):
                part_nms = ['torso', 'front_left_foot', 'front_right_foot',
                            'left_back_foot', 'right_back_foot']
            else:
                assert(False)  # clrs for env not specified
            clrs = [pole_clr, pole2_clr, base_clr, pole_clr,
                    base_clr, pole2_clr, pole3_clr]
            for i, part_nm in enumerate(part_nms):
                part = self._env.unwrapped.robot.parts[part_nm]
                self._sim.changeVisualShape(
                    part.bodies[part.bodyIndex],
                    part.bodyPartIndex, rgbaColor=clrs[i])
        #    pitches = [-60, -35, -40, -20]; yaws = [-180, -20, 120, 270]
        #    self._env.unwrapped._cam_pitch = pitches[clr_scheme_id]
        #    self._env.unwrapped._cam_yaw = yaws[clr_scheme_id]

    def override_state(self, low_dim_state):
        # TODO: add support for more envs in the future.
        assert('CartPole' in self._base_env_name or
               'Pendulum' in self._base_env_name)
        unwrp_env = self._env.unwrapped
        ids_dict = self._ld_names_ids_dict
        assert(len(ids_dict.keys())==len(low_dim_state))
        if self._base_env_name.startswith('CartPole'):
            self._sim.resetJointState(  # set pole pos,vel
                unwrp_env.cartpole, 1, low_dim_state[ids_dict['theta']],
                low_dim_state[ids_dict['theta_vel']])
            self._sim.resetJointState(  # set cart pos,vel
                unwrp_env.cartpole, 0, low_dim_state[ids_dict['x']],
                low_dim_state[ids_dict['x_vel']])
        elif 'Pendulum' in self._base_env_name:
            unwrp_env.robot.slider.reset_current_position(
                low_dim_state[ids_dict['x']], low_dim_state[ids_dict['x_vel']])
            theta = np.arctan2(low_dim_state[ids_dict['theta_sin']],
                               low_dim_state[ids_dict['theta_cos']])
            unwrp_env.robot.j1.reset_current_position(
                theta, low_dim_state[ids_dict['theta_vel']])
            if 'Double' in self._base_env_name:
                theta1 = np.arctan2(low_dim_state[ids_dict['theta1_sin']],
                                    low_dim_state[ids_dict['theta1_cos']])
                unwrp_env.robot.j2.reset_current_position(
                    theta1, low_dim_state[ids_dict['theta1_vel']])
        unwrp_env.state = self.compute_low_dim_state()
        return None
