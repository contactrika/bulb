"""
Rearrangement env for Franka Emika.
"""

import os
import numpy as np
import pybullet

from gym_bullet_extensions.bullet_manipulator import BulletManipulator

from .rearrange_env import RearrangeEnv


class FrankaRearrangeEnv(RearrangeEnv):
    def __init__(self, version, variant='Ycb',
                 obs_resolution=64, obs_ptcloud=False,
                 rnd_init_pos=False, statics_in_lowdim=False,
                 control_mode='torque', debug=False, visualize=False):
        self.robot = BulletManipulator(
            os.path.join('franka_robot', 'franka_small_fingers.urdf'),
            control_mode=control_mode,
            ee_joint_name='panda_joint7', ee_link_name='panda_hand',
            base_pos=[-0.7,0,0],
            rest_arm_qpos=[-0.0412, 0.5043, 0.0048, -2.6232, -0.1651, 3.1273, 0.9135],
            dt=1.0/100.0, kp=([200.0]*7 + [1.0]*2), kd=([2.0]*7 + [0.1]*2),
            min_z=0.0,
            visualize=visualize, cam_dist=0.44, cam_yaw=90, cam_pitch=-65,
            cam_target=(0.05, 0, 0), default_ground=False)
        super(FrankaRearrangeEnv, self).__init__(
            version=version, variant=variant,
            obs_resolution=obs_resolution, obs_ptcloud=obs_ptcloud,
            rnd_init_pos=rnd_init_pos, statics_in_lowdim=statics_in_lowdim,
            debug=debug)
        if visualize:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            self.robot.sim.resetDebugVisualizerCamera(  # was: cam dist=0.37
                cameraDistance=0.44, cameraYaw=self.robot.cam_yaw,
                cameraPitch=-65, cameraTargetPosition=(0.05, 0, 0))
        self.rnd_qpos_fxn = self.qpos_for_random_ee_pose

    def qpos_for_random_ee_pose(self):
        dim = 3+3  # EE pos + Euler angles ori
        ctrl_lows = np.zeros([dim])
        ctrl_highs = np.zeros([dim])
        # Region in front of the robot. Restricted to be relatively low but
        # above the ground (table) to avoid random motions outside of the
        # main workspace.
        ctrl_lows[0] = -0.5; ctrl_highs[0] = 0.5   # x
        ctrl_lows[1] = -0.5; ctrl_highs[1] = 0.5   # y
        ctrl_lows[2] = 0.05; ctrl_highs[2] = 0.35  # z
        # Gripper orientation encoded in euler angles
        ctrl_lows[3:6] = -np.pi;  ctrl_highs[3:6] = np.pi
        # Sample a random point within the given range.
        rnd_pos_ori = np.random.rand(dim)*(ctrl_highs-ctrl_lows) + ctrl_lows
        ee_pos = rnd_pos_ori[0:3]
        # Restrict the gripper to point mostly down (not above 90 deg).
        rpys = rnd_pos_ori[3:6]
        small_angle = np.pi/2
        rid = 0  #  restrict roll (down)
        almost_down = np.pi-small_angle
        if np.abs(rpys[rid])<almost_down:
            rpys[rid] = almost_down*np.sign(rpys[rid])
        max_rpy = np.array([np.pi,small_angle,small_angle])
        rpys = np.clip(rpys, -max_rpy, max_rpy)
        ee_quat = np.array(self.robot.sim.getQuaternionFromEuler(rpys.tolist()))
        # Compute the corresponding qpos.
        qpos = self.robot.ee_pos_to_qpos(ee_pos, ee_quat, fing_dist=0)
        return qpos, ee_pos, ee_quat
