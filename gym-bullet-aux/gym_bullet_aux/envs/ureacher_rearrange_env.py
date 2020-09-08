"""
Rearrangement env for a simple 2-link Reacher but from URDF file.
"""
import os
import numpy as np
import pybullet

from gym_bullet_extensions.bullet_manipulator import BulletManipulator

from .rearrange_env import RearrangeEnv


class UreacherRearrangeEnv(RearrangeEnv):
    def __init__(self, version, max_episode_len, control_mode='torque',
                 obs_resolution=64, obs_ptcloud=False, variant='Ycb',
                 rnd_init_pos=False, statics_in_lowdim=False,
                 visualize=False, debug_level=0):
        # Note: RearrangeEnv expects that we created self.robot already.
        data_folder = os.path.join(os.path.split(__file__)[0], 'data')
        self.robot = BulletManipulator(
            os.path.join(data_folder, 'reacher.urdf'),
            control_mode=control_mode,
            ee_joint_name='reacher_joint4', ee_link_name='reacher_link4',
            base_pos=[0,0,0],
            dt=1.0/100.0, kp=([200.0]*7 + [1.0]*2), kd=([2.0]*7 + [0.1]*2),
            visualize=visualize, cam_dist=0.4, cam_yaw=90, cam_pitch=-89,
            cam_target=(0.0, 0, 0), default_ground=False)
        super(UreacherRearrangeEnv, self).__init__(
            version=version, max_episode_len=max_episode_len,
            obs_resolution=obs_resolution, obs_ptcloud=obs_ptcloud,
            variant=variant, rnd_init_pos=rnd_init_pos,
            statics_in_lowdim=statics_in_lowdim, debug_level=debug_level)
        if visualize:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            self.robot.sim.resetDebugVisualizerCamera(  # was: cam dist=0.37
                cameraDistance=0.44, cameraYaw=self.robot.cam_yaw,
                cameraPitch=-65, cameraTargetPosition=(0.05, 0, 0))

    def step(self, action):
        qpos = self.robot.get_qpos(); qvel = self.robot.get_qvel()
        rest = np.abs(qpos[0]) - np.pi
        if rest > 0:
            jpos = (rest-np.pi)*np.sign(qpos[0])
            #print('reset ', qpos, qvel, 'to', jpos, -qvel)
            self.robot.sim.resetJointState(
                bodyUniqueId=self.robot.info.robot_id,
                jointIndex=self.robot.info.joint_ids[0],
                targetValue=jpos, targetVelocity=qvel[0])
        return super(UreacherRearrangeEnv, self).step(action)
