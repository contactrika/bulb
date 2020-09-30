"""
Rearrangement envs for a simple 2-link Reacher from MJCF/XML and from a URDF.
"""
import os
import numpy as np
import pybullet

from gym_bullet_extensions.bullet_manipulator import BulletManipulator

from .reacher_sim import ReacherBulletSimulation
from .rearrange_env import RearrangeEnv


class ReacherRearrangeEnv(RearrangeEnv):
    def __init__(self, version, variant='Ycb',
                 obs_resolution=64, obs_ptcloud=False,
                 rnd_init_pos=False, control_mode='torque',
                 debug=False, visualize=False):
        # Note: RearrangeEnv expects that we create self.robot.
        if obs_ptcloud:
            data_folder = os.path.join(os.path.split(__file__)[0], 'data')
            self.robot = BulletManipulator(
                os.path.join(data_folder, 'reacher.urdf'),
                control_mode=control_mode,
                ee_joint_name='reacher_joint4', ee_link_name='reacher_link4',
                base_pos=[0,0,0],
                dt=1.0/100.0, kp=([200.0]*7 + [1.0]*2), kd=([2.0]*7 + [0.1]*2),
                visualize=visualize, cam_dist=0.4, cam_yaw=90, cam_pitch=-89,
                cam_target=(0.0, 0, 0), default_ground=False)
            if visualize:
                pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
                pybullet.configureDebugVisualizer(
                    pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
                self.robot.sim.resetDebugVisualizerCamera(
                    cameraDistance=0.44, cameraYaw=self.robot.cam_yaw,
                    cameraPitch=-65, cameraTargetPosition=(0.05, 0, 0))
        else:
            self.robot = ReacherBulletSimulation(
                robot_desc_file='reacher.xml',
                gui=visualize, camera_distance=0.40)
        super(ReacherRearrangeEnv, self).__init__(
            version=version, variant=variant,
            obs_resolution=obs_resolution, obs_ptcloud=obs_ptcloud,
            rnd_init_pos=rnd_init_pos, debug=debug)

    def step(self, action):
        if self._obs_ptcloud:
            qpos = self.robot.get_qpos(); qvel = self.robot.get_qvel()
            rest = np.abs(qpos[0]) - np.pi
            if rest > 0:
                jpos = (rest-np.pi)*np.sign(qpos[0])
                self.robot.sim.resetJointState(
                    bodyUniqueId=self.robot.info.robot_id,
                    jointIndex=self.robot.info.joint_ids[0],
                    targetValue=jpos, targetVelocity=qvel[0])
        return super(ReacherRearrangeEnv, self).step(action)
