"""
Rearrangement envs for a simple 2-link Reacher from MJCF/XML and from a URDF.
"""
import os
import numpy as np
import pybullet

from ..utils.bullet_manipulator import BulletManipulator
from ..utils.bullet_reacher import BulletReacher
from .rearrange_env import RearrangeEnv


class ReacherRearrangeEnv(RearrangeEnv):
    def __init__(self, version, variant='Ycb',
                 rnd_init_pos=False, control_mode='torque',
                 obs_resolution=64, obs_ptcloud=False,
                 debug=False, visualize=False):
        if obs_ptcloud:
            data_folder = os.path.join(os.path.split(__file__)[0], 'data')
            robot = BulletManipulator(
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
                robot.sim.resetDebugVisualizerCamera(
                    cameraDistance=0.44, cameraYaw=robot.cam_yaw,
                    cameraPitch=-65, cameraTargetPosition=(0.05, 0, 0))
        else:
            robot = BulletReacher(robot_desc_file='reacher.xml',
                                  gui=visualize, camera_distance=0.40)
        super(ReacherRearrangeEnv, self).__init__(
            version, variant, robot, rnd_init_pos,
            obs_resolution, obs_ptcloud, debug)

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
