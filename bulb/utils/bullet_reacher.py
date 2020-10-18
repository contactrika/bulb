"""
A pyBullet gym env for planar 2-link reacher (with an improved visualization).
"""

from copy import copy
import os
import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc


class BulletReacher:
    # The following default fields are based on the structure of the default
    # reacher.xml from pybullet_data (same as the standard MJCF reacher).
    LINK_NAMES = ['base_link', 'shoulder_link', '', 'elbow_link',
                  'hand_link', 'hand_center',
                  'fing_l', 'fing_r', 'tip_l', 'tip_r']
    EE_LINK_ID = 4
    CONTROLLED_JOINTS = [0, 2]
    TORQUE_LIMITS = np.asarray([3.0, 3.0])
    JOINT_LIMITS = np.array([31/32,26/32])*np.pi

    def __init__(self, robot_desc_file, gui=False, camera_distance=1.0):
        self.ee_idx = BulletReacher.EE_LINK_ID
        self.controlled_joints = BulletReacher.CONTROLLED_JOINTS
        self.torque_limits = BulletReacher.TORQUE_LIMITS
        self.dof = len(self.controlled_joints)
        # Set up bullet client and load plane
        self.sim = bc.BulletClient(
            connection_mode=pybullet.GUI if gui else pybullet.DIRECT)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = pybullet.loadURDF('plane.urdf',[0,0,0])
        self.sim.changeVisualShape(self.plane_id, -1, rgbaColor=(0,0,0,1))
        # Load robot.
        data_path = os.path.join(
            os.path.split(__file__)[0], '..', 'envs', 'data')
        if not os.path.isabs(robot_desc_file):
            robot_desc_file = os.path.join(data_path, robot_desc_file)
        if robot_desc_file.endswith('.urdf'):
            robot_id = self.sim.loadURDF(robot_desc_file, [0,0,0])
        else:
            world_id, robot_id = self.sim.loadMJCF(robot_desc_file)
            self.sim.changeVisualShape(world_id, -1, rgbaColor=(1,1,1,1))
        self.robot_id = robot_id
        for link_id in [1,3,4,5]:
            rgb = [1,1,1] if link_id in [1,4,5] else [1,0.6,0.15]
            self.sim.changeVisualShape(self.robot_id, link_id,
                                       rgbaColor=(*rgb,1))
        # Init bullet sim params.
        self.sim.setGravity(0, 0, -9.81)
        # default: https://github.com/bulletphysics/bullet3/issues/1460
        dt = 0.01  # 100Hz
        self.sim.setTimeStep(dt)
        self.sim.setRealTimeSimulation(0)
        self.sim.setPhysicsEngineParameter(numSolverIterations=5, numSubSteps=2)
        self.sim.setJointMotorControlArray(
            self.robot_id, self.controlled_joints,
            pybullet.VELOCITY_CONTROL, forces=np.zeros(self.dof))
        self.cam_dist = camera_distance
        self.cam_yaw = 90; self.cam_pitch = -89; self.cam_target = [0.0, 0, 0]
        if gui:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
            self.sim.resetDebugVisualizerCamera(
                cameraDistance=self.cam_dist,
                cameraYaw=self.cam_yaw, cameraPitch=self.cam_pitch,
                cameraTargetPosition=self.cam_target)

    def disconnect(self):
        self.sim.disconnect()

    @staticmethod
    def create_visual_area(shape, center, radius, rgba):
        assert(shape==pybullet.GEOM_SPHERE or pybullet.GEOM_CAPSULE)
        kwargs = {'shapeType':shape, 'radius':radius}
        viz_kwargs = copy(kwargs); col_kwargs = copy(kwargs)
        if shape == pybullet.GEOM_CAPSULE:
            # these don't seem to be set correctly by pybullet...
            viz_kwargs['length'] = 0.02; col_kwargs['height'] = 0.02
        viz_shp_id = pybullet.createVisualShape(
            visualFramePosition=[0,0,0], rgbaColor=rgba, **viz_kwargs)
        col_shp_id = pybullet.createCollisionShape(
            collisionFramePosition=[0,0,0], **col_kwargs)
        # Only using this for visualization, so mass=0 (fixed body).
        sphere_body_id = pybullet.createMultiBody(
            baseMass=0, baseInertialFramePosition=[0,0,0],
            baseCollisionShapeIndex=col_shp_id,
            baseVisualShapeIndex=viz_shp_id,
            basePosition=center, useMaximalCoordinates=True)
        return sphere_body_id

    def get_maxforce(self):
        return self.torque_limits

    def reset(self):
        joint_pos = np.zeros(self.dof)
        self.reset_to_qpos(joint_pos)

    def reset_to_qpos(self, joint_pos):
        joint_vel = np.zeros(self.dof)
        for i in range(self.dof):
            self.sim.resetJointState(bodyUniqueId=self.robot_id,
                                     jointIndex=self.controlled_joints[i],
                                     targetValue=joint_pos[i],
                                     targetVelocity=joint_vel[i])

    def reset_objects(self, ids, poses, quats):
        for objid in range(len(ids)):
            self.sim.resetBasePositionAndOrientation(
                ids[objid], poses[objid], quats[objid])

    def apply_joint_torque(self, torque):
        torque = torque.clip(-self.torque_limits, self.torque_limits)
        self.sim.setJointMotorControlArray(
            bodyIndex=self.robot_id, jointIndices=self.controlled_joints,
            controlMode=pybullet.TORQUE_CONTROL, forces=torque)
        self.sim.stepSimulation()
        # Keep qpos in [-(pi-2eps),pi-2eps]
        qpos = self.get_qpos(); qvel = self.get_qvel()
        for j in range(len(qpos)):
            lim = BulletReacher.JOINT_LIMITS[j]
            if np.abs(qpos[j]) > lim:
                jpos = lim*np.sign(qpos[j])
                self.sim.resetJointState(
                    bodyUniqueId=self.robot_id,
                    jointIndex=self.controlled_joints[j],
                    targetValue=jpos, targetVelocity=-qvel[j])
        if (np.abs(self.get_qpos()) > np.pi).any():
            print('Invalid robot qpos', qpos)
            assert(False)  # invalid robot qpos

    def inverse_dynamics(self, des_acc):
        cur_joint_states = self.sim.getJointStates(
            self.robot_id, self.controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self.dof)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.dof)]
        torques = self.sim.calculateInverseDynamics(
            self.robot_id, cur_joint_angles, cur_joint_vel, des_acc)
        return np.asarray(torques)

    def get_qpos(self):
        joint_states = self.sim.getJointStates(
            self.robot_id, self.controlled_joints)
        qpos = [joint_states[i][0] for i in range(self.dof)]
        return np.array(qpos)

    def get_qvel(self):
        joint_states = self.sim.getJointStates(
            self.robot_id, self.controlled_joints)
        qvel = [joint_states[i][1] for i in range(self.dof)]
        return np.array(qvel)

    def get_ee_pos(self):
        ee_state = self.sim.getLinkState(self.robot_id, self.ee_idx)
        return np.array(ee_state[0])

    def collisions_ok(self, object_ids, debug=False):
        ok = True
        pts = pybullet.getContactPoints()
        for pt in pts:
            # See getContactPoints() docs (page 43)
            body_bullet_ids = [pt[1], pt[2]]
            if self.robot_id not in body_bullet_ids: continue
            # Suppose 1st body was the robot.
            other_bullet_id = pt[2]  # suppose 2nd body is the non-robot
            link_bullet_id = pt[3]   # get link of the robot that collided
            if other_bullet_id == self.robot_id:
                other_bullet_id = pt[1]  # the other point must be non-robot
                link_bullet_id = pt[4]   # and robot link is from 2nd body then
            if other_bullet_id not in object_ids: continue
            obj_num = object_ids.index(other_bullet_id)
            if debug:
                print(BulletReacher.LINK_NAMES[link_bullet_id],
                      'of the robot collided with object number', obj_num)
                print('contact pt', pt)
            ok = False
            break
        return ok
