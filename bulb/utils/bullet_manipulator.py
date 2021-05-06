"""
PyBullet simulator setup for 7DOF manipulators (Franka Emika, Sawyer).
Supports gripper control for Franka Emika robot.
"""
from copy import copy
import os
import time

import numpy as np

import pybullet_utils.bullet_client as bclient
import pybullet_data
import pybullet


class ManipulatorInfo:
    def __init__(self, robot_id, joint_ids, joint_names,
                 joint_minpos, joint_maxpos,
                 joint_maxforce, joint_maxvel,
                 ee_link_id, arm_jids_lst, ee_jid, finger_jids_lst,
                 left_ee_link_id=None, left_arm_jids_lst=None,
                 left_ee_jid=None, left_finger_jids_lst=None):
        self.robot_id = robot_id
        self.joint_ids = joint_ids
        self.joint_names = joint_names
        self.joint_minpos = joint_minpos
        self.joint_maxpos = joint_maxpos
        self.joint_maxforce = joint_maxforce
        self.joint_maxvel = joint_maxvel
        self.ee_link_id = ee_link_id
        self.arm_jids_lst = arm_jids_lst
        self.ee_jid = ee_jid
        self.finger_jids_lst = finger_jids_lst
        self.left_ee_link_id = left_ee_link_id
        self.left_arm_jids_lst = left_arm_jids_lst
        self.left_ee_jid = left_ee_jid
        self.left_finger_jids_lst = left_finger_jids_lst
        self.dof = len(joint_ids)

    def print(self):
        print('ManipulatorInfo: robot_id', self.robot_id,
              '\n joint_ids', self.joint_ids,
              '\n joint_names', self.joint_names,
              '\n joint_minpos', self.joint_minpos,
              '\n joint_maxpos', self.joint_maxpos,
              '\n joint_maxforce', self.joint_maxforce,
              '\n joint_maxvel', self.joint_maxvel,
              '\n ee_link_id', self.ee_link_id,
              '\n right_arm_jids_lst', self.arm_jids_lst,
              '\n ee_jid', self.ee_jid,
              '\n finger_jids_lst', self.finger_jids_lst,
              '\n left_ee_link_id', self.left_ee_link_id,
              '\n left_arm_jids_lst', self.left_arm_jids_lst,
              '\n left_ee_jid', self.left_ee_jid,
              '\n left_finger_jids_lst', self.left_finger_jids_lst)


class BulletManipulator:
    # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12077
    MAX_VELOCITY = 100.0  # 100 rad/s
    GRAVITY = -9.81
    # Minimum allowed distance of EE to ground/table.

    def __init__(self, robot_desc_file, ee_joint_name, ee_link_name,
                 control_mode, base_pos, rest_arm_qpos=None,
                 left_ee_joint_name=None, left_ee_link_name=None,
                 left_fing_link_prefix=None, left_joint_suffix=None,
                 left_rest_arm_qpos=None,
                 dt=1.0/240.0, kp=1.0, kd=0.1, min_z=0.0,
                 visualize=False, cam_dist=1.5, cam_yaw=25, cam_pitch=-35,
                 cam_target=(0.5, 0, 0), debug_level=0):
        assert(control_mode in
               ('ee_position', 'position', 'velocity', 'torque'))
        self.control_mode = control_mode
        self.dt = dt; self.kp = kp; self.kd = kd; self.min_z = min_z
        self.debug_level = debug_level
        self.visualize = visualize; self.cam_dist = cam_dist
        self.cam_yaw = cam_yaw; self.cam_pitch = cam_pitch
        self.cam_target = list(cam_target)
        # Create and connect bullet simulation client.
        if visualize:
            self.sim = bclient.BulletClient(connection_mode=pybullet.GUI)
            # disable aux menus in the gui
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
            # don't render during init
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,0)
        else:
            self.sim = bclient.BulletClient(connection_mode=pybullet.DIRECT)
        self._aux_sim = bclient.BulletClient(
            connection_mode=pybullet.DIRECT)
        # Load ground.
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._aux_sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = self.sim.loadURDF('plane.urdf', [0,0,0])
        self._aux_sim.loadURDF('plane.urdf', [0,0,0])
        # Note: changing ground color doesn't work even for plane_transparent.urdf
        # pybullet.changeVisualShape(self.plane_id, -1, rgbaColor=[1, 1, 1, 1])
        #
        # Load robot from URDF.
        if not os.path.isabs(robot_desc_file):
            robot_desc_file = os.path.join(os.path.split(__file__)[0], '..',
                                           'envs', 'data', robot_desc_file)
        print('robot_desc_file', robot_desc_file)
        self.info = self.load_robot(
            robot_desc_file, ee_joint_name, ee_link_name,
            left_ee_joint_name, left_ee_link_name,
            left_fing_link_prefix, left_joint_suffix,
            base_pos=base_pos, base_quat=[0,0,0,1])
        # Set simulation parameters.
        # time step: https://github.com/bulletphysics/bullet3/issues/1460
        self.sim.setRealTimeSimulation(0)
        self.sim.setGravity(0, 0, BulletManipulator.GRAVITY)
        self.sim.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=5, numSubSteps=2)
        # Init aux sim.
        self._aux_sim.setRealTimeSimulation(0)
        self._aux_sim.setGravity(0, 0, BulletManipulator.GRAVITY)
        self._aux_sim.setPhysicsEngineParameter(
            fixedTimeStep=self.dt, numSolverIterations=5, numSubSteps=2)
        # Turn on FT sensors (for model-free gravity compensation compute).
        # BulletManipulatorSim.toggle_ft_sensors(self.sim, self.robots, on=True)
        # Reset to initial position and visualize.
        self.rest_qpos = (self.info.joint_maxpos+self.info.joint_minpos)/2
        if rest_arm_qpos is not None:
            assert(len(self.info.arm_jids_lst)==len(rest_arm_qpos))
            self.rest_qpos[self.info.arm_jids_lst] = rest_arm_qpos[:]
        if left_rest_arm_qpos is not None:
            assert(len(self.info.left_arm_jids_lst)==len(left_rest_arm_qpos))
            self.rest_qpos[self.info.left_arm_jids_lst] = left_rest_arm_qpos[:]
        self.reset()
        if self.visualize:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING,1)
            self.refresh_viz()

    def load_robot(self, robot_path, ee_joint_name, ee_link_name,
                   left_ee_joint_name, left_ee_link_name,
                   left_fing_link_prefix, left_joint_suffix,
                   base_pos, base_quat):
        robot_id = self.sim.loadURDF(
            robot_path, basePosition=base_pos, baseOrientation=base_quat,
            useFixedBase=True, flags=pybullet.URDF_USE_SELF_COLLISION)
        self._aux_sim.loadURDF(
            robot_path, basePosition=base_pos, baseOrientation=base_quat,
            useFixedBase=True, flags=pybullet.URDF_USE_SELF_COLLISION)
        joint_ids = []; joint_names = []
        joint_minpos = []; joint_maxpos = []
        joint_maxforce = []; joint_maxvel = []
        ee_link_id = None; ee_jid = None
        left_ee_link_id = None; left_ee_jid = None
        finger_jids_lst = []; left_finger_jids_lst = []
        arm_jids_lst = []; left_arm_jids_lst = []
        for j in range(pybullet.getNumJoints(robot_id)):
            _, jname, jtype, _, _, _, _, _, \
            jlowlim, jhighlim, jmaxforce, jmaxvel, link_name, _, _, _, _ = \
                pybullet.getJointInfo(robot_id, j)
            jname = jname.decode("utf-8"); link_name = link_name.decode("utf-8")
            if jtype in [pybullet.JOINT_REVOLUTE, pybullet.JOINT_PRISMATIC]:
                joint_ids.append(j); joint_names.append(jname)
                joint_minpos.append(jlowlim); joint_maxpos.append(jhighlim)
                joint_maxforce.append(jmaxforce); joint_maxvel.append(jmaxvel)
            jid = len(joint_ids)-1 # internal index (not pybullet id)
            if jtype == pybullet.JOINT_REVOLUTE:
                if (left_joint_suffix is not None and
                        jname.endswith(left_joint_suffix)):
                    left_arm_jids_lst.append(jid)
                else:
                    arm_jids_lst.append(jid)
            if jtype == pybullet.JOINT_PRISMATIC:
                if ((left_fing_link_prefix is not None) and
                        jname.startswith(left_fing_link_prefix)):
                    left_finger_jids_lst.append(jid)
                else:
                    finger_jids_lst.append(jid)
            if jname == ee_joint_name: ee_jid = jid
            if link_name == ee_link_name: ee_link_id = j  # for IK
            if jname == left_ee_joint_name: left_ee_jid = jid
            if link_name == left_ee_link_name: left_ee_link_id = j  # for IK
        assert(ee_link_id is not None)
        assert(ee_jid is not None)
        info = ManipulatorInfo(
            robot_id, np.array(joint_ids), np.array(joint_names),
            np.array(joint_minpos), np.array(joint_maxpos),
            np.array(joint_maxforce), np.array(joint_maxvel),
            ee_link_id, arm_jids_lst, ee_jid, finger_jids_lst,
            left_ee_link_id, left_arm_jids_lst,
            left_ee_jid, left_finger_jids_lst)
        if self.debug_level>=0: info.print()
        return info

    def reset(self):
        self.reset_to_qpos(self.rest_qpos)

    def reset_to_qpos(self, qpos):
        qpos = np.clip(qpos, self.info.joint_minpos, self.info.joint_maxpos)
        for jid in range(self.info.dof):
            self.sim.resetJointState(
                bodyUniqueId=self.info.robot_id,
                jointIndex=self.info.joint_ids[jid],
                targetValue=qpos[jid], targetVelocity=0)
            self._aux_sim.resetJointState(
                bodyUniqueId=self.info.robot_id,
                jointIndex=self.info.joint_ids[jid],
                targetValue=qpos[jid], targetVelocity=0)
        # We need these to be called after every reset. This fact is not
        # documented, and bugs resulting from not zeroing-out velocity and
        # torque control values this way are hard to reproduce.
        self.sim.setJointMotorControlArray(
            self.info.robot_id, self.info.joint_ids.tolist(),
            pybullet.VELOCITY_CONTROL, forces=[0]*self.info.dof)
        self.sim.setJointMotorControlArray(
            self.info.robot_id, self.info.joint_ids.tolist(),
            pybullet.TORQUE_CONTROL, forces=[0]*self.info.dof)

    def refresh_viz(self):
        time.sleep(0.1)
        self.sim.resetDebugVisualizerCamera(
            cameraDistance=self.cam_dist, cameraYaw=self.cam_yaw,
            cameraPitch=self.cam_pitch, cameraTargetPosition=self.cam_target)

    def reset_objects(self, ids, poses, quats):
        for objid in range(len(ids)):
            self.sim.resetBasePositionAndOrientation(
                ids[objid], poses[objid], quats[objid])

    def disconnect(self):
        self.sim.disconnect()

    def set_joint_limits(self, minpos, maxpos):
        self.info.joint_minpos[:] = minpos[:]
        self.info.joint_maxpos[:] = maxpos[:]

    def get_minpos(self):
        return self.info.joint_minpos

    def get_maxpos(self):
        return self.info.joint_maxpos

    def get_maxforce(self):
        return self.info.joint_maxforce

    def get_maxvel(self):
        return self.info.joint_maxvel

    def get_max_fing_dist(self):
        farr = np.array(self.info.finger_jids_lst)
        return self.info.joint_maxpos[farr].sum() # for symmetric commands

    def get_qpos(self):
        joint_states = self.sim.getJointStates(
            self.info.robot_id, self.info.joint_ids)
        qpos = [joint_states[i][0] for i in range(self.info.dof)]
        return np.array(qpos)

    def get_qvel(self):
        joint_states = self.sim.getJointStates(
            self.info.robot_id, self.info.joint_ids)
        qvel = [joint_states[i][1] for i in range(self.info.dof)]
        return np.array(qvel)

    def get_fing_dist(self):
        joint_states = self.sim.getJointStates(
            self.info.robot_id, self.info.joint_ids)
        fing_dists = [joint_states[i][0] for i in self.info.finger_jids_lst]
        return np.array(fing_dists).sum()

    def get_ee_pos(self):
        pos, _, _, _ = self.get_ee_pos_ori_vel()
        return pos

    def get_ee_pos_ori_vel(self):
        # Returns (in world coords): EE 3D position, quaternion orientation,
        # linear and angular velocity.
        ee_state = self.sim.getLinkState(
            self.info.robot_id, self.info.ee_link_id, computeLinkVelocity=1)
        return np.array(ee_state[0]), np.array(ee_state[1]),\
               np.array(ee_state[6]), np.array(ee_state[7])

    def _ee_pos_to_qpos_raw(self, ee_pos, ee_quat=None, fing_dist=0.0,
                            left_ee_pos=None, left_ee_quat=None,
                            left_fing_dist=0.0, debug=False):
        ee_ori = None if ee_quat is None else ee_quat.tolist()
        qpos = pybullet.calculateInverseKinematics(
            self.info.robot_id, self.info.ee_link_id,
            targetPosition=ee_pos.tolist(), targetOrientation=ee_ori,
            lowerLimits=self.info.joint_minpos.tolist(),
            upperLimits=self.info.joint_maxpos.tolist(),
            jointRanges=(self.info.joint_maxpos-self.info.joint_minpos).tolist(),
            restPoses=self.rest_qpos.tolist(),
            #solver=pybullet.IK_SDLS,
            maxNumIterations=1000, residualThreshold=0.0001)
        qpos = np.array(qpos)
        if debug: print('_ee_pos_to_qpos_raw() qpos from IK', qpos)
        for jid in self.info.finger_jids_lst:
            qpos[jid] = np.clip(  # finger info (not set by IK)
                fing_dist/2.0, self.info.joint_minpos[jid],
                self.info.joint_maxpos[jid])
        # Take care of left arm, if needed.
        if len(self.info.left_arm_jids_lst)>0:
            if left_ee_pos is not None:
                left_qpos = np.array(pybullet.calculateInverseKinematics(
                    self.info.robot_id, self.info.left_ee_link_id,
                    left_ee_pos.tolist(),
                    None if left_ee_quat is None else left_ee_quat.tolist(),
                    maxNumIterations=1000, residualThreshold=0.0001))
            else:
                left_qpos = self.get_qpos()
            qpos[self.info.left_arm_jids_lst] = \
                left_qpos[self.info.left_arm_jids_lst]
        for jid in self.info.left_finger_jids_lst:
            qpos[jid] = np.clip(  # finger info (not set by IK)
                left_fing_dist/2.0, self.info.joint_minpos[jid],
                self.info.joint_maxpos[jid])
        # IK will find solutions outside of joint limits, so clip.
        qpos = np.clip(qpos, self.info.joint_minpos, self.info.joint_maxpos)
        return qpos

    def move_to_qpos(self, tgt_qpos, mode, kp=None, kd=None):
        if kp is None: kp = self.kp
        if kd is None: kd = self.kd
        tgt_qvel = np.zeros_like(tgt_qpos)
        self.move_to_qposvel(tgt_qpos, tgt_qvel, mode, kp, kd)

    def reset_joint(self, jid, jpos, jvel):
        self.sim.resetJointState(
            bodyUniqueId=self.info.robot_id,
            jointIndex=self.info.joint_ids[jid],
            targetValue=jpos, targetVelocity=jvel)
        self._aux_sim.resetJointState(
            bodyUniqueId=self.info.robot_id,
            jointIndex=self.info.joint_ids[jid],
            targetValue=jpos, targetVelocity=jvel)

    def get_ok_qvel(self, tgt_qvel):
        # A code that is also reasonable for hardware Yumi sanity checks.
        if 'yumi' in self.info.joint_names[0]:
            HW_VEL_SCALING = 0.5 #0.1  # for 500Hz in sim instead of 100Hz on hw
            ok_tgt_qvel = np.copy(tgt_qvel)*HW_VEL_SCALING
            LIM_SC = 0.8
            # Hard joint limits to prevent real robot from getting into very
            # awkward poses (for reasonable hardware exploration runs).
            JOINT_MINPOS = np.array([
                0.8000, -1.5000, -2.9409*LIM_SC, -2.1555*LIM_SC,
                -5.0615*LIM_SC, -1.5359*LIM_SC, -3.9968*LIM_SC])
            JOINT_MAXPOS = np.array(
                [2.9409*LIM_SC, 0.7592*LIM_SC, 1.0000, 0.6000,
                 5.0615*LIM_SC, 2.4086*LIM_SC, 3.9968*LIM_SC])
        else:
            ok_tgt_qvel = np.copy(tgt_qvel)
            LIM_SC = 0.95
            JOINT_MINPOS = np.copy(self.info.joint_minpos[0:7])*LIM_SC
            JOINT_MAXPOS = np.copy(self.info.joint_maxpos[0:7])*LIM_SC
        qpos = self.get_qpos()
        for jid in range(JOINT_MINPOS.shape[0]):
            if qpos[jid]<JOINT_MINPOS[jid] or qpos[jid]>JOINT_MAXPOS[jid]:
                ok_tgt_qvel[jid] = 0.0
                self.reset_joint(jid, qpos[jid], ok_tgt_qvel[jid])
        # Stop all motion if the robot is too close to the ground/table.
        ee_pos = self.get_ee_pos()
        if ee_pos[2]<=self.min_z:
            ok_tgt_qvel = None
            for jid in range(JOINT_MINPOS.shape[0]):
                self.reset_joint(jid, qpos[jid], 0.0)
        return ok_tgt_qvel

    def move_with_qvel(self, tgt_qvel, mode, kp=None, kd=None):
        if kp is None: kp = self.kp
        if kd is None: kd = self.kd
        tgt_qpos = np.zeros_like(tgt_qvel)
        ok_tgt_qvel = self.get_ok_qvel(tgt_qvel)
        if ok_tgt_qvel is not None:
            self.move_to_qposvel(tgt_qpos, ok_tgt_qvel, mode, kp, kd)
        else:
            self.move_to_qposvel(
                tgt_qpos, np.zeros_like(tgt_qvel), mode, kp, kd)

    def move_to_qposvel(self, tgt_qpos, tgt_qvel, mode, kp, kd):
        assert(mode in [pybullet.POSITION_CONTROL,
                        pybullet.VELOCITY_CONTROL,
                        pybullet.PD_CONTROL])
        kps = kp if type(kp)==list else [kp]*self.info.dof
        kds = kd if type(kd)==list else [kd]*self.info.dof
        rbt_tgt_qpos = np.clip(
            tgt_qpos, self.info.joint_minpos, self.info.joint_maxpos)
        rbt_tgt_qvel = np.clip(
            tgt_qvel, -1.0*self.info.joint_maxvel, self.info.joint_maxvel)
        # PD example: https://github.com/bulletphysics/bullet3/issues/2152
        # cpp implementation with example kp, kd values (line 1731):
        # bullet3/examples/SharedMemory/b3RobotSimulatorClientAPI_NoDirect.cpp
        # ATTENTION: it is extremely important to set maximum forces when
        # executing PD control. This is not documented, but PyBullet seems
        # to have a memory corruption problem (when high torques are
        # applied the simulation can get permanently and silently corrupted
        # without any warnings). Save/restore state does not help, need to
        # delete and re-instantiate the whole simulation.
        if mode==pybullet.POSITION_CONTROL:
            self.sim.setJointMotorControlArray(
                bodyUniqueId=self.info.robot_id,
                jointIndices=self.info.joint_ids.tolist(),
                targetPositions=rbt_tgt_qpos, targetVelocities=rbt_tgt_qvel,
                controlMode=pybullet.POSITION_CONTROL,
                positionGains=kps,  # e.g. 0.1
                velocityGains=kds,  # e.g. 1.0
                forces=self.info.joint_maxforce)  # see page 22 of pybullet docs
        elif mode==pybullet.VELOCITY_CONTROL:
            self.sim.setJointMotorControlArray(
                bodyUniqueId=self.info.robot_id,
                jointIndices=self.info.joint_ids.tolist(),
                targetPositions=rbt_tgt_qpos, targetVelocities=rbt_tgt_qvel,
                controlMode=pybullet.VELOCITY_CONTROL,
                positionGains=kps,  # e.g. 0.1
                velocityGains=kds,  # e.g. 1.0
                forces=self.info.joint_maxforce)  # see page 22 of pybullet docs
        else:  # PD_CONTROL
            self.sim.setJointMotorControlArray(
                bodyUniqueId=self.info.robot_id,
                jointIndices=self.info.joint_ids.tolist(),
                targetPositions=rbt_tgt_qpos.tolist(),
                targetVelocities=rbt_tgt_qvel.tolist(),
                controlMode=pybullet.PD_CONTROL,
                positionGains=kps,  # e.g. 0.1
                velocityGains=kds,  # e.g. 1.0
                forces=self.info.joint_maxforce.tolist())  # see docs page 22
        self.sim.stepSimulation()
        self.obey_joint_limits()

    def move_to_ee_pos(self, tgt_ee_pos, tgt_ee_quat, fing_dist=0.0,
                       left_ee_pos=None, left_ee_quat=None, left_fing_dist=0.0,
                       mode=pybullet.POSITION_CONTROL, debug=True):
        qpos = None; num_tries = 10
        for i in range(num_tries):
            qpos = self.ee_pos_to_qpos(
                tgt_ee_pos, tgt_ee_quat, fing_dist,
                left_ee_pos, left_ee_quat, left_fing_dist)
            if qpos is not None: break  # ok solution found
        if qpos is None:
            if debug: print('ee pos not good:', tgt_ee_pos, tgt_ee_quat)
        else:
            self.move_to_qpos(qpos, mode=mode)

    def action_low_high_ranges(self):
        if self.control_mode == 'ee_position':  # EE pos, quat, fing dist
            low = np.array([-1,-1,0, -1,-1,-1,-1, 0.0])
            high = np.array([1,1,1, 1,1,1,1, self.get_max_fing_dist()])
        elif self.control_mode == 'position':
            low = self.get_minpos()
            high = self.get_maxpos()
        elif self.control_mode == 'velocity':
            low = -self.get_maxvel()
            high = self.get_maxvel()
        elif self.control_mode == 'torque':
            low = -self.get_maxforce()
            high = self.get_maxforce()
        else:
            assert(False)  # unknown control mode
        return low, high

    def apply_action(self, action):
        if self.control_mode == 'ee_position':
            des_ee_pos = action[0:3]
            des_ee_quat = action[3:7]
            assert(np.isclose(np.linalg.norm(des_ee_quat), 1.0))  # invalid quat
            des_fing_dist = action[7]
            des_left_ee_pos = None; des_left_ee_quat = None;
            des_left_fing_dist = None
            if action.shape[0] > 8:
                ofst = 8;
                des_left_ee_pos = action[ofst:ofst+3]
                des_left_ee_quat = action[ofst+3:ofst+3+7]
                des_left_fing_dist = action[ofst+7]
            self.move_to_ee_pos(
                des_ee_pos, des_ee_quat, fing_dist=des_fing_dist,
                left_ee_pos=des_left_ee_pos, left_ee_quat=des_left_ee_quat,
                left_fing_dist=des_left_fing_dist,
                mode=pybullet.POSITION_CONTROL)
        elif self.control_mode == 'position':
            print('qpos', action, 'kp', self.kp, 'kd', self.kd)
            self.move_to_qpos(action, mode=pybullet.POSITION_CONTROL,
                              kp=self.kp, kd=self.kd)
        elif self.control_mode == 'velocity':
            self.move_with_qvel(action, mode=pybullet.VELOCITY_CONTROL,
                                kp=self.kp, kd=self.kd)
        elif self.control_mode == 'torque':
            self.apply_joint_torque(action, compensate_gravity=True)
        else:
            assert(False)  # unknown control mode

    def apply_joint_torque(self, torque, compensate_gravity=True):
        if np.allclose(torque, 0): return  # nothing to do
        torque = np.copy(torque)
        if compensate_gravity:
            #gcomp_torque = self.compute_bullet_gravity_compensation()
            gcomp_torque = self.inverse_dynamics(np.zeros_like(torque))
            torque += gcomp_torque
        # final clip check and command torques
        torque = np.clip(
            torque, -1.0*self.info.joint_maxforce, self.info.joint_maxforce)
        self.sim.setJointMotorControlArray(
            bodyIndex=self.info.robot_id, jointIndices=self.info.joint_ids,
            controlMode=pybullet.TORQUE_CONTROL, forces=torque.tolist())
        self.sim.stepSimulation()
        self.obey_joint_limits()

    def get_ee_jacobian(self, left=False):
        qpos = self.get_qpos(); qvel = self.get_qvel()
        ee_link_id = self.info.ee_link_id
        if left: ee_link_id = self.info.left_ee_link_id
        J_lin, J_ang = self.sim.calculateJacobian(
            bodyUniqueId=self.info.robot_id, linkIndex=ee_link_id,
            localPosition=[0, 0, 0],
            objPositions=qpos.tolist(), objVelocities=qvel.tolist(),
            objAccelerations=[0]*self.info.dof)
        return np.array(J_lin), np.array(J_ang)

    def inverse_dynamics(self, des_acc):
        qpos = self.get_qpos(); qvel = self.get_qvel()
        torques = self.sim.calculateInverseDynamics(
            self.info.robot_id, qpos.tolist(), qvel.tolist(), des_acc.tolist())
        return np.array(torques)

    def obey_joint_limits(self):
        qpos = self.get_qpos()
        for jid in range(self.info.dof):
            jpos = qpos[jid]; jvel = 0; ok = True
            if jpos < self.info.joint_minpos[jid]:
                jpos = self.info.joint_minpos[jid]; ok = False
            if jpos > self.info.joint_maxpos[jid]:
                jpos = self.info.joint_maxpos[jid]; ok = False
            if not ok:
                if self.debug_level>0:
                    print('fixing joint', self.info.joint_ids[jid],
                          'from pos', qpos[jid], 'to', jpos)
                self.sim.resetJointState(
                    bodyUniqueId=self.info.robot_id,
                    jointIndex=self.info.joint_ids[jid],
                    targetValue=jpos, targetVelocity=jvel)
        qpos = self.get_qpos()
        assert((qpos>=self.info.joint_minpos).all())
        assert((qpos<=self.info.joint_maxpos).all())

    def render_debug(self, width=600):
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target, distance=self.cam_dist,
            yaw=self.cam_yaw, pitch=self.cam_pitch, roll=0, upAxisIndex=2)
        height = width
        proj_matrix = self.sim.computeProjectionMatrixFOV(
            fov=90, aspect=float(width)/height, nearVal=0.01, farVal=100.0)
        w, h, rgba_px, depth_px, segment_mask = self.sim.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_TINY_RENDERER)
        #import scipy.misc
        #scipy.misc.imsave('/tmp/outfile.png', rgba_px)
        return rgba_px  # HxWxRGBA uint8

    def create_visual_area(self, shape, center, radius, rgba):
        assert(shape==pybullet.GEOM_SPHERE or pybullet.GEOM_CYLINDER)
        kwargs = {'shapeType':shape}
        viz_kwargs = copy(kwargs); col_kwargs = copy(kwargs)
        if shape == pybullet.GEOM_SPHERE or shape == pybullet.GEOM_CYLINDER:
            viz_kwargs['radius'] = radius
            col_kwargs['radius'] = radius
        if shape == pybullet.GEOM_CYLINDER:
            viz_kwargs['length'] = 0.02; col_kwargs['height'] = 0.02
        if shape == pybullet.GEOM_BOX:
            viz_kwargs['halfExtents'] = radius
            col_kwargs['halfExtents'] = radius
        viz_shp_id = pybullet.createVisualShape(
            visualFramePosition=[0,0,0], rgbaColor=rgba, **viz_kwargs)
        col_shp_id = pybullet.createCollisionShape(
            collisionFramePosition=[100,100,100], **col_kwargs)  # outside
        # Only using this for visualization, so mass=0 (fixed body).
        sphere_body_id = pybullet.createMultiBody(
            baseMass=0, baseInertialFramePosition=[0,0,0],
            baseCollisionShapeIndex=col_shp_id,
            baseVisualShapeIndex=viz_shp_id,
            basePosition=center, useMaximalCoordinates=True)
        return sphere_body_id

    def compute_bullet_gravity_compensation(self):
        # This function implements a simple gravity compensation solution
        # as suggested in PyBullet docs on page 70.
        gcomp_torques = []
        qpos = self.get_qpos()
        self._aux_sim.setJointMotorControlArray(
            bodyUniqueId=self.info.robot_id, jointIndices=self.info.joint_ids,
            targetPositions=qpos.tolist(),
            controlMode=pybullet.POSITION_CONTROL)
        self._aux_sim.stepSimulation()
        for jid, j in enumerate(self.info.joint_ids):
            _, _, _, torque_applied = self._aux_sim.getJointState(
                self.info.robot_id, j)
            gcomp_torques.append(torque_applied)
        return np.array(gcomp_torques)

    def ee_pos_to_qpos(self, ee_pos, ee_quat, fing_dist,
                       left_ee_pos=None, left_ee_quat=None, left_fing_dist=0.0,
                       debug=False):
        qpos = self._ee_pos_to_qpos_raw(
            ee_pos, ee_quat, fing_dist,
            left_ee_pos, left_ee_quat, left_fing_dist, debug=debug)
        ok = self.collisions_ok(qpos)
        if not ok and debug:
            print('ee_pos not ok', ee_pos, ee_quat, fing_dist)
            euler = self._aux_sim.getEulerFromQuaternion(ee_quat)
            print('euler', euler); print('qpos', qpos)
        return qpos if ok else None

    def collisions_ok(self, qpos, debug=False):
        ok = True
        for jid in range(self.info.dof):
            self._aux_sim.resetJointState(
                bodyUniqueId=self.info.robot_id,
                jointIndex=self.info.joint_ids[jid],
                targetValue=qpos[jid], targetVelocity=0)
        self._aux_sim.stepSimulation()
        # Check collision with ground/table.
        pts = self._aux_sim.getContactPoints()
        for pt in pts:
            body_bullet_ids = [pt[1], pt[2]]  # getContactPoints() docs p.43
            if ((self.info.robot_id in body_bullet_ids) and
                (self.plane_id in body_bullet_ids)):
                if body_bullet_ids[0] == self.info.robot_id:
                    rbt_link = pt[3]
                else:
                    rbt_link = pt[4]
                if rbt_link==0: continue  # base contact w/ground?
                if debug:
                    msg = 'robot link {:d} would collide with ground/table'
                    print(msg.format(rbt_link))
                ok = False
                break
        return ok
