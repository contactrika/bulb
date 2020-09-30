"""
Base class for visually-realistic env for re-arrangement tasks.
"""
import os

import gym
import pybullet

from ..utils import render_utils
from .aux_env import AuxEnv
from .rearrange_utils import *


class RearrangeEnv(gym.Env, AuxEnv):
    OBJECT_XYS = np.array(
        [[0.24,0.12],  [0.03,-0.12], [0.25,-0.12], [0.05,0.16]])
    OBJECT_EULERS = [[0,0,0],[0,0,np.pi/2],[0,0,0],[0,0,0]]
    OBJECT_LISTS = {
        'Ycb': [
            ['002_master_chef_can', '009_gelatin_box', '013_apple', '016_pear'],
            ['007_tuna_fish_can', '009_gelatin_box', '014_lemon', '017_orange'],
            ['002_master_chef_can', '004_sugar_box', '016_pear', '014_lemon'],
            ['005_tomato_soup_can', '004_sugar_box', '014_lemon', '013_apple'],
            # lists with unseen objects (plum, pudding box)
            ['002_master_chef_can', '008_pudding_box', '017_orange', '018_plum'],
            ['004_sugar_box', '017_orange', '018_plum', '008_pudding_box']],
        'OneYcb':[
            ['005_tomato_soup_can'], ['004_sugar_box'], ['002_master_chef_can'],
            ['008_pudding_box'], ['016_pear'], ['008_pudding_box']],
        'Geom':([['cylinder.urdf', 'cube.urdf', 'sphere.urdf', 'sphere1.urdf']]*5+
                [['cube.urdf', 'sphere.urdf', 'sphere.urdf', 'cube.urdf']]),
        'OneGeom':[['cylinder.urdf'], ['cube.urdf'], ['cylinder.urdf'],
                   ['cube.urdf'], ['sphere1.urdf'], ['cube.urdf']]
    }
    TABLE_TEXTURES = ['_birch','_gray','_orange','_blue','_green','_birch']
    TABLE_TEXTURES_RGB = np.array(
        [[220,200,170],[165,165,165], [165,75,40], [82,60,124],
         [90,190,180], [220,200,170]], dtype=float)/255.0
    TABLE_TEXTURES += ['']*len(TABLE_TEXTURES)  # simple color bkgrnd
    GEOM_COLORS = [[[0,0,1],[1,0,1],[1,0,0],[1,1,0]],
                   [[1,1,0],[0,1,1],[1,0,1],[0,1,0]],
                   [[0,1,0],[0,0,1],[1,1,0],[0,1,1]],
                   [[1,0,0],[1,1,0],[0,1,0],[1,0,1]],
                   [[1,0,1],[1,1,0],[1,0,0],[0.9,0.9,0.9]],
                   [[0,1,1],[0,1,0],[1,1,0],[0,0,1]]]*2
    OBJ_XYZ_MINS = np.array([-0.40, -0.40, 0.00])-0.05  # 5cm slack
    OBJ_XYZ_MAXS = np.array([ 0.40,  0.40, 0.15])+0.05  # 5cm slack
    PTCLOUD_BOX_SIZE = 0.5  # 50cm cube cut off for point cloud observations
    # For param randomization.
    SIM_PARAM_INFO = {'RGB_R':0, 'RGB_G':1, 'RGB_B':2}  # background RGB
    SIM_PARAM_MINS = np.array([0.0, 0.0, 0.0])
    SIM_PARAM_MAXS = np.array([1.0, 1.0, 1.0])
    # [*rgba[0:3], *dims, shape, mass, restit, lat_fric, rol_fric, spin_fric]:
    SIM_PARAM_OBJ_PROPS_NAMES = [
        'OBJ_RGB_R', 'OBJ_RGB_G', 'OBJ_RGB_B', 'RADIUS_A', 'RADIUS_B', 'HEIGHT',
        'SHAPE', 'MASS', 'RESTITUTION', 'LATERAL_FRICTION', 'ROLING_FRICTION',
        'SPINNING_FRICTION']
    SIM_PARAM_OBJ_MINS = np.array(
        [0, 0, 0,  0.01, 0.01, 0.01,  0,  0.01,  0.000,  0.000, 0.000, 0.000])
    SIM_PARAM_OBJ_MAXS = np.array(
        [1, 1, 1,  0.15, 0.15, 0.15,  3,  1.00,  1.000,  1.000, 0.010, 0.010])
    GEOM_FROM_SHAPE = [pybullet.GEOM_SPHERE, pybullet.GEOM_CYLINDER,
                       pybullet.GEOM_BOX, pybullet.GEOM_MESH]
    NAME_FROM_SHAPE = ['geomsphere', 'geomcylinder', 'geombox', 'mesh']
    # Target areas:
    #   left for packaged food (boxes, cylinders)
    #   right for fruit (spheres)
    # Other colors for later: orange [1, 0.5, 0, 0.5]
    TARGETS_POS = np.array([[0.2, -0.2, 0.01], [0.2, 0.2, 0.01]])
    TARGETS_RGBA = np.array([[0.6, 0.6, 0.9, 0.7], [0.9, 0.9, 0.9, 0.7]])
    # Note: GEOM_CYLINDER loaded with loadURDF() causes getDynamicsInfo
    # to not report dims or inertia diagonal correctly; so use hard-coded init.
    CYLINDER_DEFAULT_DIMS = [0.06, 0.06, 0.12/2]

    def __init__(self, version, variant, obs_resolution, obs_ptcloud,
                 rnd_init_pos, debug=False):
        self._version = version
        self._variant = variant
        self._rnd_init_pos = rnd_init_pos
        self._num_init_rnd_act = 10 if rnd_init_pos else 0
        self._max_episode_steps = 50
        self._num_action_repeats = 4 # apply same torque action k num sim steps
        self._obs_resolution = obs_resolution
        self._obs_ptcloud = obs_ptcloud
        self._debug = debug
        self._stepnum = 0
        self._episode_rwd = 0.0
        assert(hasattr(self, 'robot'))  # self.robot should be set
        self._max_torque = self.robot.get_maxforce()
        self._ndof = self._max_torque.shape[0]
        if self._ndof == 9:  # 7DoF manipulator with 2 last joints being fingers
            self._ndof = 7
            self._max_torque[7:] = 0  # ignore finger joints
        data_dir = os.path.join(os.path.split(__file__)[0], 'data')
        # Load borders.
        brpfx = os.path.join(data_dir, 'table', 'borders')
        borders_file = brpfx+RearrangeEnv.TABLE_TEXTURES[self._version]+'.urdf'
        self._border_id = self.robot.sim.loadURDF(
            borders_file, useFixedBase=True, globalScaling=0.8)
        if RearrangeEnv.TABLE_TEXTURES[self._version]== '':
            self.robot.sim.changeVisualShape(
                self._border_id, 0, rgbaColor=(0,0,0,1))  # black background
        self._blank_texture_id = self.robot.sim.loadTexture(
            os.path.join(data_dir, 'table', 'table.png'))
        # Load objects.
        res = self.load_objects(self.robot.sim, data_dir)
        self._object_names, self._init_object_poses, self._init_object_quats, \
            self._object_ids, self._object_props = res
        # Initialize data needed for point cloud observations.
        self._cam_object_ids = None
        if self._obs_ptcloud:
            if hasattr(self.robot, 'robot_id'):
                self._cam_object_ids = [self.robot.robot_id]
            else:
                assert(self.robot, 'info')
                self._cam_object_ids = [self.robot.info.robot_id]
            self._cam_object_ids.extend(self._object_ids)
        # Define obs and action space shapes.
        # Note: in this env aux state in this envs is normalized to [-1,1]
        # (not on [0,1] to avoid rescaling sin,cos,rot dims).
        self._ld_names = self.compute_low_dim_state_names()
        self._ld_space = gym.spaces.Box(
                -1.0, 1.0, shape=[len(self._ld_names)], dtype=np.float32)
        if self._obs_resolution is None:
            self.observation_space = self._ld_space
        elif self._obs_ptcloud:
            state_sz = self._obs_resolution * 3  # 3D points in point cloud
            self.observation_space = gym.spaces.Box(
                -1.0*RearrangeEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz),
                RearrangeEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz))
        else:  # RGB images
            self.observation_space = gym.spaces.Box(  # channels: 3 color
                0.0, 1.0, shape=[3, self._obs_resolution, self._obs_resolution],
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=[self._ndof], dtype=np.float32)
        super(RearrangeEnv, self).__init__()
        if debug:
            print('Created RearrangeEnv with observation_space',
                  self.observation_space, 'action_space', self.action_space,
                  'action limits', self.action_space.low, self.action_space.high,
                  'low_dim_observation_space', self._ld_space,
                  'low_dim_state_names', self._ld_names)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @property
    def low_dim_state_space(self):
        return self._ld_space

    @property
    def low_dim_state_names(self):
        return self._ld_names

    def seed(self, seed):
        np.random.seed(seed)

    def close(self):
        self.robot.disconnect()

    def reset(self):
        self._stepnum = 0
        self._episode_rwd = 0.0
        if self._rnd_init_pos:
            self.reset_to_random_pose()
        else:
            self.robot.reset()
            self.robot.reset_objects(
                self._object_ids, self._init_object_poses, self._init_object_quats)
        # Make initial random actions, then return the starting state.
        for t in range(self._num_init_rnd_act):
            rnd01 = np.random.rand(self._max_torque.shape[0])
            torque = (rnd01-0.5)*self._max_torque
            self.robot.apply_joint_torque(torque)
        return self.compute_obs()

    def step(self, action):
        # Assume: robot is torque controlled and action is scaled to [0,1]
        torque = np.hstack(
            [action, np.zeros(self._max_torque.shape[0] - self._ndof)])
        torque = np.clip((torque-0.5) * 2 * self._max_torque,
                         -self._max_torque, self._max_torque)
        # Apply torque action to joints
        for sub_step in range(self._num_action_repeats):  # repeat for faster sim
            self.robot.apply_joint_torque(torque)  # advances sim inside
            if not self.in_workspace(): break
        next_obs = self.compute_obs()
        ee_ok = self.in_workspace()
        # Update internal counters.
        self._stepnum += 1
        # Report reward starts and other info.
        rwd = self.compute_reward()
        self._episode_rwd += rwd
        info = {}
        done = (self._stepnum == self._max_episode_steps) or not ee_ok
        if self._debug and self._stepnum%10==0:
            print('step', self._stepnum, 'action', action, 'torque', torque)
            if self._obs_resolution is None: print('state', next_obs)
        if done:
            # Unused steps get a reward same as the last ok step
            self._episode_rwd += rwd * (self._max_episode_steps - self._stepnum)
            info['episode'] = {'r': float(self._episode_rwd), 'l': self._stepnum}
            if self._debug: print('tot_rwd {:.4f}'.format(self._episode_rwd))
        return next_obs, rwd, done, info

    def override_state(self, ld_state):
        qpos, obj_props, obj_poses, obj_quats = \
            self.low_dim_to_sim_state(ld_state)
        self.robot.reset_to_qpos(qpos)
        self._object_props[:] = obj_props[:]
        for objid in range(len(self._object_ids)):
            # Do not override object properties, since we use mesh objects.
            #self.override_object_properties()
            self.robot.sim.resetBasePositionAndOrientation(
                self._object_ids[objid], obj_poses[objid].tolist(),
                obj_quats[objid].tolist())

    def render(self, mode='rgb_array', close=False):
        pass  # done by pybullet GUI

    def render_obs(self, override_resolution=None, debug_out_dir=None):
        obs = render_utils.render_obs(
            self.robot.sim, self._stepnum,
            self._obs_resolution, override_resolution,
            self.robot.cam_dist, self.robot.cam_target,
            self.robot.cam_pitch, self.robot.cam_yaw,
            self._obs_ptcloud, self._cam_object_ids, 'Rearrange',
            RearrangeEnv.PTCLOUD_BOX_SIZE, debug_out_dir)
        return obs

    def compute_obs(self):
        if self._obs_resolution is None:
            return self.compute_low_dim_state()
        else:
            return self.render_obs()

    def compute_reward(self):
        dists = []
        for i, obj_id in enumerate(self._object_ids):
            nm = self._object_names[i]
            obj_pos, _ = self.robot.sim.getBasePositionAndOrientation(obj_id)
            tgt_id = 0 if nm.endswith(('_box', '_can')) else 1
            tgt_pos = RearrangeEnv.TARGETS_POS[tgt_id]
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(tgt_pos[:2]))
            dists.append(min(dist,1.0))   # max distance is 1 meter
        mean_dist = np.array(dists).mean()
        rwd = (1.0-mean_dist)/float(self._max_episode_steps)
        return rwd

    def compute_sim_state(self):
        obj_props = np.copy(self._object_props)
        obj_poses = []; obj_quats = []
        for obj in self._object_ids:
            obj_pos, obj_quat = self.robot.sim.getBasePositionAndOrientation(obj)
            obj_poses.append(obj_pos)
            obj_quats.append(obj_quat)
        obj_poses = np.array(obj_poses)
        # Don't care about reporting object positions outside of workspace
        obj_poses = np.clip(obj_poses, RearrangeEnv.OBJ_XYZ_MINS,
                            RearrangeEnv.OBJ_XYZ_MAXS)
        obj_quats = np.array(obj_quats)
        return obj_props, obj_poses, obj_quats

    def compute_low_dim_state(self):
        obj_props, obj_poses, obj_quats = self.compute_sim_state()
        ld_state = self.sim_to_low_dim_state(
            obj_props, obj_poses, obj_quats)
        return ld_state

    def compute_low_dim_state_names(self):
        ld_names = []
        obj_props, obj_poses, obj_quats = self.compute_sim_state()
        qpos = self.robot.get_qpos()
        for j in range(qpos.shape[0]):
            ld_names.append('j'+str(j)+'_sin')
            ld_names.append('j'+str(j)+'_cos')
        num_obj = len(obj_props); props_sz = obj_props.shape[1]
        for i in range(num_obj):
            shape_id = int(self._object_props[i][
                RearrangeEnv.SIM_PARAM_OBJ_PROPS_NAMES.index('SHAPE')])
            obj_shape_name = RearrangeEnv.NAME_FROM_SHAPE[shape_id]
            if obj_shape_name=='': obj_shape_name = 'obj'
            obj_pfx = obj_shape_name+str(i)
            for j in range(props_sz):
                #nm = RearrangeEnv.SIM_PARAM_OBJ_PROPS_NAMES[j]
                ld_names.append('obj'+str(i)+'_prop'+str(j))
            ld_names.append(obj_pfx+'_x')
            ld_names.append(obj_pfx+'_y')
            ld_names.append(obj_pfx+'_z')
            for j in range(9): ld_names.append(obj_pfx+'_rot'+str(j))
        # Sanity checks.
        ld_state = self.sim_to_low_dim_state(obj_props, obj_poses, obj_quats)
        if ld_state.shape[0]!=len(ld_names):
            print('ld_state.shape[0]', ld_state.shape[0],
                  ' vs len(ld_names)', len(ld_names), ld_names)
        assert(ld_state.shape[0]==len(ld_names))
        return ld_names

    def reset_to_random_pose(self):
        if hasattr(self, 'rnd_qpos_fxn'):
            rnd_qpos = None
            while rnd_qpos is None:
                rnd_qpos, end_ee_pos, rnd_ee_quat = self.rnd_qpos_fxn()
                if self._debug:
                    print('qpos_for_random_ee_pose', rnd_qpos, 'end_ee_pos',
                          end_ee_pos, 'rnd_ee_quat', rnd_ee_quat)
        else:  # random from [-pi,pi] for each joint (minus eps for stability)
            rnd_qpos = (np.random.rand(self._ndof) - 0.5) * 2 * (0.75 * np.pi)
        self.robot.reset_to_qpos(rnd_qpos)
        # TODO: update this when using random object shapes
        min_z = 0.03 if 'Ycb' in self._variant else 0.04
        xyrng = 0.15
        mins = np.array([-xyrng, -xyrng, min_z])
        maxes = np.array([xyrng, xyrng, 0.075])
        for objid in range(len(self._object_ids)):
            obj_pos = np.random.rand(3)
            obj_pos = denormalize(obj_pos, mins, maxes)
            obj_quat = np.random.rand(4)
            obj_quat = obj_quat/np.linalg.norm(obj_quat)
            self.robot.sim.resetBasePositionAndOrientation(
                self._object_ids[objid], obj_pos, obj_quat)
        for t in range(20):  # let the objects emerge from problematic poses
            self.robot.sim.stepSimulation()

    def sim_to_low_dim_state(self, obj_props, obj_poses, obj_quats):
        # Add robot qpos.
        qpos = self.robot.get_qpos()
        #assert((qpos>=-np.pi).all() and (qpos<=np.pi).all())
        qpos_all_sin = np.sin(qpos).reshape(-1,1)
        qpos_all_cos = np.cos(qpos).reshape(-1,1)
        qpos_all_sin_cos = np.hstack([qpos_all_sin, qpos_all_cos])
        if obj_props is None: return qpos_all_sin_cos.reshape(-1)
        # Add object properties and dynamics (pos, ori).
        if ((obj_props<RearrangeEnv.SIM_PARAM_OBJ_MINS).any() or
            (obj_props>RearrangeEnv.SIM_PARAM_OBJ_MAXS).any()):
            print('obj_props outside min/max bounds'); print(obj_props)
            print('deltas:')
            print(obj_props-RearrangeEnv.SIM_PARAM_OBJ_MINS)
            print(RearrangeEnv.SIM_PARAM_OBJ_MAXS-obj_props)
            assert(False)  # obj_props outside min/max bounds
        obj_props_normed = normalize(
            obj_props, RearrangeEnv.SIM_PARAM_OBJ_MINS,
            RearrangeEnv.SIM_PARAM_OBJ_MAXS)
        if ((obj_poses<RearrangeEnv.OBJ_XYZ_MINS).any() or
            (obj_poses>RearrangeEnv.OBJ_XYZ_MAXS).any()):
            print('obj_poses outside min/max bounds', obj_poses)
            assert(False)  # obj_poses outside min/max bounds
        obj_poses_normed = normalize(
            obj_poses, RearrangeEnv.OBJ_XYZ_MINS, RearrangeEnv.OBJ_XYZ_MAXS)
        all_obj_rot = quat2mat(np.array(obj_quats))
        ld_state = np.hstack([
            qpos_all_sin_cos.reshape(-1), obj_props_normed.reshape(-1),
            obj_poses_normed.reshape(-1), all_obj_rot.reshape(-1)])
        return ld_state

    def low_dim_to_sim_state(self, ld_state):
        assert(ld_state.min() >= -1 and ld_state.max() <= 1)
        # Parse flat_state as static and dynamic simulation state.
        qpos_sz = self._ndof * 2  # dof*2 since angle as sin,cos
        # Parse out qpos. This always comes 1st, since it is always known.
        qpos_all_sin_cos = ld_state[0:qpos_sz]
        qpos = sin_cos_to_theta(qpos_all_sin_cos)
        assert(qpos.min() >= -np.pi and qpos.max() <= np.pi)
        if ld_state.shape[0]==qpos_sz: return qpos, None, None, None, None
        # Parse objects static and dynamic state.
        num_obj = len(self._object_ids)
        obj_prop_sz = len(RearrangeEnv.SIM_PARAM_OBJ_PROPS_NAMES) # static state
        obj_dyn_sz = 3+9  # obj pos (3D), ori as 3x3 rotation matrix
        obj_state_sz = num_obj*(obj_prop_sz + obj_dyn_sz)
        assert(ld_state.shape[0] == (qpos_sz + obj_state_sz))
        ofst_strt = qpos_sz; ofst_fnsh = ofst_strt+num_obj*obj_prop_sz
        obj_props_normed = ld_state[ofst_strt:ofst_fnsh]
        ofst_strt = ofst_fnsh; ofst_fnsh = ofst_strt+num_obj*3  # 3D poses
        obj_poses_normed = ld_state[ofst_strt:ofst_fnsh]
        ofst_strt = ofst_fnsh; ofst_fnsh = ofst_strt+num_obj*9  # 3x3 rot matrix
        all_obj_rot = ld_state[ofst_strt:ofst_fnsh]
        obj_props = denormalize(
            obj_props_normed.reshape(num_obj, obj_prop_sz),
            RearrangeEnv.SIM_PARAM_OBJ_MINS, RearrangeEnv.SIM_PARAM_OBJ_MAXS)
        obj_poses = denormalize(
            obj_poses_normed.reshape(num_obj, 3),
            RearrangeEnv.OBJ_XYZ_MINS, RearrangeEnv.OBJ_XYZ_MAXS)
        obj_quats = mat2quat(all_obj_rot.reshape(-1,3,3))
        return qpos, obj_props, obj_poses, obj_quats

    def override_object_properties(
            self, objid, obj_props, obj_poses, obj_quats, debug=False):
        # Change color, shape, mass, friction, restitution of the object.
        rgb_r, rgb_g, rgb_b, radius_a, radius_b, halfExtent, shape, mass, \
            restitution, lat_fric, rol_fric, spin_fric = obj_props[objid]
        viz_geom_type = RearrangeEnv.GEOM_FROM_SHAPE[int(round(shape))]
        if viz_geom_type==pybullet.GEOM_MESH:
            viz_geom_type = pybullet.GEOM_CYLINDER
        if debug:
            print('override_object_properties with obj_props', obj_props)
            msg = 'createMultiBody for obj {:d}: rgb {:.4f} {:.4f} {:.4f}'
            msg += ' radius_a {:.4f} radius_b {:.4f} halfExtent {:.4f}'
            print(msg.format(objid, rgb_r, rgb_g, rgb_b,
                             radius_a, radius_b, halfExtent))
            print('pos', obj_poses[objid])
            msg = 'changeDynamics: mass {:.4f} restitution {:.4f}'
            msg += ' lat_fric {:.4f} roll_fric {:.4f} spin_fric {:.4f}'
            print(msg.format(mass, restitution, lat_fric, rol_fric, spin_fric))
        # No way to update collision shape, need to remove and re-insert.
        # https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=12034
        self.robot.sim.removeBody(self._object_ids[objid])
        new_viz_shape_id = pybullet.createVisualShape(
            shapeType=viz_geom_type, rgbaColor=[rgb_r, rgb_g, rgb_b, 1.0],
            radius=radius_a, length=halfExtent*2,
            halfExtents=[radius_a, radius_b, halfExtent])
        new_col_shape_id = pybullet.createCollisionShape(
            shapeType=viz_geom_type,
            radius=radius_a, height=halfExtent*2,
            halfExtents=[radius_a, radius_b, halfExtent])
        new_obj_id = pybullet.createMultiBody(
            baseMass=mass, basePosition=obj_poses[objid].tolist(),
            baseOrientation=obj_quats[objid].tolist(),
            baseVisualShapeIndex=new_viz_shape_id,
            baseCollisionShapeIndex=new_col_shape_id)
        self._object_ids[objid] = new_obj_id
        # TODO: Check whether pybullet changes inertial matrix based on
        #  mass and shape type, or whether we need to re-specify manually.
        self.robot.sim.changeDynamics(
            self._object_ids[objid], -1, mass=mass, restitution=restitution,
            lateralFriction=lat_fric, rollingFriction=rol_fric,
            spinningFriction=spin_fric)

    def in_workspace(self):
        # First check whether all the objects are still in the workspace.
        for tmpi, obj in enumerate(self._object_ids):
            obj_pos, obj_quat = self.robot.sim.getBasePositionAndOrientation(obj)
            if ((np.array(obj_pos)<RearrangeEnv.OBJ_XYZ_MINS).any() or
                (np.array(obj_pos)>RearrangeEnv.OBJ_XYZ_MAXS).any()):
                if self._debug:
                    print('Object', self._object_names[tmpi], 'outside range')
                return False
        # Now do robot-related checks.
        ee_pos = self.robot.get_ee_pos()
        # Check whether ee is too high above the objects.
        if hasattr(self, 'max_object_z') and ee_pos[2] > 0.2+self.max_object_z:
            if self._debug:
                print('step', self._stepnum, 'ee_pos too high', ee_pos)
            return False
        # Check that ee is above the table crate.
        if (ee_pos[0]<-0.5 or ee_pos[0]>0.5 or
            ee_pos[1]<-0.5 or ee_pos[1]>0.5):
            if self._debug:
                print('step', self._stepnum, 'ee_pos not in workspace', ee_pos)
            return False
        return True

    def load_objects(self, sim, data_dir):
        obj_lists = RearrangeEnv.OBJECT_LISTS[self._variant]
        object_names = obj_lists[self._version % len(obj_lists)]
        # Make a list of object files, their properties,
        # initial positions and orientations.
        object_poses = []; object_quats = []
        object_files = []; object_masses = []; object_scales = []
        geom_colors = RearrangeEnv.GEOM_COLORS[self._version]
        xys = RearrangeEnv.OBJECT_XYS
        max_object_z = 0
        for i, nm in enumerate(object_names):
            info = YCB_OBJECT_INFOS[nm]
            fname = nm
            if nm.startswith('0'):  # YCB mesh
                fname = os.path.join('ycb', nm, 'google_16k', 'textured_ok.obj')
            object_files.append(fname)
            z = info['z']*info['s'][2]
            if z*2 > max_object_z: max_object_z = z*2  # assume cntr
            object_poses.append(np.array([xys[i,0], xys[i,1], z]))
            object_quats.append(np.array(
                pybullet.getQuaternionFromEuler(RearrangeEnv.OBJECT_EULERS[i])))
            object_masses.append(info['m'])
            object_scales.append(info['s'])
        # Load objects into PyBullet simulation.
        object_ids = []; object_props = []
        for i in range(len(object_files)):
            object_file = os.path.join(data_dir, object_files[i])
            if object_file.endswith('.urdf'):
                obj_id = sim.loadURDF(
                    object_file, object_poses[i].tolist())
                sim.changeVisualShape(
                    obj_id, -1, rgbaColor=(*geom_colors[i],1))
            elif object_file.endswith('.obj'):  # load mesh info
                viz_shape_id = sim.createVisualShape(
                    shapeType=pybullet.GEOM_MESH, rgbaColor=None,
                    fileName=object_file, meshScale=object_scales[i])
                col_shape_id = sim.createCollisionShape(
                    shapeType=pybullet.GEOM_MESH,
                    fileName=object_file, meshScale=object_scales[i])
                obj_id = sim.createMultiBody(
                    baseMass=object_masses[i],
                    basePosition=object_poses[i].tolist(),
                    baseCollisionShapeIndex=col_shape_id,
                    baseVisualShapeIndex=viz_shape_id,
                    baseOrientation=object_quats[i].tolist())
            else:
                print('Unknown object file extension', object_file)
                assert(False)
            viz_info = sim.getVisualShapeData(obj_id)
            assert(len(viz_info)==1)
            _, _, geom, _, _, _, _, rgba, *_ = viz_info[0]
            shape = RearrangeEnv.GEOM_FROM_SHAPE.index(geom)
            # ATTENTION: pybullet not reporting dims correctly (nor geom here)
            res = sim.getCollisionShapeData(obj_id, -1)[0]
            dims = res[3]
            if geom in (pybullet.GEOM_CYLINDER, pybullet.GEOM_MESH):
                dims = RearrangeEnv.CYLINDER_DEFAULT_DIMS
            elif geom==pybullet.GEOM_BOX:  # loadURDF() doesn't actually use
                dims = np.array(dims)/2    # halfExtents, so divide here
            # ATTENTION: pybullet not reporting inert_diag correctly
            # flags=pybullet.DYNAMICS_INFO_REPORT_INERTIA not present any more.
            mass, lat_fric, inert_diag, inert_pos, ori, restit, \
                rol_fric, spin_fric, contact_damp, contact_stiff, *_ \
                = sim.getDynamicsInfo(obj_id, -1)
            assert(mass>0)
            props = [*rgba[0:3], *dims, shape, mass, restit,
                     lat_fric, rol_fric, spin_fric]
            if self._debug:
                print('Loaded', object_file, 'at', object_poses[i],
                      'props', props)
            object_ids.append(obj_id)
            object_props.append(np.array(props))
        return object_names, object_poses, object_quats, object_ids, object_props
