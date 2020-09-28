"""
Base class for visually-realistic env for re-arrangement tasks.
"""
import imageio
import os
import numpy as np
import gym
import pybullet

from ..utils.process_camera import ProcessCamera, plot_ptcloud
from .all_cam_vals import ALL_CAM_VALS
from .rearrange_utils import *


class RearrangeEnv(gym.Env):
    OBJECT_XYS = np.array(
        [[0.24,0.12],  [0.03,-0.12], [0.25,-0.12], [0.05,0.16]])
    #    [[0.16,0.08],  [0.03,-0.08], [0.17,-0.08], [0.05,0.12]])
    OBJECT_EULERS = [[0,0,0],[0,0,np.pi/2],[0,0,0],[0,0,0]]
    OBJECT_LISTS = {
        'Ycb': [['002_master_chef_can', '009_gelatin_box', '013_apple', '016_pear'],
                ['007_tuna_fish_can', '009_gelatin_box', '014_lemon', '017_orange'],
                ['002_master_chef_can', '004_sugar_box', '016_pear', '014_lemon'],
                ['005_tomato_soup_can', '004_sugar_box', '014_lemon', '013_apple'],
                # lists with unseen objects (plum, pudding box)
                ['002_master_chef_can', '008_pudding_box', '017_orange', '018_plum'],
                ['004_sugar_box', '017_orange', '018_plum', '008_pudding_box']],
        'OneYcb':[['005_tomato_soup_can'],
                  ['004_sugar_box'],
                  ['002_master_chef_can'],
                  ['008_pudding_box'],
                  ['016_pear'],
                  ['008_pudding_box']],
        'Geom':([['cylinder.urdf', 'cube.urdf', 'sphere.urdf', 'sphere1.urdf']]*5+
                [['cube.urdf', 'sphere.urdf', 'sphere.urdf', 'cube.urdf']]),
        'OneGeom':[['cylinder.urdf'],
                   ['cube.urdf'],
                   ['cylinder.urdf'],
                   ['cube.urdf'],
                   ['sphere1.urdf'],
                   ['cube.urdf']]
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
    #props = [*rgba[0:3], *dims, shape, mass, restit, lat_fric, rol_fric, spin_fric]
    SIM_PARAM_OBJ_PROPS_NAMES = [
        'OBJ_RGB_R', 'OBJ_RGB_G', 'OBJ_RGB_B', 'RADIUS_A', 'RADIUS_B', 'HEIGHT',
        'SHAPE', 'MASS', 'RESTITUTION', 'LATERAL_FRICTION', 'ROLING_FRICTION',
        'SPINNING_FRICTION']
    SIM_PARAM_OBJ_MINS = np.array(
        [0, 0, 0,  0.01, 0.01, 0.01,  0,  0.01,  0.000,  0.000, 0.000, 0.000])
    SIM_PARAM_OBJ_MAXS = np.array(
        [1, 1, 1,  0.15, 0.15, 0.15,  2,  1.00,  1.000,  1.000, 0.010, 0.010])
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

    def __init__(self, version, max_episode_steps, obs_resolution, obs_ptcloud,
                 variant, rnd_init_pos, statics_in_lowdim, debug_level=0):
        if debug_level>0: print('RearrangeEnv.__init__()...')
        # Note: RearrangeEnv expects that we created self.robot already.
        assert(hasattr(self, 'robot'))
        assert((obs_resolution in [64, 128, 256]) or
               obs_resolution is None or obs_ptcloud)
        self.version = version
        self.variant = variant
        self.remove_robot = False  # for debugging
        self.rnd_init_pos = rnd_init_pos
        self.num_init_rnd_act = 10 if rnd_init_pos else 0
        self.statics_in_lowdim = statics_in_lowdim
        self._max_episode_steps = max_episode_steps
        self.num_action_repeats = 4 # apply same torque action k num sim steps
        self.obs_resolution = obs_resolution
        self.obs_ptcloud = obs_ptcloud
        self.debug_level = debug_level
        self.max_torque = self.robot.get_maxforce()
        self.ndof = self.max_torque.shape[0]
        if self.ndof == 9:  # 7DoF manipulator with 2 last joints being fingers
            self.ndof = 7
            self.max_torque[7:] = 0  # ignore finger joints
        # Load table or borders.
        data_folder = os.path.split(__file__)[0]
        #sim.resetBasePositionAndOrientation(
        #    robot.robot_id, [0,0,RearrangeEnv.TABLE_HEIGHT], [0,0,0,1])
        #table_file = os.path.join(data_folder, 'data', 'table', 'table.urdf')
        #self.table_id = sim.loadURDF(
        #    table_file, [0,0,0], useFixedBase=True,
        #    globalScaling=RearrangeEnv.TABLE_SCALING)
        brpfx = os.path.join(data_folder, 'data', 'table', 'borders')
        borders_file = brpfx+RearrangeEnv.TABLE_TEXTURES[version]+'.urdf'
        self.border_id = self.robot.sim.loadURDF(
            borders_file, useFixedBase=True, globalScaling=0.8)
        if RearrangeEnv.TABLE_TEXTURES[version]!='':
            self.bkgrnd = RearrangeEnv.TABLE_TEXTURES_RGB[version]
        else:
            self.bkgrnd = np.array([0,0,0])
            self.robot.sim.changeVisualShape(self.border_id, 0,
                                             rgbaColor=(*self.bkgrnd,1))
        self.blank_texture_id = self.robot.sim.loadTexture(
            os.path.join(data_folder, 'data', 'table', 'table.png'))
        # Note: create_visual_area() impacts physics, so use only for debug
        #if debug_level>:
        #    for i in range(len(RearrangeEnv.TARGETS_POS)):
        #        self.robot.create_visual_area(
        #            pybullet.GEOM_SPHERE, RearrangeEnv.TARGETS_POS[i],
        #            radius=0.01, rgba=RearrangeEnv.TARGETS_RGBA[i])
        # Load objects.
        obj_lists = RearrangeEnv.OBJECT_LISTS[self.variant]
        num_obj_groups = len(obj_lists)
        self.object_names = obj_lists[version%num_obj_groups]
        self.max_object_z = 0  # maximum height of objects
        self.init_object_poses = []
        #quat = pybullet.getQuaternionFromEuler([0, 0, 0])
        self.init_object_quats = []
        geom_colors = RearrangeEnv.GEOM_COLORS[version]
        obj_files = []; obj_masses = []; obj_scales = []
        print('Loading objects', self.object_names, 'geom_colors', geom_colors)
        xys = RearrangeEnv.OBJECT_XYS
        #if self.random_object_init_slot:
        #    xys = xys[np.random.permutation(xys.shape[0])]
        for i, nm in enumerate(self.object_names):
            info = YCB_OBJECT_INFOS[nm]
            fname = nm
            if nm.startswith('0'):  # YCB mesh
                fname = os.path.join('ycb', nm, 'google_16k', 'textured_ok.obj')
            obj_files.append(fname)
            z = info['z']*info['s'][2]
            if z*2 > self.max_object_z: self.max_object_z = z*2  # assume cntr
            self.init_object_poses.append(np.array([xys[i,0], xys[i,1], z]))
            self.init_object_quats.append(np.array(
                pybullet.getQuaternionFromEuler(RearrangeEnv.OBJECT_EULERS[i])))
            obj_masses.append(info['m'])
            obj_scales.append(info['s'])
        self.object_ids, self.object_props = RearrangeEnv.load_objects(
            self.robot.sim, obj_files, self.init_object_poses,
            self.init_object_quats, obj_masses, obj_scales, geom_colors)
        # Initialize data needed for point cloud observations.
        if 'Rearrange' not in ALL_CAM_VALS.keys():
            # Print camera info to store in _cam_vals
            # Note: has to be done after env reset().
            ProcessCamera.compute_cam_vals(  # for point clouds
                cam_dist=self.robot.cam_dist, cam_tgt=self.robot.cam_target,
                cam_yaws=ProcessCamera.CAM_YAWS,
                cam_pitches=ProcessCamera.CAM_PITCHES)
            input('continue')
        for tmpi in range(self.robot.sim.getNumBodies()):
            robot_name, scene_name = self.robot.sim.getBodyInfo(tmpi)
            robot_name = robot_name.decode("utf8")
            body_id = self.robot.sim.getBodyUniqueId(tmpi)
            print('_cam_object_ids:', body_id, robot_name, scene_name)
            #self._cam_object_ids.append(body_id)
        if self.obs_ptcloud:
            self.cam_vals = ALL_CAM_VALS['Rearrange']  # needed for pt_clouds
            if not hasattr(self.robot, 'robot_id'):
                self.robot.robot_id = self.robot.info.robot_id
            self._cam_object_ids = [self.robot.robot_id]
            self._cam_object_ids.extend(self.object_ids)
        # Define obs and action space shapes.
        self.aux_nms = []  # names for low-dim state (flat_state)
        _, aux = self.get_obs_and_aux(aux_nms_to_fill=self.aux_nms)
        if self.obs_resolution is None:
            self.observation_space = gym.spaces.Box(
                0.0, 1.0, shape=[len(self.aux_nms)], dtype=np.float32)
        elif self.obs_ptcloud:
            state_sz = self.obs_resolution*3  # 3D points in point cloud
            self.observation_space = gym.spaces.Box(
                -1.0*RearrangeEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz),
                RearrangeEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz))
        else:  # RGB images
            self.observation_space = gym.spaces.Box(  # channels: 3 color
                0.0, 1.0, shape=[3, self.obs_resolution, self.obs_resolution],
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
            0.0, 1.0, shape=[self.ndof], dtype=np.float32)
        super(RearrangeEnv, self).__init__()
        # TODO: make same interface to enable RL on low-dim state as Aux envs.
        # Note: in this env aux state in this envs is normalized to [-1,1]
        # (not on [0,1] to avoid rescaling sin,cos,rot dims).
        if debug_level>0:
            print('Created RearrangeEnv with observation_space',
                  self.observation_space, 'action_space', self.action_space,
                  'action limits', self.action_space.low, self.action_space.high,
                  'aux', aux.shape, 'aux_nms', self.aux_nms)

    @staticmethod
    def load_objects(sim, object_files, object_poses, object_quats,
                     object_masses, object_scales, geom_colors):
        # Subclasses can call this method to load custom objects.
        data_folder = os.path.split(__file__)[0]
        data_path = os.path.join(data_folder, 'data')
        num_obj = len(object_files)
        object_ids = []; obj_props = []
        # Note: setting useFixedBase=True for loadURDF() breaks collision
        # detection, so we use another way to ensure obstacle is stationary.
        for i in range(num_obj):
            object_file = os.path.join(data_path, object_files[i])
            if object_file.endswith('.urdf'):
                obj_id = sim.loadURDF(object_file, object_poses[i].tolist())
                sim.changeVisualShape(obj_id, -1, rgbaColor=(*geom_colors[i],1))
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
            # ATTENTON: pybullet not reporting dims correcly (nor geom here)
            _, _, _, dims, _, _, _, = sim.getCollisionShapeData(obj_id, -1)[0]
            if geom in (pybullet.GEOM_CYLINDER, pybullet.GEOM_MESH):
                dims = RearrangeEnv.CYLINDER_DEFAULT_DIMS
            elif geom==pybullet.GEOM_BOX:  # loadURDF() doesn't actually use
                dims = np.array(dims)/2    # halfExtents, so divide here
            # ATTENTON: pybullet not reporting inert_diag correctly
            # flags=pybullet.DYNAMICS_INFO_REPORT_INERTIA not present any more.
            mass, lat_fric, inert_diag, inert_pos, ori, restit, \
                rol_fric, spin_fric, contact_damp, contact_stiff, *_ \
                = sim.getDynamicsInfo(obj_id, -1)
            assert(mass>0)
            """
            # A few simple computations from basic formulas in e.g.:
            # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
            eps = 0.0001  # 1mm
            inert_diag = np.array(inert_diag)
            if (inert_diag - inert_diag.mean()).mean()<eps:  # approx sphere
                dims = np.sqrt(np.array(inert_diag)*2.5/mass)
                print('sphere')
            elif np.abs(inert_diag[0]-inert_diag[1])<eps:  # approx cylinder
                r = np.sqrt(inert_diag[2]*2/mass)
                tmp = np.array(inert_diag[0])*12/mass
                h = np.sqrt(tmp-3*r*r)
                dims = np.array([r,r,h])
                print('cylinder')
            else:  # try a cubiod
                tmp = np.array(inert_diag)*12/mass
                hwd = 0.5*tmp.sum()
                dims = hwd - tmp
                print('cuboid')
            """
            props = [*rgba[0:3], *dims, shape, mass, restit,
                     lat_fric, rol_fric, spin_fric]
            print('Loaded', object_file, 'at', object_poses[i], 'props', props)
            object_ids.append(obj_id)
            obj_props.append(np.array(props))
        return object_ids, obj_props

    def seed(self, seed):
        np.random.seed(seed)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def reset(self):
        self.stepnum = 0
        self.done = False  # used to avoid losing last frame in vec envs
        self.episode_rwd = 0.0
        if self.remove_robot:
            self.robot.sim.resetBasePositionAndOrientation(
                self.robot.robot_id, [1,0,0], [0,0,0,1])
        if self.rnd_init_pos:
            self.reset_to_random_pose()
        else:
            self.robot.reset()
            #if self.random_object_init_slot:
            #    num_obj = len(self.init_object_poses)
            #    xys = np.array(self.init_object_poses)
            #    xys = xys[np.random.permutation(xys.shape[0])]
            #    self.init_object_poses = xys
            self.robot.reset_objects(
                self.object_ids, self.init_object_poses, self.init_object_quats)
        # Make initial random actions, then return the starting state.
        for t in range(self.num_init_rnd_act):
            torque = (np.random.rand(self.max_torque.shape[0])-0.5)*self.max_torque
            self.robot.apply_joint_torque(torque)
        pixel_obs, _ = self.get_obs_and_aux()
        return pixel_obs

    def step(self, action):
        if np.isnan(action).all() or self.done:  # just return current obs, aux
            obs, aux = self.get_obs_and_aux()
            info = {'aux': aux, 'aux_nms': self.aux_nms,
                    'episode': {'r': self.episode_rwd, 'l': self.stepnum}}
            return obs, 0.0, self.done, info
        # Assume: robot is torque controlled and action is scaled to [0,1]
        torque = np.hstack(
            [action, np.zeros(self.max_torque.shape[0]-self.ndof)])
        torque = np.clip((torque-0.5)*2*self.max_torque,
                         -self.max_torque, self.max_torque)
        if self.debug_level>0 and self.stepnum%50==0:
            print('step', self.stepnum, 'action', action, 'torque', torque)
        # Apply torque action to joints
        for sub_step in range(self.num_action_repeats):  # repeat for faster sim
            self.robot.apply_joint_torque(torque)  # advances sim inside
            if not self.in_workspace(): break
        next_obs, next_aux = self.get_obs_and_aux()
        ee_ok = self.in_workspace()
        # Update internal counters.
        self.stepnum += 1
        # Report reward starts and other info.
        rwd = self.compute_reward()
        self.episode_rwd += rwd
        done = (self.stepnum == self._max_episode_steps) or not ee_ok
        info = {'aux': next_aux}
        if done:
            # Unused steps get a reward same as the last ok step
            self.episode_rwd += rwd * (self._max_episode_steps - self.stepnum)
            info['episode'] = {'r': float(self.episode_rwd), 'l': self.stepnum}
            if self.debug_level>0: print('tot_rwd {:.4f}'.format(self.episode_rwd))
            self.done = True; done = False  # will repeat last frame
        return next_obs, rwd, done, info

    def render(self, mode="rgb_array", close=False):
        pass  # implemented in pybullet

    def compute_reward(self):
        #if self.random_episode: return np.random.rand()
        dists = []
        for i, obj_id in enumerate(self.object_ids):
            nm = self.object_names[i]
            obj_pos, _ = self.robot.sim.getBasePositionAndOrientation(obj_id)
            tgt_id = 0 if nm.endswith(('_box', '_can')) else 1
            tgt_pos = RearrangeEnv.TARGETS_POS[tgt_id]
            dist = np.linalg.norm(np.array(obj_pos[:2]) - np.array(tgt_pos[:2]))
            dists.append(min(dist,1.0))   # max distance is 1 meter
        mean_dist = np.array(dists).mean()
        rwd = (1.0-mean_dist)/float(self._max_episode_steps)
        return rwd

    def render_obs(self, debug_out_dir=None):
        debug = debug_out_dir is not None
        if self.obs_resolution is None: return None
        if self.obs_ptcloud:
            ptcloud = ProcessCamera.get_ptcloud_obs(
                self.robot.sim, self._cam_object_ids, self.obs_resolution,
                width=100, cam_vals_list=self.cam_vals,
                box_lim=RearrangeEnv.PTCLOUD_BOX_SIZE,
                view_elev=30, view_azim=-70,
                debug_out_dir=debug_out_dir)
            return ptcloud
        rgba_px = self.robot.render_debug(width=self.obs_resolution)
        if debug:
            imageio.imwrite('/tmp/obs_st'+str(self.stepnum)+'.png', rgba_px)
        float_obs = rgba_px[:,:,0:3].astype(float)/255.
        float_obs = float_obs.transpose((2,0,1))
        return float_obs

    def get_obs_and_aux(self, aux_nms_to_fill=None):
        bkgrnd = None; obj_props = None; obj_poses = None; obj_quats = None
        # Include background and objects' state
        bkgrnd = self.bkgrnd[:]
        obj_props = np.copy(self.object_props)
        obj_poses = []; obj_quats = []
        for obj in self.object_ids:
            obj_pos, obj_quat = self.robot.sim.getBasePositionAndOrientation(obj)
            obj_poses.append(obj_pos)
            obj_quats.append(obj_quat)
        obj_poses = np.array(obj_poses)
        # Don't care about reporting object positions outside of workspace
        obj_poses = np.clip(obj_poses, RearrangeEnv.OBJ_XYZ_MINS,
                            RearrangeEnv.OBJ_XYZ_MAXS)
        obj_quats = np.array(obj_quats)
        flat_state = self.sim_to_flat_state(
            bkgrnd, obj_props, obj_poses, obj_quats, aux_nms_to_fill)
        pixel_obs = flat_state if self.obs_resolution is None else self.render_obs()
        return pixel_obs, flat_state

    def reset_to_random_pose(self):
        if hasattr(self, 'rnd_qpos_fxn'):
            rnd_qpos = None
            while rnd_qpos is None:
                rnd_qpos, end_ee_pos, rnd_ee_quat = self.rnd_qpos_fxn()
                if self.debug_level>0:
                    print('qpos_for_random_ee_pose', rnd_qpos, 'end_ee_pos',
                          end_ee_pos, 'rnd_ee_quat', rnd_ee_quat)
        else:  # random from [-pi,pi] for each joint (minus eps for stability)
            rnd_qpos = (np.random.rand(self.ndof)-0.5)*2*(0.75*np.pi)
        self.robot.reset_to_qpos(rnd_qpos)
        # TODO: update this when using random object shapes
        min_z = 0.03 if 'Ycb' in self.variant else 0.04
        xyrng = 0.15
        mins = np.array([-xyrng, -xyrng, min_z])
        maxes = np.array([xyrng, xyrng, 0.075])
        for objid in range(len(self.object_ids)):
            obj_pos = np.random.rand(3)
            obj_pos = denormalize(obj_pos, mins, maxes)
            obj_quat = np.random.rand(4)
            obj_quat = obj_quat/np.linalg.norm(obj_quat)
            self.robot.sim.resetBasePositionAndOrientation(
                self.object_ids[objid], obj_pos, obj_quat)
        for t in range(20):  # let the objects emerge from problematic poses
            self.robot.sim.stepSimulation()

    def sim_to_flat_state(self, bkgrnd, obj_props, obj_poses, obj_quats,
                          aux_nms_to_fill=None):
        obj_info = []
        # Add background and static object properties (if needed)
        if self.statics_in_lowdim:
            flat_state = bkgrnd
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
            obj_info.append(obj_props_normed)
        else:
            flat_state = float(self.version)/10
        # Add robot qpos.
        qpos = self.robot.get_qpos()
        #assert((qpos>=-np.pi).all() and (qpos<=np.pi).all())
        qpos_all_sin = np.sin(qpos).reshape(-1,1)
        qpos_all_cos = np.cos(qpos).reshape(-1,1)
        qpos_all_sin_cos = np.hstack([qpos_all_sin, qpos_all_cos])
        flat_state = np.hstack([flat_state, qpos_all_sin_cos.reshape(-1)])
        if aux_nms_to_fill is not None:
            if self.statics_in_lowdim:
                for b in range(bkgrnd.shape[0]):
                    aux_nms_to_fill.append('bkgrnd'+str(b))
            else:
                aux_nms_to_fill.append('version_id')
            for j in range(len(qpos_all_sin_cos)):
                aux_nms_to_fill.append('j'+str(j)+'_sin')
                aux_nms_to_fill.append('j'+str(j)+'_cos')
        if obj_props is None: return flat_state
        # Add object dynamics (pos, ori).
        if ((obj_poses<RearrangeEnv.OBJ_XYZ_MINS).any() or
            (obj_poses>RearrangeEnv.OBJ_XYZ_MAXS).any()):
            print('obj_poses outside min/max bounds', obj_poses)
            assert(False)  # obj_poses outside min/max bounds
        obj_poses_normed = normalize(
            obj_poses, RearrangeEnv.OBJ_XYZ_MINS, RearrangeEnv.OBJ_XYZ_MAXS)
        all_obj_rot = convert_all(obj_quats, 'quat_to_mat')
        obj_info.append(obj_poses_normed)
        obj_info.append(all_obj_rot)
        flat_state = np.hstack([flat_state, np.hstack(obj_info).reshape(-1)])
        if aux_nms_to_fill is not None:
            num_obj = len(obj_props); props_sz = obj_props.shape[1]
            #print('num_obj', num_obj, 'props_sz', props_sz,
            #      'bkgrnd', bkgrnd.shape, 'obj_props', obj_props.shape,
            #      'obj_poses', obj_poses.shape,
            #      'all_obj_rot', all_obj_rot.shape,
            #      'obj_info', obj_info.shape)
            for i in range(num_obj):
                shape_id = int(self.object_props[i][
                    RearrangeEnv.SIM_PARAM_OBJ_PROPS_NAMES.index('SHAPE')])
                obj_shape_name = RearrangeEnv.NAME_FROM_SHAPE[shape_id]
                if obj_shape_name=='': obj_shape_name = 'obj'
                obj_pfx = obj_shape_name+str(i)
                if self.statics_in_lowdim:
                    for j in range(props_sz):
                        nm = RearrangeEnv.SIM_PARAM_OBJ_PROPS_NAMES[j]
                        aux_nms_to_fill.append('obj'+str(i)+'_prop'+str(j))
                aux_nms_to_fill.append(obj_pfx+'_x')
                aux_nms_to_fill.append(obj_pfx+'_y')
                aux_nms_to_fill.append(obj_pfx+'_z')
                for j in range(9): aux_nms_to_fill.append(obj_pfx+'_rot'+str(j))
            if flat_state.shape[0]!=len(aux_nms_to_fill):
                print('flat_state.shape[0]', flat_state.shape[0],
                      ' vs len(aux_nms_to_fill)', len(aux_nms_to_fill))
            assert(flat_state.shape[0]==len(aux_nms_to_fill))
        return flat_state

    def flat_to_sim_state(self, flat_state):
        assert(flat_state.min()>=0 and flat_state.max()<=1)
        # Parse flat_state as static and dynamic simulation state.
        qpos_sz = self.ndof*2  # dof*2 since angle as sin,cos
        # Parse out qpos. This always comes 1st, since it is always known.
        qpos_sin_cos = flat_state[0:qpos_sz]
        qpos = sin_cos_to_theta(qpos_sin_cos)
        assert(qpos.min() >= -np.pi and qpos.max() <= np.pi)
        if flat_state.shape[0]==qpos_sz: return qpos, None, None, None, None
        # Parse out background, objects' static and dynamic state.
        bkgrnd_sz = len(RearrangeEnv.SIM_PARAM_INFO)
        num_obj = len(self.object_ids)
        obj_prop_sz = len(RearrangeEnv.SIM_PARAM_OBJ_PROPS_NAMES) # static state
        obj_dyn_sz = 3+9  # obj pos (3D), ori as 3x3 rotation matrix
        obj_state_sz = num_obj*(obj_prop_sz + obj_dyn_sz)
        assert(flat_state.shape[0] == (qpos_sz+bkgrnd_sz+obj_state_sz))
        ofst = qpos_sz
        bkgrnd = flat_state[ofst:ofst+bkgrnd_sz]
        ofst += bkgrnd_sz
        obj_state = flat_state[ofst:].reshape(num_obj, obj_prop_sz+obj_dyn_sz)
        obj_props = denormalize(
            obj_state[:,0:obj_prop_sz],
            RearrangeEnv.SIM_PARAM_OBJ_MINS, RearrangeEnv.SIM_PARAM_OBJ_MAXS)
        ofst = obj_prop_sz
        obj_poses = denormalize(
            obj_state[:,ofst:ofst+3],
            RearrangeEnv.OBJ_XYZ_MINS, RearrangeEnv.OBJ_XYZ_MAXS)
        ofst += 3
        all_obj_rot = obj_state[:,ofst:ofst+9]
        obj_quats = convert_all(all_obj_rot, 'mat_to_quat')
        return qpos, bkgrnd, obj_props, obj_poses, obj_quats

    def set_obj_state(self, obj_props, obj_poses, obj_quats):
        # Set background properties.
        # TODO: figure out how to do background color... print(bkgrnd)
        # Change color, shape, mass, friction, restitution of the objects.
        num_obj = len(self.object_ids)
        assert(obj_props.shape[0]==obj_poses.shape[0]==obj_quats.shape[0]==num_obj)
        assert(obj_props.shape[1]==len(RearrangeEnv.SIM_PARAM_OBJ_PROPS_NAMES))
        assert(obj_poses.shape[1]==3)
        assert(obj_quats.shape[1]==4)
        self.object_props[:] = obj_props[:]
        for objid in range(len(self.object_ids)):
            rgb_r, rgb_g, rgb_b, radius_a, radius_b, halfExtent, shape, mass, \
                restitution, lat_fric, rol_fric, spin_fric = obj_props[objid]
            viz_geom_type = RearrangeEnv.GEOM_FROM_SHAPE[int(round(shape))]
            if viz_geom_type==pybullet.GEOM_MESH:
                viz_geom_type = pybullet.GEOM_CYLINDER
            set_z = (obj_poses is None)
            if obj_poses is None: obj_poses = np.array(self.init_object_poses)
            if set_z: obj_poses[objid][2] = halfExtent
            if obj_quats is None: obj_quats = np.array(self.init_object_quats)
            # Print information we parsed from sim_params
            if self.debug_level>0:
                print('set_obj_state(): obj_props', obj_props)
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
            self.robot.sim.removeBody(self.object_ids[objid])
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
            self.object_ids[objid] = new_obj_id
            # TODO: Does bullet change inertial matrix based on mass, shape
            # TODO: and type, or do we need to re-specify manually?
            self.robot.sim.changeDynamics(
                self.object_ids[objid], -1, mass=mass, restitution=restitution,
                lateralFriction=lat_fric, rollingFriction=rol_fric,
                spinningFriction=spin_fric)

    def override_state(self, flat_state):
        qpos, bkgrnd, obj_props, obj_poses, obj_quats = \
            self.flat_to_sim_state(flat_state)
        self.robot.reset_to_qpos(qpos)
        self.bkgrnd[:] = bkgrnd[:]
        self.robot.sim.changeVisualShape(
                self.border_id, 0, rgbaColor=(*bkgrnd,1),
                textureUniqueId=self.blank_texture_id)
        if obj_props is not None:
            self.set_obj_state(obj_props, obj_poses, obj_quats)

    def in_workspace(self):
        # First check whether all the objects are still in the workspace.
        for tmpi, obj in enumerate(self.object_ids):
            obj_pos, obj_quat = self.robot.sim.getBasePositionAndOrientation(obj)
            if ((np.array(obj_pos)<RearrangeEnv.OBJ_XYZ_MINS).any() or
                (np.array(obj_pos)>RearrangeEnv.OBJ_XYZ_MAXS).any()):
                if self.debug_level>0:
                    print('Object', self.object_names[tmpi], 'outside range')
                return False
        # Now do robot-related checks.
        if 'Reacher' in self.__class__.__name__: return True
        if hasattr(self, 'remove_robot'): return True  # not including robot
        ee_pos = self.robot.get_ee_pos()
        # Check whether ee is too high above the objects.
        if hasattr(self, 'max_object_z') and ee_pos[2] > 0.2+self.max_object_z:
            if self.debug_level>0:
                print('step', self.stepnum, 'ee_pos too high', ee_pos)
            return False
        # Check that ee is above the table crate.
        if (ee_pos[0]<-0.5 or ee_pos[0]>0.5 or
            ee_pos[1]<-0.5 or ee_pos[1]>0.5):
            if self.debug_level>0:
                print('step', self.stepnum, 'ee_pos not in workspace', ee_pos)
            return False
        return True
