"""
Block on incline env.
"""
import os

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)
import gym

import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

from ..utils import render_utils
from .aux_env import AuxEnv
from .rearrange_utils import YCB_OBJECT_INFOS


class BlockOnInclineEnv(gym.Env, AuxEnv):
    NVRNTS = 6  # 6 various objects from Ycb dataset
    OBJECT_LISTS = {
        'Ycb': ['004_sugar_box', '008_pudding_box', '009_gelatin_box',
                '005_tomato_soup_can', '007_tuna_fish_can',
                '002_master_chef_can'],
        'Geom' : ['block.urdf', 'block.urdf', 'block.urdf',
                  'cylinder.urdf', 'cylinder_large.urdf', 'cylinder_short.urdf']
    }
    PLANE_TEXTURES = ['red_marble', 'blue_bright', 'purple_squares',
                      'wood_stripes', 'dark_pattern', 'orange_pattern']
    INCLINE_TEXTURES = ['red_metal_strip', 'marble', 'birch', 'green',
                        'retro_purple_long', 'wood']
    INCLINE_COLORS = [[1,0.5,0],[0.8,0.8,0.8],[0.9,0.4,0.4],
                      [0,1,0],[0.5,0.15,0.15],[1.0,0.6,0]]
    GEOM_COLORS = [[1,1,0.5],[0.2,0.2,0.2],[1,0,0],[1,0,1],[0,0.5,1],[0,0,1]]
    OBS_INIT_POS = np.array([-0.05, 0, 0.7])
    TGT_POS_X = 0.5; MIN_POS_X = 0.0; MAX_POS_Y = 0.1
    MAX_POS_X = 1.0  # terminate episode after object reaches with x coord
    CLIP_POS_X = 1.27
    PTCLOUD_BOX_SIZE = 1.3  # 1.3m cube cut off for point cloud observations
    MIN_VEL = 0.0; CLIP_VEL = 3.10
    MIN_MASS = 0.05; MAX_MASS = 0.50
    MIN_FRIC = 0.10; MAX_FRIC = 0.50
    MIN_THETA = 0.25; MAX_THETA = 0.50
    # MIN/MAX texture_id, obj_id, (fric), incline_theta, pos_t, v_t
    LD_STATE_MINS = np.array([     0,      0, MIN_THETA,  MIN_POS_X,  MIN_VEL])
    LD_STATE_MAXS = np.array([NVRNTS, NVRNTS, MAX_THETA, CLIP_POS_X, CLIP_VEL])
    LD_STATE_WFRIC_MINS = np.array(
        [     0,      0, MIN_FRIC, MIN_THETA,  MIN_POS_X, MIN_VEL])
    LD_STATE_WFRIC_MAXS = np.array(
        [NVRNTS, NVRNTS, MAX_FRIC, MAX_THETA, CLIP_POS_X, CLIP_VEL])

    def __init__(self, version, variant, obs_resolution, obs_ptcloud, scale,
                 report_fric=False, randomize=True,
                 debug=False, visualize=False):
        self._version = version
        self._variant = variant
        self._obs_resolution = obs_resolution  # e.g. 64 for 64x64 images
        self._obs_ptcloud = obs_ptcloud  # whether obs should be point clouds
        self._report_fric = report_fric
        self._randomize = randomize
        self._debug = debug
        self._visualize = visualize
        self._use_tn = True  # encode block position as length travelled
        self._max_episode_steps = 50
        self._nskip = 4  # sim nskip steps at a time
        self._sim = bc.BulletClient(
            connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        self._sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._texture_id = 0; self._obj_mass = 0.05; self._fric = 0.5  # default
        self._stepnum = 0
        self._episode_rwd = 0.0
        # Load plane, block and incline.
        # Note that sim.loadTexture() does not work for default plane, but
        # seems to work for custom planes in the data folder.
        data_path = os.path.join(os.path.split(__file__)[0], 'data')
        num_textures = len(BlockOnInclineEnv.PLANE_TEXTURES)
        assert(num_textures==len(BlockOnInclineEnv.INCLINE_TEXTURES))
        self.plane_texture_ids = []; self.incline_texture_ids = []
        for txtri in range(num_textures):
            nm = BlockOnInclineEnv.PLANE_TEXTURES[txtri]
            self.plane_texture_ids.append(self._sim.loadTexture(
                os.path.join(data_path, 'planes', nm+'.png')))
            nm = BlockOnInclineEnv.INCLINE_TEXTURES[txtri]
            self.incline_texture_ids.append(self._sim.loadTexture(
                os.path.join(data_path, 'inclines', nm+'.png')))
        self.plane_id = self._sim.loadURDF(
            os.path.join(data_path, 'planes', 'plane.urdf'))
        self.incline_id, self.incline_pitch = BlockOnInclineEnv.load_incline(
            self._sim, self._version, self._variant)
        self.block_id = BlockOnInclineEnv.load_object(
            self._sim, version, variant, scale)
        # Init bullet sim params.
        self._sim.setGravity(0, 0, -9.81)
        # default: https://github.com/bulletphysics/bullet3/issues/1460
        self._sim.setTimeStep(0.01)  # 100Hz
        self._sim.setRealTimeSimulation(0)
        self._sim.setPhysicsEngineParameter(numSolverIterations=5, numSubSteps=2)
        self._cam_dist = 0.8; self._cam_target = [0.5, 0.2, 0.1]
        self._cam_yaw = 35; self._cam_pitch = -45
        self._cam_object_ids = [self.block_id]  # needed for point clouds
        if visualize:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,0)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,0)
            self._sim.resetDebugVisualizerCamera(
                cameraDistance=self._cam_dist,
                cameraYaw=self._cam_yaw, cameraPitch=self._cam_pitch,
                cameraTargetPosition=self._cam_target)
            self._dbg_txt_id = self._sim.addUserDebugText(
                text='_', textPosition=[0.7,0,0.62], textSize=5,
                textColorRGB=[1,0,0])
            self._dbg_txt1_id = self._sim.addUserDebugText(
                text='_', textPosition=[0.7,0,0.57],
                textSize=5, textColorRGB=[0,1,0])
        self.aux_nms = BlockOnInclineEnv.dim_names(self._use_tn, self._report_fric)
        if self._obs_resolution is None:
            self.observation_space = gym.spaces.Box(
                0.0, 1.0, shape=[len(self.aux_nms)], dtype=np.float32)
        elif self._obs_ptcloud:
            state_sz = self._obs_resolution * 3  # 3D points in point cloud
            self.observation_space = gym.spaces.Box(
                -1.0*BlockOnInclineEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz),
                BlockOnInclineEnv.PTCLOUD_BOX_SIZE*np.ones(state_sz))
        else:  # RGB images
            self.observation_space = gym.spaces.Box(  # channels: 3 color
                0.0, 1.0, shape=[3, self._obs_resolution, self._obs_resolution],
                dtype=np.float32)
        self.action_space = gym.spaces.Box(
            -2.0, 2.0, shape=[1], dtype=np.float32)
        if debug>0: print('Created BlockOnInclineEnv')

    @staticmethod
    def dim_names(use_tn, report_fric):  # low-dim state names
        if report_fric:
            static = ['texture_id', 'object_id', 'fric', 'theta']
        else:
            static = ['texture_id', 'object_id', 'theta']
        dynamic = ['x', 'x_vel']
        if not use_tn:
            dynamic = ['x', 'y', 'z', 'x_vel', 'y_vel', 'z_vel']
        return static+dynamic

    @staticmethod
    def load_incline(sim, version, variant):
        pitch = 0.25+float(version)/20.0
        data_path = os.path.join(os.path.split(__file__)[0], 'data')
        fpath = os.path.join(data_path, 'inclines', 'incline.urdf')
        incline_id = sim.loadURDF(fpath, [0,0,0.55], useFixedBase=True)
        return incline_id, pitch

    @staticmethod
    def load_object(sim, version, variant, scale=2.5):
        data_path = os.path.join(os.path.split(__file__)[0], 'data')
        fname = BlockOnInclineEnv.OBJECT_LISTS[variant][version]
        obj_info = YCB_OBJECT_INFOS[fname]
        if variant=='Ycb':
            assert(fname.startswith('0'))  # YCB mesh
            fname = os.path.join('ycb', fname, 'google_16k', 'textured_ok.obj')
            fpath = os.path.join(data_path, fname)
            mesh_scale = (np.array(obj_info['s'])*scale).tolist()
            viz_shape_id = sim.createVisualShape(
                shapeType=pybullet.GEOM_MESH, rgbaColor=None,
                fileName=fpath, meshScale=mesh_scale)
            col_shape_id = sim.createCollisionShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=fpath, meshScale=mesh_scale)
            block_id = sim.createMultiBody(
                baseMass=obj_info['m'],
                basePosition=[0,0,0], baseOrientation=[0,0,0,1],
                baseCollisionShapeIndex=col_shape_id,
                baseVisualShapeIndex=viz_shape_id)
        else:
            assert(variant=='Geom')
            block_id = sim.loadURDF(
                os.path.join(data_path, fname), globalScaling=scale)
            clr = BlockOnInclineEnv.GEOM_COLORS[version]
            sim.changeVisualShape(block_id, -1, rgbaColor=(*clr,1))
        return block_id

    @property
    def low_dim_state_space(self):
        return

    @property
    def low_dim_state_names(self):
        return

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def disconnect(self):
        self._sim.disconnect()

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self._stepnum = 0
        self._episode_rwd = 0.0
        if self._randomize:
            mins = np.array([BlockOnInclineEnv.MIN_MASS,
                             BlockOnInclineEnv.MIN_FRIC,
                             BlockOnInclineEnv.MIN_THETA])
            maxs = np.array([BlockOnInclineEnv.MAX_MASS,
                             BlockOnInclineEnv.MAX_FRIC,
                             BlockOnInclineEnv.MAX_THETA])
            rnd = np.random.rand(3)*(maxs-mins) + mins
            self._obj_mass, self._fric, self.incline_pitch = rnd[:]
            self._texture_id = np.random.randint(0, len(self.plane_texture_ids))
        else:
            self._texture_id = self._version
        rnd_eps = (np.random.rand(2)-0.5)*np.pi/8
        rnd_rpy = [np.pi/2,np.pi/2+rnd_eps[0],np.pi/2+rnd_eps[0]]
        # Reset sim elements using rnd_*
        self._sim.changeDynamics(self.block_id, -1, mass=self._obj_mass,
                                 lateralFriction=self._fric)
        self._sim.changeDynamics(self.incline_id, -1, lateralFriction=self._fric)
        incline_quat = pybullet.getQuaternionFromEuler(
            [0,self.incline_pitch,0])
        self._sim.resetBasePositionAndOrientation(
            self.incline_id, [0,0,0.55], incline_quat)
        if self._variant=='Ycb':
            self._sim.changeVisualShape(
                self.plane_id, -1, rgbaColor=(1,1,1),
                textureUniqueId=self.plane_texture_ids[self._texture_id])
            self._sim.changeVisualShape(
                self.incline_id, -1, rgbaColor=(1,1,1),
                textureUniqueId=self.incline_texture_ids[self._texture_id])
        else:
            assert(self._variant=='Geom')
            clr = BlockOnInclineEnv.INCLINE_COLORS[self._texture_id]
            self._sim.changeVisualShape(self.incline_id, -1, rgbaColor=(*clr,1))
        self._sim.resetBasePositionAndOrientation(
            self.block_id, BlockOnInclineEnv.OBS_INIT_POS.tolist(),
            pybullet.getQuaternionFromEuler(rnd_rpy))
        init_step = 0; rnd_start = 0.05+np.random.rand(1)[0]*0.1
        while init_step<self._max_episode_steps:
            self._sim.stepSimulation()  # initial fall
            block_pos, _ = self._sim.getBasePositionAndOrientation(self.block_id)
            if block_pos[0]>rnd_start: break
            init_step += 1
        if self._debug: print('init_step', init_step, 'block_pos', block_pos)
        return self.compute_obs()

    def step(self, action):
        # External force to be applied at each sub-step.
        act = np.clip(action[0],
                      self.action_space.low[0], self.action_space.high[0])
        for i in range(self._nskip):
            self._sim.applyExternalForce(
                self.block_id, -1, [0,0,act], [0,0,0], pybullet.LINK_FRAME)
            self._sim.stepSimulation()
        next_obs = self.compute_obs()
        # Update internal counters.
        self._stepnum += 1
        # Report reward starts and other info.
        block_pos, _ = self._sim.getBasePositionAndOrientation(self.block_id)
        rwd = self.compute_reward(block_pos)
        self._episode_rwd += rwd
        info = {}
        done = (self._stepnum >= self._max_episode_steps or
                block_pos[0] >= BlockOnInclineEnv.MAX_POS_X or
                block_pos[0] <= BlockOnInclineEnv.MIN_POS_X or
                np.abs(block_pos[1]) >= BlockOnInclineEnv.MAX_POS_Y)
        if done:
            info['episode'] = {'r': float(self._episode_rwd), 'l': self._stepnum}
        if self._visualize:
            dbg_txt = '{:s} {:0.2f}'.format('<' if act<0 else '>', act)
            self._dbg_txt_id = self._sim.addUserDebugText(
                text=dbg_txt, textPosition=[0.7,0,0.62], textSize=5,
                textColorRGB=[1,0,0], replaceItemUniqueId=self._dbg_txt_id)
            self._dbg_txt1_id = self._sim.addUserDebugText(
                text='{:0.2f} tot {:0.2f}'.format(rwd, self._episode_rwd),
                textPosition=[0.7,0,0.57],
                textSize=5, textColorRGB=[0,1,0],
                replaceItemUniqueId=self._dbg_txt1_id)
        return next_obs, rwd, done, info

    def compute_obs(self):
        if self._obs_resolution is None:
            obs = self.compute_low_dim_state()
        else:
            obs = self.render_obs()
        return obs

    def compute_reward(self, block_pos):
        dist = np.abs(block_pos[0]-BlockOnInclineEnv.TGT_POS_X)
        return BlockOnInclineEnv.MAX_POS_X - dist

    def compute_low_dim_state(self):
        static_state = [float(self._texture_id), float(self._version)]
        if self._report_fric: static_state.append(self._fric)
        static_state.append(self.incline_pitch)
        static_state = np.array(static_state)
        block_pos, _ = self._sim.getBasePositionAndOrientation(self.block_id)
        block_pos = np.array(block_pos)
        block_vel = self._sim.getBaseVelocity(self.block_id)
        block_vel_lin = np.array(block_vel[0])  # block_vel_ang = block_vel[1]
        if self._use_tn:
            pos_t = np.linalg.norm(block_pos - BlockOnInclineEnv.OBS_INIT_POS)
            if block_pos[0]<BlockOnInclineEnv.MIN_POS_X: pos_t = 0
            v_t = np.linalg.norm(block_vel_lin)
            dyn_state = np.array([pos_t, v_t])
        else:
            dyn_state = np.hstack([block_pos, block_vel_lin])
        raw_ld_state = np.hstack([static_state, dyn_state])
        mins = BlockOnInclineEnv.LD_STATE_WFRIC_MINS if self._report_fric \
            else BlockOnInclineEnv.LD_STATE_MINS
        maxs = BlockOnInclineEnv.LD_STATE_WFRIC_MAXS if self._report_fric \
            else BlockOnInclineEnv.LD_STATE_MAXS
        if (raw_ld_state<mins).any() or (raw_ld_state>maxs).any():
            if self._debug:
                print('clipping at step', self._stepnum, 'raw_ld_state')
                print(raw_ld_state)
            raw_ld_state = np.clip(raw_ld_state, mins, maxs)
        ld_state = (raw_ld_state - mins)/(maxs-mins)
        assert((ld_state>=0).all() and (ld_state<=1).all())
        assert(ld_state.shape[0]==len(self.aux_nms))
        return ld_state

    def override_state(self, ld_state):
        assert(False), 'TODO: implement override_state (remember to denorm)'

    def render_obs(self, override_resolution=None, debug_out_dir=None):
        obs = render_utils.render_obs(
            self._sim, self._stepnum,
            self._obs_resolution, override_resolution,
            self._cam_dist, self._cam_target,
            self._cam_pitch, self._cam_yaw,
            self._obs_ptcloud, self._cam_object_ids, 'BlockOnIncline',
            BlockOnInclineEnv.PTCLOUD_BOX_SIZE, debug_out_dir)
        return obs
