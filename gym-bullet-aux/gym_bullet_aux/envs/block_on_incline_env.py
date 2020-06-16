"""
Block on incline env.
"""

from copy import copy
import fileinput
import os
from shutil import copyfile

import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)
import gym

import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bc

from .rearrange_utils import YCB_OBJECT_INFOS


class BlockOnInclineEnv(gym.Env):
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
    TGT_POS_X = 0.5; MIN_POS_X = 0.0; MAX_POS_Y = 0.1
    MAX_POS_X = 1.0  # terminate episode after object reaches with x coord
    MIN_POS_X = 0.0; CLIP_POS_X = 1.27
    MIN_VEL = 0.0; CLIP_VEL = 3.10
    MIN_MASS = 0.05; MAX_MASS = 0.50
    MIN_FRIC = 0.10; MAX_FRIC = 0.50
    MIN_THETA = 0.25; MAX_THETA = 0.50
    # MIN/MAX texture_id, obj_id, (fric), incline_theta, pos_t, v_t
    FLAT_STATE_MINS = np.array([     0,      0, MIN_THETA,  MIN_POS_X,  MIN_VEL])
    FLAT_STATE_MAXS = np.array([NVRNTS, NVRNTS, MAX_THETA, CLIP_POS_X, CLIP_VEL])
    FLAT_STATE_WFRIC_MINS = np.array(
        [     0,      0, MIN_FRIC, MIN_THETA,  MIN_POS_X, MIN_VEL])
    FLAT_STATE_WFRIC_MAXS = np.array(
        [NVRNTS, NVRNTS, MAX_FRIC, MAX_THETA, CLIP_POS_X, CLIP_VEL])

    def __init__(self, version, variant, scale=2.5, obs_resolution=64,
                 report_fric=False, randomize=True,
                 visualize=False, debug_level=0):
        self.version = version
        self.variant = variant
        self.visualize = visualize
        self.debug_level = debug_level
        self.use_tn = True  # encode block position as length travelled
        self.obs_resolution = obs_resolution  # e.g. 64 for 64x64 images
        self.randomize = randomize
        self.report_fric = report_fric
        self.max_episode_len = 50
        self.nskip = 4  # sim nskip steps at a time
        self.obj_init_pos = np.array([-0.05,0,0.7])
        self.texture_id = 0; self.obj_mass = 0.05; self.fric = 0.5  # will reset
        self.sim = bc.BulletClient(
            connection_mode=pybullet.GUI if visualize else pybullet.DIRECT)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Load plane, block and incline.
        # Note that sim.loadTexture() does not work for default plane, but
        # seems to work for custom planes in the data folder.
        data_path = os.path.join(os.path.split(__file__)[0], 'data')
        num_textures = len(BlockOnInclineEnv.PLANE_TEXTURES)
        assert(num_textures==len(BlockOnInclineEnv.INCLINE_TEXTURES))
        self.plane_texture_ids = []; self.incline_texture_ids = []
        for txtri in range(num_textures):
            nm = BlockOnInclineEnv.PLANE_TEXTURES[txtri]
            self.plane_texture_ids.append(self.sim.loadTexture(
                os.path.join(data_path, 'planes', nm+'.png')))
            nm = BlockOnInclineEnv.INCLINE_TEXTURES[txtri]
            self.incline_texture_ids.append(self.sim.loadTexture(
                os.path.join(data_path, 'inclines', nm+'.png')))
        self.plane_id = self.sim.loadURDF(
            os.path.join(data_path, 'planes', 'plane.urdf'))
        self.incline_id, self.incline_pitch = BlockOnInclineEnv.load_incline(
            self.sim, self.version, self.variant)
        self.block_id = BlockOnInclineEnv.load_object(
            self.sim, version, variant, scale)
        # Init bullet sim params.
        self.sim.setGravity(0, 0, -9.81)
        # default: https://github.com/bulletphysics/bullet3/issues/1460
        self.dt = 0.01  # 100Hz
        self.sim.setTimeStep(self.dt)
        self.sim.setRealTimeSimulation(0)
        self.sim.setPhysicsEngineParameter(numSolverIterations=5, numSubSteps=2)
        self.cam_dist = 0.8
        self.cam_yaw = 35; self.cam_pitch = -45; self.cam_target = [0.5, 0.2, 0.1]
        #self.cam_yaw = 25; self.cam_pitch = -35
        #self.cam_dist = 0.5  # uncomment for a simple flat view
        #self.cam_yaw = 0; self.cam_pitch = -5; self.cam_target = [0.5, 0.2, 0.3]
        if visualize:
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
            self.dbg_txt_id = self.sim.addUserDebugText(
                text='_', textPosition=[0.7,0,0.62], textSize=5,
                textColorRGB=[1,0,0])
            self.dbg_txt1_id = self.sim.addUserDebugText(
                text='_', textPosition=[0.7,0,0.57],
                textSize=5, textColorRGB=[0,1,0])
        self.aux_nms = BlockOnInclineEnv.dim_names(self.use_tn, self.report_fric)
        if self.obs_resolution is not None:
            self.observation_space = gym.spaces.Box(  # channels: 3 color
                0.0, 1.0, shape=[3, self.obs_resolution, self.obs_resolution],
                dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Box(
                0.0, 1.0, shape=[len(self.aux_nms)], dtype=np.float32)
        self.action_space = gym.spaces.Box(
            -2.0, 2.0, shape=[1], dtype=np.float32)
        if debug_level>0: print('Created BlockOnInclineEnv')

    def disconnect(self):
        self.sim.disconnect()

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

    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        self.stepnum = 0
        self.tot_rwd = 0.0
        self.done = False
        if self.randomize:
            mins = np.array([BlockOnInclineEnv.MIN_MASS,
                             BlockOnInclineEnv.MIN_FRIC,
                             BlockOnInclineEnv.MIN_THETA])
            maxs = np.array([BlockOnInclineEnv.MAX_MASS,
                             BlockOnInclineEnv.MAX_FRIC,
                             BlockOnInclineEnv.MAX_THETA])
            rnd = np.random.rand(3)*(maxs-mins) + mins
            self.obj_mass, self.fric, self.incline_pitch = rnd[:]
            self.texture_id = np.random.randint(0,len(self.plane_texture_ids))
        else:
            self.texture_id = self.version
        rnd_eps = (np.random.rand(2)-0.5)*np.pi/8
        rnd_rpy = [np.pi/2,np.pi/2+rnd_eps[0],np.pi/2+rnd_eps[0]]
        # Reset sim elements using rnd_*
        self.sim.changeDynamics(self.block_id, -1, mass=self.obj_mass,
                                lateralFriction=self.fric)
        self.sim.changeDynamics(self.incline_id, -1, lateralFriction=self.fric)
        incline_quat = pybullet.getQuaternionFromEuler(
            [0,self.incline_pitch,0])
        self.sim.resetBasePositionAndOrientation(
            self.incline_id, [0,0,0.55], incline_quat)
        if self.variant=='Ycb':
            self.sim.changeVisualShape(
                self.plane_id, -1, rgbaColor=(1,1,1),
                textureUniqueId=self.plane_texture_ids[self.texture_id])
            self.sim.changeVisualShape(
                self.incline_id, -1, rgbaColor=(1,1,1),
                textureUniqueId=self.incline_texture_ids[self.texture_id])
        else:
            assert(self.variant=='Geom')
            clr = BlockOnInclineEnv.INCLINE_COLORS[self.texture_id]
            self.sim.changeVisualShape(self.incline_id, -1, rgbaColor=(*clr,1))
        self.sim.resetBasePositionAndOrientation(
            self.block_id, self.obj_init_pos.tolist(),
            pybullet.getQuaternionFromEuler(rnd_rpy))
        init_step = 0; rnd_start = 0.05+np.random.rand(1)[0]*0.1
        while init_step<self.max_episode_len:
            self.sim.stepSimulation()  # initial fall
            block_pos, _ = self.sim.getBasePositionAndOrientation(self.block_id)
            if block_pos[0]>rnd_start: break
            init_step += 1
        #rnd_vel = np.random.rand(1)[0]*0.5; th = self.incline_pitch
        #rnd_vel_vec = [rnd_vel*np.cos(th),0,rnd_vel*np.sin(th)]
        #self.sim.resetBaseVelocity(self.block_id, rnd_vel_vec)
        if self.debug_level>0:
            print('init_step', init_step, 'block_pos', block_pos)
        pixel_obs, _ = self.get_obs_and_aux()
        return pixel_obs

    def step(self, action):
        if np.isnan(action).all() or self.done:  # just return current obs, aux
            obs, aux = self.get_obs_and_aux()
            return obs, 0.0, self.done, {'aux': aux, 'aux_nms': self.aux_nms}
        # External force to be applied at each sub-step.
        act = np.clip(action[0], self.action_space.low[0], self.action_space.high[0])
        for i in range(self.nskip):
            self.sim.applyExternalForce(
                self.block_id, -1, [0,0,act], [0,0,0], pybullet.LINK_FRAME)
            self.sim.stepSimulation()
        next_obs, next_aux = self.get_obs_and_aux()
        info = {'aux': next_aux}
        self.stepnum += 1
        block_pos, _ = self.sim.getBasePositionAndOrientation(self.block_id)
        reward = self.compute_reward(block_pos)
        self.tot_rwd += reward
        done = (self.stepnum >= self.max_episode_len or
                block_pos[0]>=BlockOnInclineEnv.MAX_POS_X or
                block_pos[0]<=BlockOnInclineEnv.MIN_POS_X or
                np.abs(block_pos[1])>=BlockOnInclineEnv.MAX_POS_Y)
        if done:
            if self.debug_level>0:
                print('done at step', self.stepnum, 'block_pos', block_pos)
                self.sim_to_flat_state(debug=True)
                #input('Press Enter to continue')
            info['episode'] = {'r': float(self.tot_rwd)}
            self.done = True; done = False  # will repeat last frame
        if self.visualize:
            dbg_txt = '{:s} {:0.2f}'.format('<' if act<0 else '>', act)
            self.dbg_txt_id = self.sim.addUserDebugText(
                text=dbg_txt, textPosition=[0.7,0,0.62], textSize=5,
                textColorRGB=[1,0,0], replaceItemUniqueId=self.dbg_txt_id)
            self.dbg_txt1_id = self.sim.addUserDebugText(
                text='{:0.2f} tot {:0.2f}'.format(reward, self.tot_rwd),
                textPosition=[0.7,0,0.57],
                textSize=5, textColorRGB=[0,1,0],
                replaceItemUniqueId=self.dbg_txt1_id)
        return next_obs, reward, done, info

    def compute_reward(self, block_pos):
        dist = np.abs(block_pos[0]-BlockOnInclineEnv.TGT_POS_X)
        return BlockOnInclineEnv.MAX_POS_X - dist

    def get_obs_and_aux(self, aux_nms_to_fill=None):
        flat_state = self.sim_to_flat_state()
        if self.obs_resolution is not None:
            pixel_obs = self.render_obs(debug=False)
            return pixel_obs, flat_state
        else:
            return flat_state, flat_state

    def sim_to_flat_state(self, debug=False):
        static_state = [float(self.texture_id), float(self.version)]
        if self.report_fric: static_state.append(self.fric)
        static_state.append(self.incline_pitch)
        static_state = np.array(static_state)
        block_pos, _ = self.sim.getBasePositionAndOrientation(self.block_id)
        block_pos = np.array(block_pos)
        block_vel = self.sim.getBaseVelocity(self.block_id)
        block_vel_lin = np.array(block_vel[0])  # block_vel_ang = block_vel[1]
        if self.use_tn:
            pos_t = np.linalg.norm(block_pos - self.obj_init_pos)
            if block_pos[0]<BlockOnInclineEnv.MIN_POS_X: pos_t = 0
            v_t = np.linalg.norm(block_vel_lin)
            dyn_state = np.array([pos_t, v_t])
        else:
            dyn_state = np.hstack([block_pos, block_vel_lin])
        raw_flat_state = np.hstack([static_state, dyn_state])
        if debug:
            print('step', self.stepnum, 'raw_flat_state'); print(raw_flat_state)
        mins = BlockOnInclineEnv.FLAT_STATE_WFRIC_MINS if self.report_fric \
            else BlockOnInclineEnv.FLAT_STATE_MINS
        maxs = BlockOnInclineEnv.FLAT_STATE_WFRIC_MAXS if self.report_fric \
            else BlockOnInclineEnv.FLAT_STATE_MAXS
        if (raw_flat_state<mins).any() or (raw_flat_state>maxs).any():
            if self.debug_level>0:
                print('clipping at step', self.stepnum, 'raw_flat_state')
                print(raw_flat_state)
                input('cont')
            raw_flat_state = np.clip(raw_flat_state, mins, maxs)
        flat_state = (raw_flat_state - mins)/(maxs-mins)
        assert((flat_state>=0).all() and (flat_state<=1).all())
        assert(flat_state.shape[0]==len(self.aux_nms))
        return flat_state

    def render_obs(self, debug=False):
        rgba_px = self.render_debug(width=self.obs_resolution)
        if debug:
            import imageio
            imageio.imwrite('/tmp/obs_st'+str(self.stepnum)+'.png', rgba_px)
        float_obs = rgba_px[:,:,0:3].astype(float)/255.
        float_obs = float_obs.transpose((2,0,1))
        return float_obs

    def render_debug(self, mode="rgb_array", width=600):
        if mode != "rgb_array": return np.array([])
        height = width
        view_matrix = self.sim.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.cam_target, distance=self.cam_dist,
            yaw=self.cam_yaw, pitch=self.cam_pitch, roll=0, upAxisIndex=2)
        proj_matrix = self.sim.computeProjectionMatrixFOV(
            fov=90, aspect=float(width)/height, nearVal=0.01, farVal=100.0)
        w, h, rgba_px, depth_px, segment_mask = self.sim.getCameraImage(
            width=width, height=height, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_TINY_RENDERER)
        #import scipy.misc
        #scipy.misc.imsave('/tmp/outfile.png', rgba_px)  # needs scipy 1.0.1
        return rgba_px  # HxWxRGBA uint8
