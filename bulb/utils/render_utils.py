"""
Rendering utilities for PyBullet environments.
"""
import imageio
import os

import pybullet

from ..utils.cam_vals_cache import CAM_VALS_CACHE
from ..utils.ptcloud_utils import *


def rgba_to_rgb_torch(rgba_px):
    rgb = rgba_px[:,:,0:3].astype(float)/255.  # uint8 RGBA -> float32 RGB
    rgb = rgb.transpose((2,0,1))               # H x W x C(RGB)
    return rgb


def rgba_from_pybullet(sim, cam_dist, cam_target, cam_pitch, cam_yaw,
                       view_matrix=None, proj_matrix=None,
                       height=600, width=600):
    if view_matrix is None:
        view_matrix = sim.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target, distance=cam_dist,
            yaw=cam_yaw, pitch=cam_pitch, roll=0, upAxisIndex=2)
    if proj_matrix is None:
        proj_matrix = sim.computeProjectionMatrixFOV(
            fov=90, aspect=float(width)/height, nearVal=0.01, farVal=100.0)
    # Note: can use ER_TINY_RENDERER if ER_BULLET_HARDWARE_OPENGL is a problem.
    w, h, rgba_px, depth_px, segment_mask = sim.getCameraImage(
        width=width, height=height,
        viewMatrix=view_matrix, projectionMatrix=proj_matrix,
        renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    return rgba_px  # uint8 H x W x RGBA


def render_obs(sim, stepnum, resolution, override_resolution,
               cam_dist, cam_target, cam_pitch, cam_yaw,
               obs_ptcloud, cam_object_ids, cam_vals_cache_key, box_lim,
               debug_out_dir=None, view_matrix=None, proj_matrix=None):
    if override_resolution is not None: resolution = override_resolution
    if resolution is None: return None
    if obs_ptcloud:  # point cloud obs
        if cam_vals_cache_key not in CAM_VALS_CACHE:
            PanCamera.compute_cam_vals(
                cam_dist=cam_dist, cam_tgt=cam_target,
                cam_yaws=PanCamera.CAM_YAWS, cam_pitches=PanCamera.CAM_PITCHES)
            print('Add the above to CAM_VALS_CACHE['+cam_vals_cache_key+']')
            # Note: for PyBullet benchmark call the above after reset().
            assert(False)
        ptcloud, _ = PanCamera.render_rigid(sim, cam_object_ids,
            width=100, cam_vals_list=CAM_VALS_CACHE[cam_vals_cache_key])
        if debug_out_dir is not None:
            dbg_fnm = os.path.expanduser(os.path.join(
                debug_out_dir, f'tmp_ptcld_step{stepnum:d}.png'))
            plot_ptcloud(ptcloud, dbg_fnm, box_lim, view_elev=30, view_azim=-70)
        ptcloud = trim_ptcloud(ptcloud, box_lim, resolution)
        obs = ptcloud.reshape(-1)
    else:  # RGB pixel obs
        rgba_px = rgba_from_pybullet(
            sim, cam_dist, cam_target, cam_pitch, cam_yaw,
            view_matrix, proj_matrix, height=resolution, width=resolution)
        if debug_out_dir is not None:
            dbg_fnm = os.path.expanduser(os.path.join(
                debug_out_dir, f'tmp_step{stepnum:d}.png'))
            imageio.imwrite(dbg_fnm, rgba_px)
        obs = rgba_to_rgb_torch(rgba_px)
    return obs
