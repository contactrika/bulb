"""
Utils for rearrange envs.
"""

import numpy as np
import pybullet


YCB_OBJECT_INFOS = {
    '002_master_chef_can': {'z':0.068,'s':[1.0,1.0,0.95],'m':0.2},
    '005_tomato_soup_can': {'z':0.05,'s':[1.3,1.3,1.2],'m':0.2},
    '007_tuna_fish_can':   {'z':0.016,'s':[1.1,1.1,1.8],'m':0.2},
    '004_sugar_box':   {'z':0.086,'s':[1.1,1.1,0.85],'m':0.05},
    '008_pudding_box': {'z':0.054,'s':[1.2,1.2,1.2],'m':0.05},
    '009_gelatin_box': {'z':0.0425,'s':[1.4,1.4,1.4],'m':0.05},
    '013_apple':   {'z':0.04,'s':[1.1,1.1,1.1],'m':0.1},
    '014_lemon': {'z':0.033,'s':[1.3,1.3,1.3],'m':0.1},
    '015_peach': {'z':0.034,'s':[1.1,1.1,1.1],'m':0.1},
    '016_pear':   {'z':0.04,'s':[1.1,1.1,1.1],'m':0.1},
    '017_orange': {'z':0.04,'s':[1.1,1.1,1.1],'m':0.1},
    '018_plum': {'z':0.03,'s':[1.4,1.4,1.4],'m':0.1},
    'sphere.urdf': {'z':0.04,'s':[1,1,1],'m':0.1},    # from sphere.urdf
    'sphere1.urdf': {'z':0.06,'s':[1,1,1],'m':0.3},   # from sphere1.urdf
    'cylinder.urdf': {'z':0.06,'s':[1,1,1],'m':0.2},  # from cylinder.urdf
    'cube.urdf': {'z':0.075,'s':[1,1,1],'m':0.05},    # from cube.urdf
    'block.urdf': {'z':0.075,'s':[1,1,1],'m':0.05},   # from block.urdf
    'cylinder_large.urdf': {'z':0.06,'s':[1,1,1],'m':0.2},  # from cylinder_large.urdf
    'cylinder_short.urdf': {'z':0.06,'s':[1,1,1],'m':0.2}   # from cylinder_short.urdf
}


def normalize(arr, mins, maxes):  # to [-1,1]
    arr_in01 = (arr - mins)/(maxes-mins)
    return arr_in01*2.0-1


def denormalize(arr, mins, maxes):  # from [-1,1]
    arr_in01 = (arr+1)/2.0
    return arr_in01*(maxes-mins) + mins


def theta_to_sin_cos(qpos):
    sin_cos = np.hstack([np.sin(qpos), np.cos(qpos)])
    return sin_cos.reshape(-1) # [sin,cos,sin,cos,...]


def sin_cos_to_theta(sin_cos):
    assert(len(sin_cos.shape)==1)
    assert((sin_cos.shape[0]%2)==0)  # need [sin,cos,sin,cos,...]
    sin_cos = sin_cos.reshape(-1,2)    # [[sin,cos],[sin,cos]...]
    theta = np.arctan2(sin_cos[:,0], sin_cos[:,1])
    return theta.reshape(-1)         # [theta0,theta1,...]


def quat_to_sin_cos(quat):
    assert(len(quat.shape)==1)
    assert(quat.shape[0]==4)    # [x,y,z,w]
    euler = pybullet.getEulerFromQuaternion(quat)
    euler = np.array(euler).reshape(3,1)
    return theta_to_sin_cos(euler)  # [sin,cos,sin,cos,sin,cos]


def sin_cos_to_quat(sin_cos):
    assert(len(sin_cos.shape)==1)
    assert(sin_cos.shape[0]==6)   # need [sin,cos,sin,cos,sin,cos]
    euler = sin_cos_to_theta(sin_cos)  # [theta0,theta1,theta2]
    assert(len(euler)==3)
    quat = pybullet.getQuaternionFromEuler(euler.tolist())
    return quat


def quat_to_mat(quat):
    assert(len(quat.shape)==1)
    assert(quat.shape[0]==4)    # [x,y,z,w]
    mat = np.array(pybullet.getMatrixFromQuaternion(quat)).reshape(-1)
    return mat


def convert_all(inp, fxn):
    assert(len(inp.shape)==2)
    outs = []
    for i in range(inp.shape[0]):
        outs.append(eval(fxn)(inp[i]))
    return np.vstack(outs)


def test_override(env):
    env.robot.reset()
    # Note: URDF files speicfy height (or length) for the cylinder, but we
    # specify halfHeight to be consistent (with raidus and halfExtent for
    # shpere and cuboid). For box: URDF says it expects halfExtents, but seems
    # to actually specify h,w,d... TODO: figure out why this is the case
    bkgrnd = np.array([220,200,170])/255.0
    obj_props = np.array(
        [[0, 1, 1,  0.04, 0.04, 0.13/2,      1,  0.2,  1.0, 1.0, 0.001, 0.000],   # cylinder
         [1, 1, 0,  0.05/2, 0.10/2, 0.15/2,  2,  0.05, 1.0, 1.0, 0.000, 0.000],   # cube
         [0, 1, 0,  0.04, 0.04, 0.04,        0,  0.1,  1.0, 0.0, 0.001, 0.001],   # sphere
         [1, 0, 1,  0.06, 0.06, 0.06,        0,  0.3,  1.0, 0.0, 0.001, 0.001]])  # sphere1
    obj_poses = np.array([[-0.20,-0.15,0],[-0.20,0.15,0],
                          [-0.05,-0.10,0],[-0.05,0.10,0]])
    obj_poses[:,2] = obj_props[:,5]    # z for cylinder,cube is halfExtent,radius
    obj_quats = np.array([
        pybullet.getQuaternionFromEuler([np.pi/4,0,0]),
        pybullet.getQuaternionFromEuler([0,0,np.pi/2]),
        pybullet.getQuaternionFromEuler([0,0,0]),
        pybullet.getQuaternionFromEuler([0,0,0])])
    num_obj = 1 if 'One' in env.spec.id else len(obj_poses)
    ld_state = env.sim_to_low_dim_state(
            bkgrnd, obj_props[0:num_obj],
            obj_poses[0:num_obj], obj_quats[0:num_obj])
    env.override_state(ld_state)
    env.render_obs()


# mat2quat function from:
# https://github.com/openai/gym/blob/master/gym/envs/robotics/rotations.py
#
# Need this because PyBullet does not expose getQuaternionFromMatrix()

# Copyright (c) 2009-2017, Matthew Brett and Christoph Gohlke
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#    THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#    CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#    EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#    PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q
