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


def mat_to_quat(quat):
    # code from gym/envs/robotics/rotations.py
    # installing code that depends on mujoco-py is extremely painful...
    # TODO:check for liecense, if restrictive: re-write the code.
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))


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
    flat_state = env.sim_to_flat_state(
            bkgrnd, obj_props[0:num_obj],
            obj_poses[0:num_obj], obj_quats[0:num_obj])
    env.override_state(flat_state)
    env.render_obs()
