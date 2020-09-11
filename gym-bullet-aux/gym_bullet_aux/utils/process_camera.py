#
# Process depth and RGB camera input from PyBullet.
# The code follows basic recommendations from PyBullet forums.
# Note that it uses assumptions of the camera setup, which work in the
# current pybullet versions, but ultimately might change in the future.
# Using pybullet versions from 2.6.4 to 2.8.1 should work fine.
#
import os
import sys
import math
import time

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4, linewidth=150, threshold=np.inf, suppress=True)

import pybullet


def assert_close(ars, ars0):
    for ar, ar0 in zip(ars, ars0):
        assert(np.linalg.norm(np.array(ar)-np.array(ar0))<1e-6)


def plot_ptcloud(ptcloud, debug_out_dir, box_lim=1.0, view_elev=30, view_azim=-30):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.scatter(*ptcloud.T, marker='.', s=0.1, c=[(0.5,0.5,0.5)], alpha=0.5)
    ax.set_xlim(-box_lim, box_lim)
    ax.set_ylim(-box_lim, box_lim)
    ax.set_zlim(-box_lim, box_lim)
    ax.view_init(view_elev, view_azim)
    plt.tight_layout()
    fname = 'tmp_ptcld'+str(time.time())+'.png'
    plt.savefig(os.path.join(debug_out_dir, fname), dpi=100)
    plt.close()


class ProcessCamera:
    # In non-GUI mode we will render without X11 context but *with* GPU accel.
    # examples/pybullet/examples/testrender_egl.py
    # Note: use alpha=1 (in rgba), otherwise depth readings are not good
    # Using defaults from PyBullet.
    # See examples/pybullet/examples/pointCloudFromCameraImage.py
    PYBULLET_FAR_PLANE = 10000
    PYBULLET_NEAR_VAL = 0.01
    PYBULLET_FAR_VAL = 1000.0
    # Don't change default CAM_DIST and CAM_TGT without updating CAM_VALS.
    CAM_DIST = 0.85
    CAM_TGT = np.array([0.35, 0, 0])
    # Yaws and pitches are set to get the front and side view of the scene.
    # ATTENTION: CAM_VALS *HAVE* TO BE UPDATED IF CAM_YAWS, CAM_PITCHES,
    # CAM_DIST, or CAM_TGT_POS is changed.
    CAM_YAWS = list(range(-30,211,40)) # [-30, 0, ..., 170, 210]
    CAM_PITCHES = [-70, -10, -65, -40, -10, -25, -60]
    # The following camera values were obtained with getDebugVisualizerCamera()
    # with the above yaw,pitch,cam dist and tgt pos values and GUI on.
    # and GUI on. We cache them here, because we can't make this call when
    # we are using headless DIRECT backend.
    CAM_VALS = [
        # CAM_VALS for yaw -30 pitch -70
        [ (0.8660253882408142, 0.4698463976383209, -0.1710100769996643, 0.0, -0.5000000596046448, 0.8137977123260498, -0.2961980998516083, 0.0, 0.0, 0.3420201241970062, 0.9396926760673523, 0.0, -0.3031088709831238, -0.16444621980190277, -0.7901464700698853, 1.0) , (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0) , (0.1710100769996643, 0.2961980998516083, -0.9396926760673523) , (23094.009765625, -13333.3349609375, 0.0) , (9396.927734375, 16275.953125, 6840.40234375) , 0.8500000238418579 , (0.3499999940395355, 0.0, 0.0) ],
        # CAM_VALS for yaw 10 pitch -10
        [ (0.9848077893257141, -0.030153678730130196, 0.17101003229618073, 0.0, 0.1736481487751007, 0.17101004719734192, -0.969846248626709, 0.0, -0.0, 0.9848077297210693, 0.1736481487751007, 0.0, -0.3446827232837677, 0.01055377721786499, -0.909853458404541, 1.0) , (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0) , (-0.17101003229618073, 0.969846248626709, -0.1736481487751007) , (26261.541015625, 4630.6171875, -0.0) , (-603.0736083984375, 3420.201171875, 19696.154296875) , 0.8500000238418579 , (0.3499999940395355, 0.0, 0.0) ],
        # CAM_VALS for yaw 50 pitch -65
        [ (0.642787754535675, -0.694271981716156, 0.3237442970275879, 0.0, 0.7660443186759949, 0.582563579082489, -0.27165383100509644, 0.0, -0.0, 0.42261824011802673, 0.9063078165054321, 0.0, -0.22497569024562836, 0.24299520254135132, -0.9633104801177979, 1.0) , (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0) , (-0.3237442970275879, 0.27165383100509644, -0.9063078165054321) , (17141.0078125, 20427.84765625, -0.0) , (-13885.4384765625, 11651.271484375, 8452.3642578125) , 0.8500000238418579 , (0.3499999940395355, 0.0, 0.0) ],
        # CAM_VALS for yaw 90 pitch -40
        [ (0.0, -0.6427876353263855, 0.7660443782806396, 0.0, 0.9999999403953552, 0.0, -0.0, 0.0, -0.0, 0.7660443186759949, 0.6427876949310303, 0.0, -0.0, 0.22497573494911194, -1.1181155443191528, 1.0) , (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0) , (-0.7660443782806396, 0.0, -0.6427876949310303) , (0.0, 26666.66796875, -0.0) , (-12855.7529296875, 0.0, 15320.888671875) , 0.8500000238418579 , (0.3499999940395355, 0.0, 0.0) ],
        # CAM_VALS for yaw 130 pitch -10
        [ (-0.6427876949310303, -0.13302220404148102, 0.7544063925743103, 0.0, 0.7660443782806396, -0.11161890625953674, 0.6330222487449646, 0.0, 0.0, 0.9848076701164246, 0.1736481636762619, 0.0, 0.22497573494911194, 0.04655778408050537, -1.1140421628952026, 1.0) , (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0) , (-0.7544063925743103, -0.6330222487449646, -0.1736481636762619) , (-17141.00390625, 20427.8515625, 0.0) , (-2660.444091796875, -2232.3779296875, 19696.154296875) , 0.8500000238418579 , (0.3499999940395355, 0.0, 0.0) ],
        # CAM_VALS for yaw 170 pitch -25
        [ (-0.9848077893257141, -0.07338694483041763, 0.15737879276275635, 0.0, 0.17364828288555145, -0.416197806596756, 0.8925389051437378, 0.0, 0.0, 0.9063078165054321, 0.4226182997226715, 0.0, 0.3446827232837677, 0.02568545937538147, -0.905082643032074, 1.0) , (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0) , (-0.15737879276275635, -0.8925389051437378, -0.4226182997226715) , (-26261.541015625, 4630.62109375, 0.0) , (-1467.7388916015625, -8323.955078125, 18126.15625) , 0.8500000238418579 , (0.3499999940395355, 0.0, 0.0) ],
        # CAM_VALS for yaw 210 pitch -60
        [ (-0.8660253882408142, 0.43301278352737427, -0.2500000298023224, 0.0, -0.5000000596046448, -0.75, 0.4330126643180847, 0.0, 0.0, 0.5, 0.866025447845459, 0.0, 0.3031088709831238, -0.15155449509620667, -0.7625000476837158, 1.0) , (0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0) , (0.2500000298023224, -0.4330126643180847, -0.866025447845459) , (-23094.009765625, -13333.3359375, 0.0) , (8660.255859375, -15000.0, 9999.9990234375) , 0.8500000238418579 , (0.3499999940395355, 0.0, 0.0) ],
]
    @staticmethod
    def init(viz):
        if viz:
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 1)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_RGB_BUFFER_PREVIEW,1)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_DEPTH_BUFFER_PREVIEW,1)
            pybullet.configureDebugVisualizer(
                pybullet.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,1)
        #ProcessCamera.compute_cam_vals(ProcessCamera.CAM_DIST, ProcessCamera.CAM_TGT)

    @staticmethod
    def compute_cam_vals(cam_dist, cam_tgt, cam_yaws=None, cam_pitches=None):
        print('====== ATTENTION: if default cam_dist, cam_tgt are changed:')
        print('copy/paste the output below to ProcessCamera.CAM_VALS')
        if cam_yaws is None: cam_yaws = ProcessCamera.CAM_YAWS
        if cam_pitches is None: cam_pitches = ProcessCamera.CAM_PITCHES
        for i in range(len(cam_yaws)):
            pybullet.resetDebugVisualizerCamera(
                cameraDistance=cam_dist,
                cameraYaw=cam_yaws[i],
                cameraPitch=cam_pitches[i],
                cameraTargetPosition=cam_tgt)
            _, _, view_matrix, proj_matrix, _, cam_forward, cam_horiz, cam_vert, \
                _, _, cam_dist, cam_tgt = pybullet.getDebugVisualizerCamera()
            print('# CAM_VALS for yaw', cam_yaws[i], 'pitch', cam_pitches[i])
            print('[', view_matrix, ',', proj_matrix, ',', cam_forward, ',',
                  cam_horiz, ',', cam_vert, ',', cam_dist, ',', cam_tgt, '],')

    @staticmethod
    def get_ptcloud_obs(sim, object_ids, obs_resolution,
                        softbody_ids=None, width=100, cam_vals_list=None,
                        box_lim=1.0, view_elev=30, view_azim=-30,
                        debug_out_dir=None):
        debug = debug_out_dir is not None
        rigid_ptcloud, rigid_tracking_ids, _, _ = ProcessCamera.render_all(
            sim, object_ids, softbody_ids, width, cam_vals_list, debug)
        if len(rigid_ptcloud)==0:
            rigid_ptcloud = np.zeros((obs_resolution,3))
        elif rigid_ptcloud.shape[0] < obs_resolution:  # pad for enough pts
            rigid_ptcloud = np.concatenate([rigid_ptcloud, np.zeros(
                (obs_resolution-rigid_ptcloud.shape[0],3))])
        elif rigid_ptcloud.shape[0] > obs_resolution:
            rigid_ptcloud = rigid_ptcloud[np.random.choice(
                rigid_ptcloud.shape[0], obs_resolution, replace=False), :]
        assert(rigid_ptcloud.shape[0]==obs_resolution)
        rigid_ptcloud = np.clip(rigid_ptcloud, -1.0*box_lim, box_lim)
        if debug:
            plot_ptcloud(rigid_ptcloud, debug_out_dir,
                         box_lim, view_elev, view_azim)
        rigid_ptcloud = rigid_ptcloud.reshape(-1)
        return rigid_ptcloud

    @staticmethod
    def render_all(sim, object_ids, softbody_ids=None, width=100,
                   cam_vals_list=None, debug=False):
        rigid_ptcloud = []; rigid_tracking_ids = []
        if cam_vals_list is None: cam_vals_list = ProcessCamera.CAM_VALS
        #tic = time.time()
        for cam_vals in cam_vals_list:
            view_matrix, proj_matrix, cam_forward, cam_horiz, cam_vert, \
                cam_dist, cam_tgt = cam_vals
            w, h, rgba_px, depth_raw_cam_dists, segment_mask = sim.getCameraImage(
                width=width, height=width,
                viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                #renderer=pybullet.ER_TINY_RENDERER,
                flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
            #if debug: print('getCameraImage took {:4f}'.format(time.time()-tic))
            #tic = time.time()
            ptcloud, tracking_ids = ProcessCamera.make_ptcloud(
                sim, object_ids, depth_raw_cam_dists, segment_mask,
                cam_dist, cam_tgt, cam_forward, cam_horiz, cam_vert, debug=debug)
            #if debug: print('make_ptcloud took {:4f}'.format(time.time()-tic))
            rigid_ptcloud.extend(ptcloud)
            rigid_tracking_ids.extend(tracking_ids)
        rigid_ptcloud = np.array(rigid_ptcloud)
        rigid_tracking_ids = np.array(rigid_tracking_ids)
        # To fake SoftBody support: add vertex positions to the point cloud.
        #tic = time.time()
        deform_ptcloud = []; deform_tracking_ids = []
        if softbody_ids is not None:
            for i in range(len(softbody_ids)):
                num_verts, verts = sim.getMeshData(softbody_ids[i])
                for v in verts:
                    deform_ptcloud.append(np.array(v))
                    deform_tracking_ids.append(softbody_ids[i])
        # ATTENTION: the above is a temporary hack to use SoftBody objects
        # Ultimately, it would be good if their rendering was reported by the
        # bullet internal C/C++ code and passed to the python interface.
        #if debug: print('softbody verts took {:4f}'.format(time.time()-tic))
        deform_ptcloud = np.array(deform_ptcloud)
        deform_tracking_ids = np.array(deform_tracking_ids)
        return rigid_ptcloud, rigid_tracking_ids, \
               deform_ptcloud, deform_tracking_ids

    @staticmethod
    def make_ptcloud(sim, object_ids, depth_raw_cam_dists, segment_mask,
                     cam_dist, cam_tgt, cam_forward, cam_horiz, cam_vert, debug=False):
        far_plane = ProcessCamera.PYBULLET_FAR_PLANE
        near_val = ProcessCamera.PYBULLET_NEAR_VAL
        far_val = ProcessCamera.PYBULLET_FAR_VAL
        cam_pos = cam_tgt - cam_dist*np.array(cam_forward)
        ray_forward = (cam_tgt - cam_pos)
        ray_forward *= far_plane/np.linalg.norm(ray_forward)
        horizontal = np.array(cam_horiz); vertical = np.array(cam_vert)
        # http://web.archive.org/web/20130416194336/
        # http://olivers.posterous.com/linear-depth-in-glsl-for-real
        # z_n = 2*z-1 # z: [0,1] -> [-1,1]
        # z_d = 2*f*n/(f+n-(f-n)*z_n) = f*n/(f-(f-n)*z)  # simplifies
        tics = 0
        height, width = depth_raw_cam_dists.shape
        # 'Label' points in the cloud by the ID of the link.
        # examples/pybullet/examples/segmask_linkindex.py
        # We will use this to test TDA algorithms that need ID information,
        # and if these work well, then will later obtain this information
        # dynamically from a tracking system.
        pts_3d = []; tracking_ids = []
        grid_step = 1
        for y in range(0,height,grid_step):
            for x in range(0,width,grid_step):
                # Object ID is recorded in the lower 24 bits;
                # link is in the upper 32-24=8 bits (or 64-24 if 64 bit ints).
                #tic = time.time()
                obj_id = segment_mask[y,x] & ((1<<24)-1)  # note: y,x (not x,y)
                if obj_id not in object_ids: continue
                #link_id = (segment_mask[y,x]>>24)-1
                #print('obj_id', obj_id, 'link_id', link_id)
                #tic = time.time()
                ortho = (x*horizontal/width - horizontal/2 +
                         vertical/2 - y*vertical/height)
                vec = ray_forward + ortho  # rayTo in camera coords
                alpha = math.atan(np.linalg.norm(ortho)/far_plane)
                z_tmp = depth_raw_cam_dists[y,x]  # note: y,x (not x,y)
                depth = far_val*near_val/(far_val-(far_val-near_val)*z_tmp)
                res = (depth/np.cos(alpha))*(vec/np.linalg.norm(vec))
                # TODO: clarify why alpha is needed (strange if depth is
                # reported as projection on cam-tgt ray).
                #res = depth*(vec/np.linalg.norm(vec))
                pt_3d = cam_pos + res
                pts_3d.append(np.array(pt_3d))
                # Note: use # .append(segment_mask[y,x]) if we care about links
                tracking_ids.append(obj_id)
                #tics += time.time()-tic
        #if debug: print('depth comp took {:4f}'.format(tics))
        return pts_3d, tracking_ids

    @staticmethod
    def draw_point_clouds(fname_pfx, max_num_pts=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.axes3d import Axes3D
        draw_every = 1; max_steps = 10000
        cmap = plt.cm.jet
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        for step in range(0,max_steps+1,draw_every):
            fname = fname_pfx+'_step'+str(step)+'.npz'
            if step==0:
                print('Loading', fname)
                assert(os.path.exists(fname))  # assert npz for step=0 exists
            if not os.path.exists(fname): continue
            data = np.load(fname, allow_pickle=True)
            assert('ptcloud' in data.keys())
            assert('tracking_ids' in data.keys())
            pts  = data['ptcloud']
            ids  = data['tracking_ids']
            if step==0: print('ptcloud', pts.shape, 'tracking_ids', ids.shape)
            if pts.shape[0]==0: continue  # no pts to plot
            print('Draw step {:d} num_pts {:d} {:s}'.format(
                step, pts.shape[0],
                '' if max_num_pts is None else 'keeping '+str(max_num_pts)))
            if max_num_pts is not None:
                perm = np.random.permutation(pts.shape[0])[0:max_num_pts]
                pts = pts[perm]; ids = ids[perm]
            # Plot points using matplotlib Axes3D
            fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
            ax.scatter(pts[:,0], pts[:,1], pts[:,2], marker='.', s=2.0)
            #           c=ids[:,0], cmap=cmap)
            #if 'gripper_pos' in data.keys():
            #    gripper_pos = data['gripper_pos']
            #    ax.scatter(gripper_pos[0], gripper_pos[1], gripper_pos[2],
            #               marker='x', s=150.0, c='magenta')
            ax.set_xlim(-0.4,1.0); ax.set_ylim(-0.6,0.6); ax.set_zlim(0,0.8)
            ax.view_init(10, -25)
            plt.tight_layout()
            plt.show()


if __name__== "__main__":
    assert(len(sys.argv)>1)  # specify fname as argument
    max_num_pts = None
    if len(sys.argv) > 2: max_num_pts = int(sys.argv[2])
    fname_pfx = os.path.expanduser(sys.argv[1])
    ProcessCamera.draw_point_clouds(fname_pfx, max_num_pts)
