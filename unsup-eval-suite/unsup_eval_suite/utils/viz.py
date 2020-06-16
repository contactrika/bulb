"""
Code to visualize samples and latent walks for VAE.
"""

from collections import deque
import os
import numpy as np
import torch

import gym

from ..utils.data_utils import extract_tgts
from ..utils.prob import get_log_lik

from PIL import Image, ImageDraw, ImageFont

from .load_log_save import log_info


def make_border(img, clr_top, clr_left):
    clr_chn = img.size(0)
    for i in range(clr_chn): img[i,0,:] = clr_top[i]; img[i,:,0] = clr_left[i];


def make_pred_border(imgs, num_pred):
    if num_pred==0: return # nothing to do
    clr_left = [1,0,0]
    clr_chn = imgs.size(2)
    for i in range(clr_chn): imgs[:,-num_pred:,i,:,1:3] = clr_left[i]


def make_chosen_border(imgs, mix_ids, curr_q_id):
    bsz = imgs.size(0); clr_chn = imgs.size(2)
    assert(bsz == mix_ids.size(0))
    clr_top = [1,0.5,0]
    for bid in range(bsz):
        if curr_q_id == mix_ids[bid]:
            for i in range(clr_chn): imgs[bid,:,i,1,:] = clr_top[i]


def make_act_annotations(imgs, acts, num_pred):
    #fnt_path = os.path.join(os.path.split(__file__)[0], 'FreeMono.ttf')
    #fnt = ImageFont.truetype(fnt_path, 12)
    batch_size, seq_len, clr_chnls, data_h, data_w = imgs.size()
    for b in range(batch_size):
        for t in range(seq_len):
            act_non_zeros = torch.nonzero(acts[b,t]).reshape(-1)
            act_str = str(act_non_zeros[0].item())
            assert(data_h%8==0); assert(data_w%8==0)
            hh = data_h//8; ww = data_w//8
            #    qpos = acts[b,t,acts.size(-1)//2:]  # assuming torque control
            #    act_str = ''
            #    for i in range(len(qpos)): act_str += '{:.0f};'.format(qpos[i])
            #    assert(data_h%4==0); assert(data_w%2==0)
            #    hh = data_h//4; ww = data_w//2
            #    d.text((-2, 10), str(act_str), fill=clr, font=fnt)
            act_img = Image.new('RGB', (hh, ww), color=(0, 0, 0))
            d = ImageDraw.Draw(act_img)  # for color will use range
            clr = (1, 1, 1) if t<seq_len-num_pred else (1, 0, 0)
            d.text((2, -1), str(act_str), fill=clr)
            act_img = act_img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
            act_img = torch.tensor(np.array(act_img).swapaxes(0,2))
            imgs[b,t,:,-hh:,-ww:] = act_img[:,:,:]


def add_image_seqs(tb_writer, name, x_1toT, epoch):
  batch_size, seq_len, clr_chnls, data_h, data_w = x_1toT.size()
  seqs = torch.zeros(clr_chnls, data_h*batch_size, data_w*seq_len)
  for b in range(batch_size):
      for t in range(seq_len):
          seqs[:,data_h*b:data_h*(b+1),data_w*t:data_w*(t+1)] = x_1toT[b,t,:,:,:]
  tb_writer.add_image(name, seqs, epoch)


def add_video(name, imgs, epoch, tb_writer):
    #strt = time.time()
    from tensorboardX import summary
    from tensorboardX.proto.summary_pb2 import Summary
    vid_imgs = np.array(imgs)  # TxHxWxC, C=RGBA
    video = summary.make_video(vid_imgs, fps=24)
    vs = Summary(value=[Summary.Value(tag=name+'_video', image=video)])
    tb_writer.file_writer.add_summary(vs, global_step=epoch)
    # add_video takes NxTxHxWxC and fails on RGBA
    #tb_writer.add_video(name+'_video', vid_tensor=vid_imgs, fps=4)
    #print('{:s} video took {:2f}'.format(name, time.time()-strt))


def compose_img(all_imgs, all_txts, nviz, data_h, data_w):
    # Compose one image to viz all imgs.
    max_seq_len = all_imgs[0].size(1)
    for img_seq in all_imgs:
        if img_seq.size(1)>max_seq_len: max_seq_len = img_seq.size(1)
    seqs = torch.zeros(3, data_h*nviz*len(all_imgs), data_w*(max_seq_len+1))
    for t in range(max_seq_len+1):
        hoffst = 0
        wid = data_w*t
        for b in range(nviz):
            for k in range(len(all_imgs)):
                curr_img = None
                if t == 0:  # slot with t=0 gets text, images start @t=1
                    txtimg = Image.new('RGB', (data_h, data_w), color=(0, 0, 0))
                    d = ImageDraw.Draw(txtimg)  # for color will use range
                    d.text((2, 10), all_txts[k], fill=(1, 1, 0))  # 0-1 not 255
                    imgr = np.flip(np.array(txtimg), axis=1)  # flip along h
                    imgr = np.array(imgr).swapaxes(0,2)
                    curr_img = torch.tensor(imgr)
                elif t <= all_imgs[k].size(1):  # more imgs in this seq
                    curr_img = all_imgs[k][b,t-1,:,:,:]
                if curr_img is not None:
                    # Add teal border to separate images, yellow to end batch.
                    clr_top = [1,1,0] if k==0 else [0,1,1]
                    make_border(curr_img, clr_top, [0,1,1])
                    # Now save this in the overall image.
                    _, tmp_h, tmp_w = curr_img.size()
                    seqs[:, hoffst:hoffst+tmp_h, wid:wid+tmp_w] = curr_img
                hoffst += data_h
    return seqs


def recon_swap(unsup, x_lst,  acts_lst, f_smpl_lst, z_smpls_lst,
               imgs, txts):
    assert(len(f_smpl_lst)==2)
    for item in f_smpl_lst: assert(item is not None)
    for item in z_smpls_lst: assert(item is not None)
    # Swap statics f.
    constr_imgs0_f1, _, _ = decode_debug(
        unsup, x_lst[0], acts_lst[0], f_smpl_lst[1], z_smpls_lst[0])
    constr_imgs1_f0, _, _ = decode_debug(
        unsup, x_lst[1], acts_lst[1], f_smpl_lst[0], z_smpls_lst[1])
    make_pred_border(constr_imgs0_f1, unsup.pr.pred)
    make_pred_border(constr_imgs1_f0, unsup.pr.pred)
    imgs.extend([constr_imgs0_f1, constr_imgs1_f0])
    txts.extend(['st1_dyn0', 'st0_dyn1'])


def decode_debug(unsup, x_1toT, act_1toT, f_smpl=None, z_smpls=None):
    with torch.no_grad():
        if hasattr(unsup, 'encoder_static'):  # DSA
            res = unsup.recon(x_1toT, act_1toT, f_smpl=f_smpl, z_smpls=z_smpls)
            recon_xs, f_smpl, _, z_smpls, _ = res
        else:  # VAE, SVAE
            f_smpl = None
            recon_xs, z_smpls, _ = unsup.recon(x_1toT, act_1toT)
        return recon_xs, f_smpl, z_smpls


def viz_samples(unsup, x_1toL, act_1toL, aux_1toL, epoch, logfile,
                tb_writer, viz_env, title_prefix, max_num_viz=8):
    assert((type(x_1toL)==torch.Tensor) and (x_1toL.dim()==5))
    batch_size, tot_seq_len, clr_chnls, data_w, data_h = x_1toL.size()
    assert(batch_size > 1)
    nviz = min(max_num_viz, batch_size//2)
    sim_env = None
    if (aux_1toL is not None and hasattr(viz_env.venv, 'envs') and
        'Known' in viz_env.venv.envs[0].spec.id):
        sim_env = viz_env.venv.envs[0]
    discr_acts = isinstance(viz_env.action_space, gym.spaces.Discrete)
    # Make two sets of input images from data.
    # Rearrange batches, otherwise first nviz will be from the same world.
    perm = torch.randperm(batch_size)
    ids0 = perm[:nviz]; ids1 = perm[nviz:(nviz*2)]
    tmp_x_lst = []; acts_lst = []
    f_smpl_lst = []; z_smpls_lst = [];
    all_imgs = []; all_txts = []
    #log_info(logfile, ['{:s} viz'.format(title_prefix)])
    recon_log_lik = None; npred = 0
    if hasattr(unsup, 'pr'): npred = unsup.pr.pred
    for vid, ids in enumerate([ids0, ids1]):
        tmp_x_1toL = x_1toL[ids]
        tmp_act_1toL = act_1toL[ids]
        tmp_aux_1toL = None if aux_1toL is None else aux_1toL[ids]
        res = extract_tgts(tmp_x_1toL, tmp_act_1toL, tmp_aux_1toL,
                           unsup.pr.hist, unsup.pr.past, unsup.pr.pred)
        tmp_x_1toT, tmp_act_1toT, tmp_aux_1toT, \
            tmp_xs_tgt, tmp_acts_tgt, tmp_auxs_tgt = res

        # Visualize original image sequence (hist and pred).
        imgs_copy = tmp_x_1toL.clone()
        make_pred_border(imgs_copy, npred)
        if discr_acts:
            make_act_annotations(imgs_copy, tmp_act_1toL, npred+1)
        all_imgs.extend([imgs_copy])
        all_txts.extend(['orig'+str(vid)+'_1toL'])

        # Visualize simulator state from aux (hist and pred).
        if sim_env is not None:
            assert(tmp_aux_1toL is not None)
            sim_imgs = np.zeros([*tmp_x_1toL.size()])
            for tmp_bid in range(tmp_aux_1toL.size(0)):
                for tmp_t in range(tmp_aux_1toL.size(1)):
                    flat_state = tmp_aux_1toL[tmp_bid,tmp_t].cpu().numpy()
                    sim_env.override_state(flat_state)
                    sim_imgs[tmp_bid,tmp_t] = sim_env.render_obs()
            all_imgs.extend([torch.from_numpy(sim_imgs)])
            all_txts.extend(['sim'+str(vid)+'_1toL'])

        # Visualize target images (if different from hist)
        if npred>0 and unsup.pr.hist != unsup.pr.past:
            imgs_tgt_copy = tmp_xs_tgt.clone()
            make_pred_border(imgs_tgt_copy, npred)
            if discr_acts:
                make_act_annotations(imgs_tgt_copy, tmp_acts_tgt, npred+1)
            all_imgs.extend([imgs_tgt_copy])
            all_txts.extend(['orig'+str(vid)+'_tgt'])

        # Visualize images from unsupervised learner.
        recon_xs, f_smpl, z_smpls = decode_debug(unsup, tmp_x_1toT, tmp_act_1toT)
        make_pred_border(recon_xs, npred)
        all_imgs.extend([recon_xs])
        ttl = 'recon' if unsup.pr.pred==0 else 'psr'
        all_txts.extend([ttl+str(vid)])
        tmp_x_lst.append(tmp_x_1toT)  # for swap
        acts_lst.append(tmp_act_1toT)
        f_smpl_lst.append(f_smpl); z_smpls_lst.append(z_smpls)

        # Report recon_log_lik for streaming case.
        if 'streaming' in title_prefix:
            ll = get_log_lik(tmp_xs_tgt, recon_xs)
            recon_log_lik = ll
            if recon_log_lik is not None:
                recon_log_lik = torch.cat([recon_log_lik, ll], dim=0)

    if recon_log_lik is not None:
        recon_log_lik_val = recon_log_lik.mean().item()
        log_info(logfile, ['streaming_recon_log_lik {:0.4f}'.format(
            recon_log_lik_val)])
        tb_writer.add_scalar('streaming_recon_log_lik', recon_log_lik_val, epoch)

    if hasattr(unsup, 'encoder_static'):
        recon_swap(unsup, tmp_x_lst, acts_lst, f_smpl_lst, z_smpls_lst,
                   all_imgs, all_txts)
    final_img = compose_img(all_imgs, all_txts, nviz, data_h, data_w)
    tb_writer.add_image('recon_'+title_prefix, final_img, epoch)
