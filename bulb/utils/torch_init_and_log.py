"""
Utilities for logging to both stdout and file and for visualizing images
with tensorboardX (e.g. during training with pytorch).

"""
import argparse
from datetime import datetime
import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import torch
import tensorboardX
from tensorboardX.proto.summary_pb2 import Summary


def get_logger2_args():
    parser = argparse.ArgumentParser(description="Logger2Args", add_help=False)
    parser.add_argument('--save_path_prefix', type=str,
                        default='/tmp/tmp_learn', help='Output directory')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Logging interval')
    parser.add_argument('--viz_interval', type=int, default=1,
                        help='Visualization interval')
    _, args = parser.parse_known_args()
    return args, parser


def init_gpus_and_randomness(seed, gpu):
    # https://numpy.org/doc/stable/reference/random/generated/
    # numpy.random.seed.html : not clear, so using legacy for now.
    np.random.seed(seed)
    use_cuda = (gpu is not None) and torch.cuda.is_available()
    device = 'cuda:'+str(gpu) if use_cuda else 'cpu'
    torch.manual_seed(seed)  # same seed for CUDA to get same model weights
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.deterministic = False   # faster, less reproducible
        torch.cuda.manual_seed_all(seed)
    return device


def progress_bar(iter, total, pfx='', sfx='', decimals=1,
                 length=100, fill='=', endprint="\r"):
    # Based on: https://stackoverflow.com/questions/3173320/
    # text-progress-bar-in-the-console
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iter/float(total)))
    filled = int(length * (iter+1)//total)
    bar = '=' * filled + '>' + '-' * (length - filled)
    print(f'\r{pfx} |{bar}| {percent}% {sfx}', end=endprint)
    if iter+1==total: print('')


def make_border(img, clr_top, clr_left):
    # Draw a thin borders on the left side or top of img.
    clr_chn = img.size(0)
    for i in range(clr_chn): img[i,0,:] = clr_top[i]; img[i,:,0] = clr_left[i];


def make_pred_border(imgs, num_pred):
    # Draw a thin vertical red border on the left side of the given img.
    if num_pred==0: return # nothing to do
    clr_left = [1,0,0]
    clr_chn = imgs.size(2)
    for i in range(clr_chn): imgs[:,-num_pred:,i,:,1:3] = clr_left[i]


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


def combine(imgs_dict, num_steps, w, h):
    # Combine images in imgs_dict for each step side-by-side.
    from PIL import Image
    combo_imgs = []
    for st in range(num_steps):
        combo_img = Image.new('RGBA', (w*len(imgs_dict.keys()), h))
        x_offset = 0
        for key in imgs_dict.keys():
            im = imgs_dict[key][st]
            if im.dtype != np.uint8: im = (im*255).astype(np.uint8)
            if im.shape[0]==3: im = im.transpose(1,2,0)  # guess chnl is 0th dim
            imim = Image.fromarray(im)
            d = ImageDraw.Draw(imim)  # for color will use range
            d.text((5, 2), key, fill=(1, 1, 0))  # 0-1 not 255
            combo_img.paste(imim, box=(x_offset,0))
            x_offset += w
        combo_imgs.append(np.array(combo_img))
    return combo_imgs


def add_video(name, imgs, epoch, tb_writer):
    # Make a video from the given imgs and output to tensorboard.
    strt = time.time()
    vid_imgs = np.array(imgs)  # TxHxWxC, C=RGBA
    video = tensorboardX.summary.make_video(vid_imgs, fps=12)  # slower: fps<24
    vs = Summary(value=[Summary.Value(tag=name+'_video', image=video)])
    tb_writer.file_writer.add_summary(vs, global_step=epoch)
    # add_video takes NxTxHxWxC and fails on RGBA
    #tb_writer.add_video(name+'_video', vid_tensor=vid_imgs, fps=4)
    print('{:s} video took {:2f}'.format(name, time.time()-strt))


def image_from_env(viz_env, obs, act, data_h, data_w):
    # Make an RGB image representation by overriding the state of viz_env with
    # the given obs.
    if hasattr(viz_env, 'override_state'):
        state = np.clip(obs, viz_env.low_dim_state_space.low,
                        viz_env.low_dim_state_space.high)
        viz_env.override_state(state)
        img = torch.from_numpy(viz_env.render_obs(override_resolution=data_h))
    else:
        assert(viz_env.spec.id=='Pendulum-v0')
        sin_th, cos_th, th_dot = obs[:]
        viz_env.unwrapped.state[0] = np.arctan2(sin_th, cos_th)
        viz_env.unwrapped.state[1] = th_dot
        viz_env.unwrapped.last_u = act[0].item()
        img = Image.fromarray(viz_env.render(mode='rgb_array'))
        img = img.resize((data_w,data_h), Image.ANTIALIAS)
        img = ImageOps.invert(img)
        img = np.array(img)
        img = torch.from_numpy(img).float().permute(2,0,1)
        img = (255-img[:,:,:])/255.0  # convert to [0,1] floats
    return img


class Logger2:
    # Logger with utilities to output to stdout, logfile and tensorboard.
    # Note: not using logging module from python, because it does not play
    # well with multiprocessing and tensorboardX (e.g. need to do TB imports
    # before the logger is initialized if using stdout and file handlers.
    def __init__(self, save_path_prefix, use_tensorboardX=True):
        date_str = datetime.strftime(datetime.today(), "%y%m%d_%H%M%S")
        save_path = os.path.join(os.path.expanduser(save_path_prefix), date_str)
        assert(not os.path.exists(save_path)); os.makedirs(save_path)
        log_fnm = os.path.join(save_path, 'log.txt')
        self._log_file = open(log_fnm, 'w', buffering=1)
        self._save_path = save_path
        self._checkpt_path = os.path.join(save_path, 'checkpt-%04d.pth' % 0)
        self._tb_writer = None
        if use_tensorboardX:
            self._tb_writer = tensorboardX.SummaryWriter(save_path)
        print('Logger2', log_fnm)

    def __del__(self):
        self._log_file.close()

    @property
    def checkpt_path(self):
        return self._checkpt_path

    @property
    def save_path(self):
        return self._save_path

    @property
    def tb_writer(self):
        return self._tb_writer

    def log(self, data_lst):
        # Log to stdout and to log_file
        tm = datetime.now().strftime('%H:%M:%S')
        if type(data_lst) is not list: data_lst = [data_lst]  # for common api
        for data in data_lst:
            print(tm, data)
            self._log_file.write(tm+' '+str(data)+'\n')

    def log_tb_scalars(self, dct, epoch):
        # Log scalars in dct to tensorboard.
        if self._tb_writer is None: return  # nothing to do
        for k,v in dct.items():
            self._tb_writer.add_scalar(k,v,epoch)
        self._tb_writer.flush()

    def log_tb_object(self, obj, title=''):
        # Print all fields of the given object as text in tensorboard.
        if self._tb_writer is None: return  # nothing to do
        text_str = ''
        for member in vars(obj):
            # Tensorboard uses markdown-like formatting, hence '  \n'.
            text_str += '  \n{:s}={:s}'.format(
                str(member), str(getattr(obj, member)))
        self._tb_writer.add_text(title, text_str, 0)
        self.log(title+' '+text_str)

    def viz_pred_seq(self, epoch, obs, act, pred, nn_out, pfx, viz_env):
        # Visualize historical observations from obs and future observations
        # from pred vs a predictive sequence in nn_out.
        bsz, hist_len, obs_sz = obs.size(); pred_len = pred.size(1)
        losses = torch.pow(nn_out-pred, 2).view(bsz, -1).mean(dim=1)
        nviz=8  # make it a multiple of 4 for convenience
        losses = losses.detach().cpu()
        _, good_ids = torch.topk(losses,k=int(nviz/4),largest=False)
        _, bad_ids = torch.topk(losses,k=int(nviz/4),largest=True)
        rnd_ids = np.random.choice(bsz, size=int(nviz/2))
        ids = np.concatenate(
            [good_ids.numpy(), bad_ids.numpy(), rnd_ids], axis=0)
        obs = obs.cpu().numpy(); act = act.cpu().numpy()
        pred = pred.cpu().numpy(); nn_out = nn_out.detach().cpu().numpy()
        chnl=3; data_h=128; data_w=128
        true_imgs = torch.zeros(nviz,hist_len+pred_len,chnl,data_h,data_w)
        pred_imgs = torch.zeros(nviz,hist_len+pred_len,chnl,data_h,data_w)
        for i in range(ids.shape[0]):
            bid = ids[i]
            for tid in range(hist_len):
                true_imgs[i,tid,:,:,:] = image_from_env(
                    viz_env, obs[bid,tid], act[bid,tid], data_h, data_w)
                pred_imgs[i,tid,:,:,:] = true_imgs[i,tid,:,:,:]  # same hist
            for tid in range(pred_len):
                true_imgs[i,hist_len+tid,:,:,:] = image_from_env(
                    viz_env, pred[bid,tid], act[bid,tid], data_h, data_w)
                pred_imgs[i,hist_len+tid,:,:,:] = image_from_env(
                    viz_env, nn_out[bid,tid], act[bid,tid], data_h, data_w)
        make_pred_border(pred_imgs, pred_len)
        final_img = compose_img(
            [true_imgs, pred_imgs], ['true', 'pred'], nviz, data_h, data_w)
        self._tb_writer.add_image(pfx+'_true_vs_pred', final_img, epoch)
        video_combo_imgs = []  # make video images after make_pred_border()
        for i in range(ids.shape[0]):
            lbl = 'good' if i<int(nviz/4) else 'bad' if i<int(nviz/2) else 'rnd'
            pred_str = f'pred {losses[ids[i]].item():.2e} loss\n{lbl:s}'
            img_dict_for_video = {
                'true':true_imgs[i].numpy(), pred_str:pred_imgs[i].numpy()}
            video_combo_imgs.extend(combine(
                img_dict_for_video, hist_len+pred_len, data_h, data_w))
        add_video(pfx+'_true_vs_pred', video_combo_imgs, epoch, self._tb_writer)
