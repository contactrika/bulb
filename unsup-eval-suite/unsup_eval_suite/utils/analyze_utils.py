"""
Utils for analyzing alignment of latents and low-dim space.
"""

import numpy as np

import torch

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

from ..svae.svae_nets import make_MLP


GEOM_DICT = {'geomsphere':0, 'geomcylinder':1, 'geombox':2}


class IMatsTensors(torch.nn.Module):
    def __init__(self, device):
        super(IMatsTensors, self).__init__()
        I = torch.diag(torch.tensor([1,1,1])).float().to(device).unsqueeze(0)
        I1 = torch.diag(torch.tensor([-1,-1,1])).float().to(device).unsqueeze(0)
        I2 = torch.diag(torch.tensor([-1,1,-1])).float().to(device).unsqueeze(0)
        I3 = torch.diag(torch.tensor([1,-1,-1])).float().to(device).unsqueeze(0)
        self.register_buffer('I', I)
        self.register_buffer('I1', I)
        self.register_buffer('I2', I)
        self.register_buffer('I3', I)
        self.Imats = (self.I, self.I1, self.I2, self.I3)
        # [0,0,1] https://github.com/pytorch/pytorch/issues/1828
        e_z = torch.tensor([0,0,1]).float().to(device).unsqueeze(0)
        self.register_buffer('e_z', e_z)


def print_progress_header(lowdim_nms):
    print('Diff\titer', end='')
    for i in range(len(lowdim_nms)): print('\t',lowdim_nms[i],end='')
    print('')


def print_progress(iter, out_true, lowdim_hat):
    diff = torch.abs(out_true - lowdim_hat)
    if iter is not None:
        print('Anlz\t{:04d}'.format(iter), end='')
    else:
        print('Test\t', end='')
    for d in range(diff.size(-1)):
        val = diff[:,d].mean().item()
        print('\t{:0.4f}'.format(val), end='')
    print('')
    for dbg_id in [0,3,5,7]:
        if dbg_id>=out_true.size(0): break  # a tiny dataset
        print('true', out_true[dbg_id,:].cpu().numpy())
        print(' hat', lowdim_hat[dbg_id,:].detach().cpu().numpy())


def  fill_debug_dicts(lowdims, lowdim_hat, pfx, lowdim_nms,
                      debug_dict, debug_dict_hist):
    # dicts are passed by referece, so will be modified here
    diff = torch.abs(lowdims - lowdim_hat)
    nlowdims = diff.size(-1)
    for d in range(nlowdims):
        pfxd = pfx+'_'+lowdim_nms[d]
        debug_dict[pfxd+'_diff'] = diff[:,d].mean().item()
        debug_dict_hist[pfxd+'_true'] = lowdims[:,d]
        debug_dict_hist[pfxd+'_hat'] = lowdim_hat[:,d]


def compute_eucl_and_ori_starts(lowdim_nms):
    print('compute_eucl_and_ori_starts()...')
    eucl_dims = []; geom_ori_starts = []; geom_ori_types = []
    dim = 0; ndims = len(lowdim_nms)
    while dim<ndims:
        dim_nm = lowdim_nms[dim]
        geom_type = None
        for k, v in GEOM_DICT.items():
            if k in dim_nm and '_rot' in dim_nm: geom_type = v
        if 'mesh' in dim_nm and '_rot' in dim_nm:  # treat YCB objects as cyl.
            geom_type = GEOM_DICT['geomcylinder']
        if geom_type is None: eucl_dims.append(dim); dim+=1; continue  # not ori
        # Assume that ori comes as [rot0,...,rot8].
        for tmpi in range(9): assert('_rot' in lowdim_nms[dim+tmpi])
        geom_ori_starts.append(dim); geom_ori_types.append(geom_type)
        dim+=9
    print('eucl_dims', eucl_dims, 'geom_ori_starts', geom_ori_starts,
          'geom_ori_types', geom_ori_types)
    return eucl_dims, geom_ori_starts, geom_ori_types


def symm_ori_loss(rot_true, rot_hat, geom_type, Imats):
    # For cylinders: we only need to ensure that R_true and R_hat yield
    # similar trnasformations to the main axis of the cylinder (by convention
    # this would be the z axis).
    # For that, we take the last column of R_true * (R_hat)^T
    # and ensure that it is close to either [0,0,1] or [0,0,-1].
    # We will also use this approach for determining how well the principal
    # axes of a box is learned, and for spheres as a control.
    #
    # Note that in pytorch: 'Matrix multiplication is always done with using the
    # last two dimensions. All the ones before are considered as batch.'
    # https://discuss.pytorch.org/t/understanding-batch-multiplication-using-torch-matmul/16882
    best_loss = None
    rrt = torch.matmul(rot_true.transpose(1,2), rot_hat)
    for e_vec in [Imats.e_z, -Imats.e_z]:
        diff = torch.abs(rrt[:,:,2] - e_vec)
        if best_loss is None: best_loss = diff; continue
        best_loss = torch.where(diff<best_loss, diff, best_loss)
    #
    # For boxes with 3 axes of symmetry if:
    # R_hat (R_true)^T = I, I_0, I_1 or I_2 then the error is minimized.
    # I_0,I_1,I_2 are matrices with [-1,-1,1],[-1,1,-1],[1,-1,-1] on diag.
    # So the loss is the minimum over these 4 opions.
    # Note that rotation matrices are orthogonal matrices, so R^{-1} = R^T
    # Not using this code for now, since for rearrange geom envs we just
    # wanted a simple test of how well the pcincipal axes of the box is learned.
    #if geom_type==2:  # box
    #    for Imat in Imats.Imats:
    #        diff = (rrt.view(-1,9) - Imat.view(-1,9))**2
    #        if best_loss is None: best_loss = diff; continue
    #        best_loss = torch.where(diff<best_loss, diff, best_loss)
    assert(best_loss is not None)
    return best_loss


def combo_loss(x_true, x_hat, eucl_dims, geom_ori_starts, geom_ori_types,
               Imats, debug_dict=None, debug_pfx=None, lowdim_nms=None):
    losses = []
    # For euclidean dims we use L1 loss, because we care about being able to
    # reduce the error to 0 and do not care as much about being wrong on a
    # few outliers (which L2 loss would penalize heavily). L1 is also easy
    # to interpret, since it is in the same units as the target quantities
    # (no need to sqrt).
    # [ L2 used to be: loss = 0.5*((diff)**2).mean() ]
    if len(eucl_dims)>0:
        losses.append(torch.abs(x_hat[:,eucl_dims] - x_true[:,eucl_dims]))
    # Here we take care of rotational symmetries. This is needed since
    # otherwise the simplified environments with plain geometric shapes
    # could have large errors despite good reconstructions.
    for dim, geom_type in zip(geom_ori_starts, geom_ori_types):
        rot_true = x_true[:,dim:dim+9].view(-1,3,3)
        rot_hat = x_hat[:,dim:dim+9].view(-1,3,3)
        loss = symm_ori_loss(rot_true, rot_hat, geom_type, Imats)
        losses.append(loss)
        if debug_dict is not None:
            debug_dict[debug_pfx+'_'+lowdim_nms[dim]+'_ori_loss'] = loss.mean()
    assert(len(losses)>0)
    final_loss = losses[0].mean()
    for loss in losses[1:]: final_loss += loss.mean()
    return final_loss


def analyze_with_nns(train_latents, train_lowdims, test_latents, test_lowdims,
                     test_curr_latents, test_curr_lowdims, lowdim_nms, args):
    # Split train set into train and validation.
    sz = train_latents.size(0)
    assert(sz>2)  # need at least two points in the training set
    vsz = int(sz*0.3)
    if vsz==0: vsz=1
    validation_latents = train_latents[0:vsz,:]
    validation_lowdims = train_lowdims[0:vsz,:]
    train_latents = train_latents[vsz:,:]
    train_lowdims = train_lowdims[vsz:,:]
    mbsz = min(args.unsup_batch_size*4, train_latents.size(0));
    vmbsz = min(args.unsup_batch_size*4, validation_latents.size(0));
    hsz = args.analyze_hsz
    Imats = IMatsTensors(args.device)
    res = compute_eucl_and_ori_starts(lowdim_nms)
    eucl_dims, geom_ori_starts, geom_ori_types = res
    # TODO: maybe create regr_model elsewhere, and just re-init here.
    h = [hsz,hsz]
    regr_model = make_MLP(train_latents.size(-1), train_lowdims.size(-1),
                          drop=0.2, hidden=h, nl=torch.nn.ReLU(), out_nl=None)
    regr_model.to(args.device)
    regr_optim = torch.optim.Adam(
        [{"params": regr_model.parameters(), "lr": args.analyze_lr}])
    print_progress_header(lowdim_nms)
    debug_dict = {}; debug_dict_hist = {}
    max_patience = None; patience = max_patience; prev_validation_loss_mean = None
    pfx = 'anlz_'+args.env_name+'_'
    for iter in range(int(args.analyze_max_iter)):
        bids = torch.randperm(train_latents.size(0))[0:mbsz]
        train_inp = train_latents[bids]
        train_out_true = train_lowdims[bids]
        regr_model.train()
        train_lowdim_hat = regr_model(train_inp)
        loss = combo_loss(train_out_true, train_lowdim_hat,
                          eucl_dims, geom_ori_starts, geom_ori_types, Imats)
        regr_optim.zero_grad()
        loss.backward()
        regr_optim.step()
        sfx = '_'.join([str(_h) for _h in h])
        debug = (iter%1000==0 or iter+1>=args.analyze_max_iter) and len(h)==2
        if debug: print_progress(iter, train_out_true, train_lowdim_hat)
        if max_patience is not None and iter%100==0:  # do early stopping
            vbids = torch.randperm(validation_latents.size(0))[0:vmbsz]
            validation_lowdim_hat = regr_model(validation_latents[vbids])
            validation_loss = combo_loss(
                validation_lowdims[vbids], validation_lowdim_hat, eucl_dims,
                geom_ori_starts, geom_ori_types, Imats)
            improved = True; curr_validation_loss_mean= validation_loss.mean()
            if prev_validation_loss_mean is not None:
                improved = curr_validation_loss_mean < prev_validation_loss_mean
            patience = max_patience if improved else patience-1
            if improved: prev_validation_loss_mean = curr_validation_loss_mean
            print('iter {:d} validation loss {:0.4f}'.format(
                iter, curr_validation_loss_mean))
        # Eval model on the last iter and fill debug_dict_all
        if iter+1==args.analyze_max_iter or patience==0:
            fill_debug_dicts(train_out_true, train_lowdim_hat,
                             pfx+sfx+'_train', lowdim_nms,
                             debug_dict, debug_dict_hist)
            if len(geom_ori_starts)>0:
                combo_loss(train_out_true, train_lowdim_hat, eucl_dims,
                           geom_ori_starts, geom_ori_types, Imats,
                           debug_dict, pfx+sfx+'_train', lowdim_nms)
            debug_dict[pfx+sfx+'_train_loss'] = loss.mean().item()
            regr_model.eval()
            test_lowdim_hat = regr_model(test_latents)
            #print('TODO: remove after debugging')
            #test_lowdim_hat = test_latents[:,0:6]
            print_progress(None, test_lowdims, test_lowdim_hat)
            fill_debug_dicts(test_lowdims, test_lowdim_hat, pfx+sfx+'_test',
                             lowdim_nms, debug_dict, debug_dict_hist)
            if len(geom_ori_starts)>0:
                combo_loss(test_lowdims, test_lowdim_hat, eucl_dims,
                           geom_ori_starts, geom_ori_types, Imats,
                           debug_dict, pfx+sfx+'_test', lowdim_nms)
            if test_curr_latents is not None:
                test_curr_lowdim_hat = regr_model(test_curr_latents)
                print_progress(None, test_curr_lowdims, test_curr_lowdim_hat)
                fill_debug_dicts(test_curr_lowdims, test_curr_lowdim_hat,
                                 pfx+sfx+'_test_curr',
                                 lowdim_nms, debug_dict, debug_dict_hist)
                if len(geom_ori_starts)>0:
                    combo_loss(test_curr_lowdims, test_curr_lowdim_hat,
                               eucl_dims, geom_ori_starts, geom_ori_types,
                               Imats, debug_dict, pfx+sfx+'_test_curr',
                               lowdim_nms)
            regr_model.train()
        if patience==0: print('Analyze stopping early at iter', iter); break
    return debug_dict, debug_dict_hist


def analyze_irregularity(latents, lowdims, lowdim_nms, dbg_pfx):
    nmax = 500
    if latents.size(0)>nmax: latents = latents[:nmax]; lowdims = lowdims[:nmax]
    # See how well-behaved the learned data manifold is: compute
    # irregularity of stretching of the encoder map.
    # For each pair of true lowdim states: see how far the corresponding
    # latents are vs the distance between lowdim states.
    # Take the logs of all these measuremens and report the STD.
    # This is a measure of how well-behaved the stretching of the latent
    # space is in comparison to lowdim states (and is invariant to simple
    # uniform scaling of the encoder mappings).
    # We make sure that categorical fields coincide by only looking at
    # pairs with differences < 1 in all dims (assuming categorical values
    # are encoded as integers).
    cont_dims = []; cat_dims = []
    for tmpd, nm in enumerate(lowdim_nms):
        if nm.endswith('_id'):
            cat_dims.append(tmpd)
        else:
            cont_dims.append(tmpd)
    cat_max = 1e-5
    lowdim_max_cat_dists = torch.nn.functional.pdist(
        lowdims[:,cat_dims], p=float('Inf'))
    ok_max_dists = torch.where(lowdim_max_cat_dists<cat_max,
                               torch.ones_like(lowdim_max_cat_dists),
                               torch.zeros_like(lowdim_max_cat_dists))
    ok_cat_ids = torch.nonzero(ok_max_dists)
    if len(ok_cat_ids)<2: return {}, {}
    lowdim_dists = torch.nn.functional.pdist(
        lowdims[:,cont_dims])[ok_cat_ids]
    latent_dists = torch.nn.functional.pdist(
        latents[:,cont_dims])[ok_cat_ids]
    #cont_max = 0.1
    #ok_cont_dists = torch.where(
    #    lowdim_dists<cont_max,
    #    lowdim_dists, torch.zeros_like(lowdim_dists))
    #ok_cont_ids = torch.nonzero(ok_cont_dists)
    #if len(ok_cont_ids)==0: return {}, {}
    #lowdim_dists = lowdim_dists[ok_cont_ids]
    #latent_dists = latent_dists[ok_cont_ids]
    eps = 1e-10
    ratios = (latent_dists+eps)/(lowdim_dists+eps)
    log_ratios = torch.log(ratios)
    debug_dict = {'anlz_'+dbg_pfx+'irreg_std': log_ratios.std().item()}
    debug_hist_dict = {
        'anlz_'+dbg_pfx+'irreg_ratios': log_ratios,
        'anlz_'+dbg_pfx+'irreg_dists': lowdim_dists }
    print('analyze_irregularity {:0.4f}'.format(log_ratios.std().item()))
    return debug_dict, debug_hist_dict


def analyze_alignment(train_latents, train_lowdims, test_latents, test_lowdims,
                      test_curr_latents, test_curr_lowdims, lowdim_nms, args):
    print('train_latents\t', train_latents.size(),
          '\ttrain_lowdims\t', train_lowdims.size())
    print('test_latents\t', test_latents.size(),
          '\ttest_lowdims\t', test_lowdims.size())
    if test_curr_latents is not None:
        print('test_curr_latents\t', test_curr_latents.size(),
              '\ttest_curr_lowdims\t', test_curr_lowdims.size())
        assert(test_curr_lowdims.dim()==2)
    print('lowdim_nms', lowdim_nms)
    assert(train_latents.dim()==train_lowdims.dim()==2)
    assert(len(lowdim_nms)==train_lowdims.shape[1])
    # Whiten lowdims for easier fit.
    # This is used for Aux*BulletEnvs but not for Rearrange or Incline envs.
    # Rearrange envs are already nomalized to [-1,1] range. More importantly:
    # we need to interpret some of the dimensions as rotation matrices, so
    # would need to un-normalize those dimensions anyways.
    if 'Rearrange' not in args.env_name and 'Incline' not in args.env_name:
        trmean = train_lowdims.mean(dim=0, keepdim=True)
        trstd = train_lowdims.std(dim=0, keepdim=True)+1e-6  # avoid div by 0
        train_lowdims = (train_lowdims-trmean)/trstd
        test_lowdims = (test_lowdims-trmean)/trstd
        assert(torch.isfinite(train_lowdims).all())
        assert(torch.isfinite(test_lowdims).all())
        if test_curr_lowdims is not None:
            test_curr_lowdims = (test_curr_lowdims-trmean)/trstd
            assert(torch.isfinite(test_curr_lowdims).all())
    debug_dict = {}; debug_dict_hist = {}
    dct, dct_hist = analyze_with_nns(
        train_latents, train_lowdims, test_latents, test_lowdims,
        test_curr_latents, test_curr_lowdims, lowdim_nms, args)
    debug_dict.update(dct); debug_dict_hist.update(dct_hist)
    dct, dct_hist = analyze_irregularity(
        train_latents, train_lowdims, lowdim_nms, 'train_')
    debug_dict.update(dct); debug_dict_hist.update(dct_hist)
    dct, dct_hist = analyze_irregularity(
        test_latents, test_lowdims, lowdim_nms, 'test_')
    debug_dict.update(dct); debug_dict_hist.update(dct_hist)
    return debug_dict, debug_dict_hist
