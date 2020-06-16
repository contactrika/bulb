"""
Utils for processing training and test data.
"""

import numpy as np

import torch


def extract_tgts(x_1toL, act_1toL, aux_1toL, hist, past, pred):
    assert(act_1toL.size(1)>=(past+pred))
    if x_1toL is not None: x_1toL = torch.unbind(x_1toL, dim=1)
    act_1toL = torch.unbind(act_1toL, dim=1)
    if aux_1toL is not None: aux_1toL = torch.unbind(aux_1toL, dim=1)
    if x_1toL is not None: x_1toT = torch.stack(x_1toL[0:hist], dim=1)
    act_1toT = torch.stack(act_1toL[0:hist], dim=1)
    if aux_1toL is not None: aux_1toT = torch.stack(aux_1toL[0:hist], dim=1)
    ofst = past+pred
    if x_1toL is not None: xs_tgt = torch.stack(x_1toL[-ofst:], dim=1)
    acts_tgt = torch.stack(act_1toL[-ofst:], dim=1)
    if aux_1toL is not None: auxs_tgt = torch.stack(aux_1toL[-ofst:], dim=1)
    if x_1toL is None:
        x_1toT = None; xs_tgt = None
    else:
        assert(xs_tgt.size(1) == past+pred)
    assert(acts_tgt.size(1) == past+pred)
    if aux_1toL is None:
        aux_1toT = None; auxs_tgt = None
    else:
        assert(auxs_tgt.size(1) == past+pred)
    return x_1toT, act_1toT, aux_1toT, xs_tgt, acts_tgt, auxs_tgt


#
# TODO: move code below to old.
#
def gaussian(x, sigma):
    c = 1.0/(sigma*np.sqrt(2*np.pi))
    res = c*torch.exp(-0.5*(x/sigma)**2)
    return res


def plot_clusters_2d(X, centers=None, cluster_smpl_ids=None):
    import matplotlib.pyplot as plt
    X = X.cpu().numpy()
    plt.scatter(X[:,0], X[:,1], alpha=0.5, s=10)
    if centers is not None:
        centers = centers.cpu().numpy()
        plt.scatter(centers[:,0], centers[:,1], marker='x', c='red', s=50)
    if cluster_smpl_ids is not None:
        cluster_smpls = X[cluster_smpl_ids]
        plt.scatter(cluster_smpls[:,0], cluster_smpls[:,1],
                    marker='.', c='green', s=10)
    plt.show()


# A basic meanshift algo to form clusters; a brief explanation by Jeremy Howard:
# https://youtu.be/bZmJvmxfH6I?t=4617
def meanshift(X_original, debug=False, debug_centers=None):
    X = X_original.clone()
    if debug: print('Starting meanshift X', X.size())
    n = X.size(0)
    niter = 10
    mbsz = 512    # minibatch size
    sigma = 0.25  # TODO: tune this
    for it in range(niter):
        for i in range(0,n,mbsz):
            s = slice(i,min(n,i+mbsz))
            dists = torch.cdist(X,X[s])
            weights = gaussian(dists, sigma).transpose(0,1)
            numer = torch.matmul(weights,X)
            denom = weights.sum(dim=1,keepdim=True)
            newXs = numer / denom
            diff = torch.abs(X[s]-newXs).mean()
            if diff<1e-4: break  # converged
            X[s] = newXs
            if debug and i==0:
                min_dists, _ = torch.topk(dists, k=n//10, largest=False)
                print('meanshift iter {:d} diff {:0.4f} min_dists {:0.4f}'.format(
                      it, diff, min_dists.mean().item()))
                plot_clusters_2d(X, debug_centers)
    return X


def sample_cluster(X, nsmpls, debug=False):
    # Pick a cluster center as a point with a large number of close neighbors.
    assert(nsmpls<X.size(0))  # can't return more pts than size of X
    if debug: print('sample_cluster()...')
    nmax = 200  # max num pts over which to compute torch.cdist
    cluster_smpl_ids = None
    num_iters = 20
    slim_ids = torch.randperm(X.size(0))[0:nmax]
    Xslim = X[slim_ids]
    dists = torch.cdist(Xslim,Xslim); dists_max = dists.max()
    if debug:
        str = 'nsmpls {:d} dists mean {:0.4f} max {:0.4f}'
        print(str.format(nsmpls, dists.mean(), dists_max))
    for i in range(num_iters):
        sigma = dists_max*float(i+1)/float(num_iters)
        slim_cnts = (dists<sigma).sum(dim=1); slim_cnts_max = slim_cnts.max()
        if debug:
            print('sigma {:0.4f} slim nsmpls {:d}'.format(sigma, slim_cnts_max))
        if slim_cnts_max<=1: continue
        center = Xslim[slim_cnts.argmax(),:]
        cluster_all_dists = torch.cdist(X,center.unsqueeze(0))
        cluster_all_dists = cluster_all_dists.squeeze()
        cluster_smpl_ids = (cluster_all_dists<sigma).nonzero().squeeze()
        cluster_dist_mean = cluster_all_dists[cluster_smpl_ids].mean()
        actual_nsmpls = cluster_smpl_ids.size(0)
        if debug:
            print(' cluster mean dist {:0.4f} actual_nsmpls {:d}'.format(
                cluster_dist_mean, actual_nsmpls))
        if actual_nsmpls>=nsmpls:
            cluster_smpl_ids = cluster_smpl_ids[:nsmpls]
            break  # done
    return cluster_smpl_ids, cluster_dist_mean, dists_max

def make_fake_data():
    npts_per_cluster = 100
    nclusters = 3
    centers = torch.rand(nclusters,2)*10
    sigma = 0.5
    X = []
    for i in range(nclusters):
        data = (torch.randn(npts_per_cluster,2))*sigma + centers[i]
        X.append(data)
    X = torch.cat(X, dim=0)
    return X, centers


def main():
    X, centers = make_fake_data()
    plot_clusters_2d(X, centers)
    #Xclst, cluster_smpl_ids = meanshift(X, debug=True, debug_centers=centers)
    cluster_smpl_ids, _, _ = sample_cluster(X, nsmpls=25, debug=True)
    plot_clusters_2d(X, centers, cluster_smpl_ids)


if __name__ == '__main__':
    main()
