"""
Params for SVAE.
"""
import argparse
import logging
import sys

import torch.nn as nn


def get_parser():
    parser = argparse.ArgumentParser(description="SVAE", add_help=False)
    parser.add_argument('--svae_noconv', action='store_true',
                        help='Whether to use MLPs for all nets')
    parser.add_argument('--svae_novi', action='store_true',
                        help='Whether to drop KL')
    parser.add_argument('--svae_nolstm', action='store_true',
                        help='Use GRU for RNN instead of LSTM')
    parser.add_argument('--svae_kl_beta', type=float, default=1.0,
                        help='ELBO KL beta')
    parser.add_argument('--svae_decoder_extra_epochs', type=int, default=0,
                        help='Number of extra epochs to pretrain decoder')
    return parser


class SVAEParams():
    def __init__(self, hidden_size=512, static_size=8, dynamic_size=32,
                 hist=16, past=4, pred=8, logvar_limit=6, mu_nl=nn.Sigmoid(),
                 conv_nflt=64, ulo_static_size=0, debug=False):
        self.clr_chn = 3                 # number of color channels (1 or 3)
        self.obj_sz = 28
        self.knl_sz = 4                    # conv kernel size
        self.strd_sz = int(self.knl_sz/2)  # conv stride size
        self.pd_sz = int(self.strd_sz/2)   # conv padding size
        self.conv_nfilters = conv_nflt     # number of conv filter
        self.comp_out_sz = 128           # size of inpt stack output (e.g. conv)
        self.hidden_sz = hidden_size     # hidden layers for all nets
        self.ulo_static_sz = ulo_static_size
        self.static_sz = static_size     # size of f in q(z_{1:T}, f | x_{1:T})
        self.dynamic_sz = dynamic_size   # size of z in q(z_{1:T}, f | x_{1:T})
        self.hist = hist; self.past = past; self.pred = pred
        assert(hist==0 or hist>=past)
        # ReLU does not have hyperparameters, works with dropout and batchnorm.
        # Other options like ELU/SELU are more suitable for very deep nets
        # and have shown some promise, but no huge gains.
        # With partialVAE ReLUs will cause variance to explode on high-dim
        # inputs like pixels from image.
        # Tanh can be useful when the range needs to be restricted,
        # but saturates and trains slower.
        # ELU showed better results for high learning rates on RL experiments.
        self.nl = nn.ELU()
        # Control latent space range.
        self.mu_nl = mu_nl
        # Stabilize training by clamping logvar outputs.
        # sqrt(exp(-6)) ~= 0.05 so 6: std min=0.05 max=20.0
        # 10: std min=0.0067 max=148.4
        logvar_limit = logvar_limit
        self.logvar_nl = nn.Hardtanh(-logvar_limit, logvar_limit)
        self.debug = debug

#                                   hid    st  dyn  hist past pred
PARAMS_VAE_REARR_ONE_LG = SVAEParams(512,  0,    32,   1,  1,  0, 20, None, 64, 0)
PARAMS_VAE_REARR_ONE    = SVAEParams(512,  0,    17,   1,  1,  0, 20, None, 64, 0)
PARAMS_PRED_REARR_ONE   = SVAEParams(512,  1,    16,   1,  1,  1, 20, None, 64, 0)
PARAMS_VAE_REARR_LG     = SVAEParams(512,  0,   106,   1,  1,  0, 20, None, 64, 0)
PARAMS_VAE_REARR        = SVAEParams(512,  0,    53,   1,  1,  0, 20, None, 64, 0)
PARAMS_PRED_REARR       = SVAEParams(512,  1,    52,   1,  1,  1, 20, None, 64, 0)
PARAMS_VAE_BLK_SM       = SVAEParams(512,  0,     5,   1,  1,  0, 20, None, 32, 1)
PARAMS_PRED_BLK_SM      = SVAEParams(512,  3,     2,   1,  1,  1, 20, None, 32, 1)
PARAMS_VAE              = SVAEParams(512,  0,  None,   1,  1,  0)
PARAMS_PRED_SEQ2        = SVAEParams(512,  0,  None,   1,  1,  1)
PARAMS_SVAE_SEQ8        = SVAEParams(512,  0,  None,   8,  8,  0)
PARAMS_PRED_SEQ8        = SVAEParams(512,  0,  None,   2,  2,  6)
PARAMS_DSA_SEQ8         = SVAEParams(512, None, None,  8,  8,  0)
PARAMS_SVAE_SEQ16       = SVAEParams(512,  0,  None,  16, 16,  0)
PARAMS_PRED_SEQ16       = SVAEParams(512,  0,  None,   4,  4, 12)
PARAMS_DSA_SEQ16        = SVAEParams(512, None, None, 16, 16,  0)
PARAMS_SVAE_SEQ24       = SVAEParams(512,  0,  None,  24, 24,  0)
PARAMS_PRED_SEQ24       = SVAEParams(512,  0,  None,   8,  8, 16)
PARAMS_DSA_SEQ24        = SVAEParams(512, None, None, 24, 24,  0)
PARAMS_SVAE             = SVAEParams(512,  0,  None,  32, 32,  0)
PARAMS_PRED             = SVAEParams(512,  0,  None,   8,  8, 24)
PARAMS_DSA              = SVAEParams(512, None, None, 32, 32,  0)
