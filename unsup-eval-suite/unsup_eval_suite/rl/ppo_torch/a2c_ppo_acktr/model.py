"""
Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distributions import Bernoulli, Categorical, DiagGaussian
from .utils import init

from .distributions import FixedNormal
from .utils import AddBias


class DiagGaussianShell(nn.Module):
    def __init__(self, num_outputs):
        super(DiagGaussianShell, self).__init__()
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, action_mean):
        zeros = torch.zeros_like(action_mean)
        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBaseLong # MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)
        #self.base = base(obs_shape[0], action_space.shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        #dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = (recurrent is not None)

        if self._recurrent:
            self.use_lstm = recurrent=='LSTM'
            recurrent_nn_class = eval('nn.'+recurrent)
            self.rnn = recurrent_nn_class(
                recurrent_input_size, hidden_size,
                num_layers=1, bias=True, batch_first=False)
            for name, param in self.rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size*2 if self.use_lstm else self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_rnn(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            if self.use_lstm:
                hcxs = (hxs*masks).unsqueeze(0)
                h = hcxs[:,:,:self._hidden_size].contiguous()
                c = hcxs[:,:,self._hidden_size:].contiguous()
                x, hcxs = self.rnn(x.unsqueeze(0), (h,c))
                hxs = torch.cat([hcxs[0], hcxs[1]], dim=2)
            else:
                x, hxs = self.rnn(x.unsqueeze(0), (hxs*masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                if self.use_lstm:
                    hcxs = hxs*masks[start_idx].view(1, -1, 1)
                    h = hcxs[:,:,:self._hidden_size].contiguous()
                    c = hcxs[:,:,self._hidden_size:].contiguous()
                    rnn_scores, hcxs = self.rnn(x[start_idx:end_idx], (h,c))
                    hxs = torch.cat([hcxs[0], hcxs[1]], dim=2)
                else:
                    rnn_scores, hxs = self.rnn(
                        x[start_idx:end_idx],
                        hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


#
# NOTE: NN architecture for image-only domains (e.g. RGB 64x64).
#
class CNNBase64(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512,
                 float_inputs=True):
        super(CNNBase64, self).__init__(recurrent, hidden_size, hidden_size)
        self.float_inputs = float_inputs
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(1024, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        if not self.float_inputs: inputs = inputs/255.0
        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)
        nl = nn.Tanh() # nn.ELU()

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nl,
            init_(nn.Linear(hidden_size, hidden_size)), nl)

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nl,
            init_(nn.Linear(hidden_size, hidden_size)), nl)

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class MLPBaseLong(NNBase):
    def __init__(self, num_inputs, num_act_outputs, recurrent=False, hidden_size=64):
        super(MLPBaseLong, self).__init__(recurrent, num_inputs, hidden_size)
        nl = nn.Tanh()

        if recurrent:
            num_inputs = hidden_size

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nl,
            nn.Linear(hidden_size, hidden_size), nl,
            nn.Linear(hidden_size, num_act_outputs))

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nl,
            nn.Linear(hidden_size, hidden_size), nl,
            nn.Linear(hidden_size, 1))

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        self.actor.apply(init_weights)
        self.critic.apply(init_weights)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        return self.critic(x), self.actor(x), rnn_hxs


class MLPBaseLongTwin(NNBase):
    def __init__(self, num_inputs, num_act_outputs, recurrent=False, hidden_size=64):
        super(MLPBaseLongTwin, self).__init__(recurrent, num_inputs, hidden_size)
        nl = nn.Tanh()

        if recurrent:
            num_inputs = hidden_size

        #self.actor_base = nn.Sequential(
        #    nn.Linear(num_inputs, hidden_size), nl,
        #    nn.Linear(hidden_size, hidden_size), nl)
        #self.actor = nn.Sequential(
        #    nn.Linear(hidden_size, num_act_outputs))
        #self.twin_actor = nn.Sequential(
        #    nn.Linear(hidden_size, num_act_outputs))

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nl,
            nn.Linear(hidden_size, hidden_size), nl,
            nn.Linear(hidden_size, num_act_outputs))

        self.twin_actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nl,
            nn.Linear(hidden_size, hidden_size), nl,
            nn.Linear(hidden_size, num_act_outputs))

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nl,
            nn.Linear(hidden_size, hidden_size), nl,
            nn.Linear(hidden_size, 1))

        self.twin_critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nl,
            nn.Linear(hidden_size, hidden_size), nl,
            nn.Linear(hidden_size, 1))

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)
        #self.actor_base.apply(init_weights)
        self.actor.apply(init_weights)
        self.twin_actor.apply(init_weights)
        self.critic.apply(init_weights)
        self.twin_critic.apply(init_weights)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_rnn(x, rnn_hxs, masks)

        # TODO: check that this passes only grads of net parts actually used.
        # TODO: generalize later.
        #act_base = self.actor_base(x)
        phi_tl_minus_tgt = (x[:,0]-x[:,5])
        phi_tl_dot = x[:,1]
        # Determine whether the tool is moving away from tgt.
        away = torch.abs(phi_tl_minus_tgt+phi_tl_dot)-torch.abs(phi_tl_minus_tgt)
        away = away.view(-1, 1)
        act_mean = torch.where(away>0, self.twin_actor(x), self.actor(x))
        crit = torch.where(away>0, self.twin_critic(x), self.critic(x))

        return crit, act_mean, rnn_hxs
