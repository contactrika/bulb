"""
Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""

import torch
import torch.nn as nn
import torch.optim as optim

import torch.distributed as dist


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.frac_train_remaining = 0  # updated externally

    def compute_loss(self, sample):
        obs_batch, recurrent_hidden_states_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, \
            old_action_log_probs_batch, adv_targ = sample

        values, action_log_probs, dist_entropy, _ = \
            self.actor_critic.evaluate_actions(
                obs_batch, recurrent_hidden_states_batch, masks_batch,
                actions_batch)

        ratio = torch.exp(action_log_probs -
                          old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        cliprange = self.clip_param * self.frac_train_remaining
        surr2 = torch.clamp(ratio, 1.0 - cliprange,
                            1.0 + cliprange) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + \
                (values - value_preds_batch).clamp(-cliprange, cliprange)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (
                value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses,
                                         value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        # NOTE: in OpenAI PPO w/ MPI:
        # params = tf.trainable_variables()
        # weight_params = [v for v in params if '/b' not in v.name]
        # l2_loss = tf.reduce_sum([tf.nn.l2_loss(v) for v in weight_params])
        # But their default Config.L2_WEIGHT is 0.0
        # loss = (pg_loss - entropy * ent_coef + vf_loss * vf_coef
        #         + l2_loss * Config.L2_WEIGHT)
        rl_loss = (value_loss*self.value_loss_coef + action_loss
                   -dist_entropy*self.entropy_coef)

        return rl_loss
