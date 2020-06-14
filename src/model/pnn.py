#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# basic construction block for PNN
class PNNLinear(nn.Module):
    def __init__(self, lid, cid, input_dim, output_dim):
        super(PNNLinear, self).__init__()
        self.lid = lid
        self.cid = cid
        self.input_dim = input_dim
        self.output_dim = output_dim

        # h^(k)_i = \sigma(W^(k)_i h^(k)_{i-1} + U^(k:j)_{i}
        #           \sigma(V^(k:j)_i \alpha^(<k)_{i-1} h^(<k)_{i-1}))

        # basic neural net
        self.w = nn.Linear(self.input_dim, self.output_dim)
        # lateral connection
        self.u = nn.ModuleList()
        # adapter
        self.v = nn.ModuleList()
        # alpha
        self.alpha = []

        # need lateral connection only if
        # 1. it is not the first column
        # 2. it is not the first layer
        if self.cid and self.lid:
            self.u.extend([
                nn.Linear(self.input_dim, self.output_dim)
                for _ in range(self.cid)
            ])
            self.v.extend([
                nn.Linear(self.input_dim, self.input_dim)
                for _ in range(self.cid)
            ])
            self.alpha.extend([
                nn.Parameter(torch.Tensor(np.random.choice([1e0, 1e-1, 1e-2])))
                for _ in range(self.cid)
            ])

    def forward(self, X):
        """feed forward for linear pnn"""
        X = [X] if not isinstance(X, list) else X
        # first part of the equation
        # current column output
        # use the latest input
        cur_out = self.w(X[-1])
        # second part of the equation
        # lateral connections
        # use old inputs from previous columns
        prev_out = [
            u(F.elu(v(alpha * x)))
            for u, v, alpha, x in zip(self.u, self.v, self.alpha, X)
        ]
        return F.elu(cur_out + sum(prev_out))


class PNNConv(nn.Module):
    def __init__(self, cid, nchannels, nactions):
        super(PNNConv, self).__init__()
        self.cid = cid
        self.nchannels = nchannels
        self.nactions = nactions
        # 6 layers neural network
        self.nlayers = 6

        # init normal nn, lateral connection, adapter layer and alpha
        self.w = nn.ModuleList()
        self.u = nn.ModuleList()
        self.v = nn.ModuleList()
        self.alpha = nn.ModuleList()
        self.u_actor = nn.ModuleList()
        self.u_critic = nn.ModuleList()

        # normal neural network
        self.w.append(
            nn.Conv2d(nchannels, 32, kernel_size=3, stride=2, padding=1))
        self.w.extend([
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            for _ in range(3)
        ])
        conv_out_size = self._get_conv_out((nchannels, 84, 84))
        self.w.append(nn.Linear(conv_out_size, 256))
        self.actor = nn.Linear(256, nactions)
        self.critic = nn.Linear(256, 1)

        # only add lateral connections and adapter layers if not first column
        # for each columns
        for i in range(self.cid):
            self.v.append(nn.ModuleList())
            # adapter layer
            self.v[i].append(nn.Identity())
            self.v[i].extend(
                [nn.Conv2d(32, 1, kernel_size=1) for _ in range(3)])
            self.v[i].append(nn.Identity())
            self.v[i].append(nn.Identity())

            # alpha
            self.alpha.append(nn.ParameterList())
            self.alpha[i].append(
                nn.Parameter(torch.Tensor(1), requires_grad=False))
            self.alpha[i].extend([
                nn.Parameter(
                    torch.Tensor(np.array(np.random.choice([1e0, 1e-1,
                                                            1e-2]))))
                for _ in range(3)
            ])
            self.alpha[i].append(
                nn.Parameter(torch.Tensor(1), requires_grad=False))
            self.alpha[i].append(
                nn.Parameter(torch.Tensor(1), requires_grad=False))

            # lateral connection
            self.u.append(nn.ModuleList())
            self.u[i].append(nn.Identity())
            self.u[i].extend([
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
                for _ in range(3)
            ])
            self.u[i].append(nn.Linear(conv_out_size, 256))
            self.u_actor.append(nn.Linear(256, self.nactions))
            self.u_critic.append(nn.Linear(256, 1))

        # init weights
        self._reset_parameters()
        self.actor.weight.data = self._normalized(self.actor.weight.data, 1e-2)
        self.critic.weight.data = self._normalized(self.critic.weight.data)

    def forward(self, x, pre_out):
        """feed forward process for a single column"""
        # put a placeholder to occupy the first layer spot
        next_out, w_out = [torch.zeros(x.shape)], x

        # pass input layer by layer
        for i in range(self.nlayers - 1):
            if i == self.nlayers - 2:
                w_out = w_out.view(w_out.size(0), -1)
                for k in range(self.cid):
                    pre_out[k][i] = pre_out[k][i].view(pre_out[k][i].size(0),
                                                       -1)
            # pass into normal network
            w_out = self.w[i](w_out)
            # u, alpha, v are only valid if cid is not zero
            # summing over for all networks from previous cols
            # u[k][i]: u network for ith layer kth column
            u_out = [
                self.u[k][i](F.relu(self.v[k][i](self.alpha[k][i] *
                                                 (pre_out[k][i]))))
                if self.cid and i else torch.zeros(w_out.shape)
                for k in range(self.cid)
            ]
            w_out = F.relu(w_out + sum(u_out))
            next_out.append(w_out)

        # last layer
        critic_out = self.critic(w_out)
        #  pre_critic_out = [
        #  self.u_critic[k](pre_out[k][self.nlayers - 1])
        #  if self.cid else torch.zeros(critic_out.shape)
        #  for k in range(self.cid)
        #  ]
        actor_out = self.actor(w_out)
        #  pre_actor_out = [
        #  self.u_actor[k](pre_out[k][self.nlayers - 1])
        #  if self.cid else torch.zeros(actor_out.shape)
        #  for k in range(self.cid)
        #  ]

        # TODO: do we need information from previous columns or not?
        #  return critic_out + F.relu(torch.tensor(sum(pre_critic_out)).clone().detach()), \
        #  actor_out + F.relu(torch.tensor(sum(pre_actor_out)).clone().detach()),\
        #  next_out
        return critic_out, actor_out, next_out

    def _normalized(self, weights, std=1.0):
        output = torch.randn(weights.size())
        output *= std / torch.sqrt(output.pow(2).sum(1, keepdim=True))
        return output

    def _get_conv_out(self, shape):
        output = torch.zeros(1, *shape)
        for i in range(self.nlayers - 2):
            output = self.w[i](output)
        return int(np.prod(output.size()))

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)


class PNN(nn.Module):
    """Progressive Neural Network"""
    def __init__(self, model_type='linear'):
        super(PNN, self).__init__()
        self.cid = 0
        self.model_type = model_type
        # complete network
        self.columns = nn.ModuleList()

    def forward(self, X):
        """
        PNN forwarding method
        X is the state of the current environment being trained
        """
        h_actor, h_critic, next_out = None, None, []

        if self.model_type == 'linear':
            # first layer pass
            h = [self.columns[i][0](X) for i in range(self.cid)]
            # rest layers pass till last layer
            for k in range(1, self.nlayers - 1):
                h = [self.columns[i][k](h[:i + 1]) for i in range(self.cid)]
            # last layer
            h_actor = self.columns[self.cid - 1][-2](h)
            h_critic = self.columns[self.cid - 1][-1](h)
        elif self.model_type == 'conv':
            for i in range(self.cid):
                h_critic, h_actor, out = self.columns[i](X, next_out)
                next_out.append(out)

        return h_critic, h_actor

    def add(self, nchannels=None, nactions=None, sizes=None):
        """
        add a column to pnn
        sizes contains a list of layers' output size
        """
        self.cid += 1 if not self.columns else 0

        if self.model_type == 'linear':
            modules = [
                PNNLinear(lid, len(self.columns), sizes[lid], sizes[lid + 1])
                for lid in range(self.nlayers)
            ]
            # adding critic layer
            modules.append(
                PNNLinear(self.nlayers - 1, len(self.columns),
                          sizes[self.nlayers - 1], 1))
            self.columns.append(nn.ModuleList(modules))
        elif self.model_type == 'conv':
            self.columns.append(PNNConv(len(self.columns), nchannels,
                                        nactions))

    def freeze(self):
        """freeze previous columns"""
        for column in self.columns:
            for params in column.parameters():
                params.requires_grad = False

        self.cid += 1

    def parameters(self):
        """return parameters of the current column"""
        if not self.cid:
            return super(PNN, self).parameters()
        else:
            return self.columns[self.cid - 1].parameters()
