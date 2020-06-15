#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PNNColumn(nn.Module):
    def __init__(self, cid, nchannels, nactions):
        super(PNNColumn, self).__init__()
        self.cid = cid
        self.nchannels = nchannels
        self.nactions = nactions
        # 6 layers neural network
        self.nlayers = 6
        self.nhidden = 256

        # init normal nn, lateral connection, adapter layer and alpha
        self.w = nn.ModuleList()
        self.u = nn.ModuleList()
        self.v = nn.ModuleList()
        self.alpha = nn.ModuleList()

        # normal neural network
        self.w.append(
            nn.Conv2d(nchannels, 32, kernel_size=3, stride=2, padding=1))
        self.w.extend([
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
            for _ in range(3)
        ])
        conv_out_size = self._get_conv_out((nchannels, 84, 84))
        self.w.append(nn.Linear(conv_out_size, self.nhidden))
        # w[-2] is the critic layer and w[-1] is the actor layer
        self.w.append(
            nn.ModuleList([
                nn.Linear(self.nhidden, 1),
                nn.Linear(self.nhidden, nactions)
            ]))

        # only add lateral connections and adapter layers if not first column
        # v[col][layer][(nnList on that layer)]
        for i in range(self.cid):
            self.v.append(nn.ModuleList())
            # adapter layer
            self.v[i].append(nn.Identity())
            self.v[i].extend(
                [nn.Conv2d(32, 1, kernel_size=1) for _ in range(3)])
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

            # lateral connection
            self.u.append(nn.ModuleList())
            self.u[i].append(nn.Identity())
            self.u[i].extend([
                nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
                for _ in range(3)
            ])
            self.u[i].append(nn.Linear(conv_out_size, self.nhidden))
            self.u[i].append(
                nn.ModuleList([
                    nn.Linear(self.nhidden, 1),
                    nn.Linear(self.nhidden, self.nactions)
                ]))

        # init weights
        self._reset_parameters()
        self.w[-1][0].weight.data = self._normalized(self.w[-1][0].weight.data)
        self.w[-1][1].weight.data = self._normalized(self.w[-1][1].weight.data,
                                                     1e-2)

        for i in range(self.cid):
            self.u[i][-1][0].weight.data = self._normalized(
                self.u[i][-1][0].weight.data)
            self.u[i][-1][1].weight.data = self._normalized(
                self.u[i][-1][1].weight.data, 1e-2)

    def forward(self, x, pre_out):
        """feed forward process for a single column"""
        # put a placeholder to occupy the first layer spot
        next_out, w_out = [torch.zeros(x.shape)], x

        # pass input layer by layer
        critic_out, actor_out = None, None
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
        critic_out = self.w[-1][0](w_out)
        pre_critic_out = [
            self.u[k][-1][0](pre_out[k][self.nlayers - 1])
            if self.cid else torch.zeros(critic_out.shape)
            for k in range(self.cid)
        ]
        actor_out = self.w[-1][1](w_out)
        pre_actor_out = [
            self.u[k][-1][1](pre_out[k][self.nlayers - 1])
            if self.cid else torch.zeros(actor_out.shape)
            for k in range(self.cid)
        ]

        # TODO: do we need information from previous columns or not?
        return critic_out + sum(pre_critic_out), \
               actor_out + sum(pre_actor_out), \
               next_out
        #  return critic_out, actor_out, next_out

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
    def __init__(self, allenvs):
        super(PNN, self).__init__()
        # current column index
        self.current = 0
        # complete network
        self.columns = nn.ModuleList()

        for i, env in enumerate(allenvs):
            nchannels = env.observation_space.shape[0]
            nactions = env.action_space.n
            self.columns.append(
                PNNColumn(len(self.columns), nchannels, nactions))
            # freeze parameters that is not on first column
            if i != 0:
                for params in self.columns[i].parameters():
                    params.requires_grad = False

    def forward(self, X):
        """
        PNN forwarding method
        X is the state of the current environment being trained
        """
        h_actor, h_critic, next_out = None, None, []

        for i in range(self.current + 1):
            h_critic, h_actor, out = self.columns[i](X, next_out)
            next_out.append(out)

        return h_critic, h_actor

    def freeze(self):
        """freeze previous columns"""
        self.current += 1

        # freeze previous columns
        for i in range(self.current + 1):
            for params in self.columns[i].parameters():
                params.requires_grad = False

        # enable grad on next column
        for params in self.columns[self.current].parameters():
            params.requires_grad = True
        # diable those constant parameters
        for i in self.current:
            self.columns[self.current].alpha[i][0].requires_grad = False
            self.columns[self.current].alpha[i][-1].requires_grad = False

    def parameters(self, cid=None):
        """return parameters of the current column"""
        if cid is None:
            return super(PNN, self).parameters()
        return self.columns[cid].parameters()
