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
        X = [X] if not isinstance(X, list) else X
        # first part of the equation
        # current column output
        # use the latest input
        cur_out = self.w(X[-1])
        # second part of the equation
        # lateral connections
        # use old inputs from previous columns
        prev_out = sum([
            u(F.relu(v(alpha * x)))
            for u, v, alpha, x in zip(self.u, self.v, self.alpha, X)
        ])
        return F.relu(cur_out + prev_out)


class PNNConv(nn.Module):
    def __init__(self, cid, input_channel, output_dim):
        super(PNNConv, self).__init__()

        self.cid = cid
        self.w = nn.ModuleList()
        self.u = nn.ModuleList()
        self.v = nn.ModuleList()
        self.alpha = []

        self.w.append(
            nn.Conv2d(input_channel, 12, kernel_size=(8, 8), stride=(4, 4)))
        self.w.append(nn.Conv2d(12, 12, kernel_size=4, stride=2))
        self.w.append(nn.Conv2d(12, 12, kernel_size=(3, 4)))
        conv_out_size = int(
            np.prod(self._get_conv_out((input_channel, 210, 160))))
        self.w.append(nn.Linear(conv_out_size, output_dim))
        self.w.append(nn.Linear(conv_out_size, 1))

        if self.cid:
            # adapter layer
            self.v.append(None)
            self.v.append(nn.Conv2d(12, 1, kernel_size=1))
            self.v.append(nn.Conv2d(12, 1, kernel_size=1))
            self.v.append(nn.Conv2d(12, 1, kernel_size=1))

            # alpha
            self.alpha.append(None)
            self.alpha.append(
                nn.Parameter(torch.Tensor(np.random.choice([1e0, 1e-1,
                                                            1e-2]))))
            self.alpha.append(
                nn.Parameter(torch.Tensor(np.random.choice([1e0, 1e-1,
                                                            1e-2]))))
            self.alpha.append(
                nn.Parameter(torch.Tensor(np.random.choice([1e0, 1e-1,
                                                            1e-2]))))

            # lateral connection
            self.u.append(None)
            self.u.append(nn.Conv2d(1, 12, kernel_size=4, stride=2))
            self.u.append(nn.Conv2d(1, 12, kernel_size=(3, 4)))
            self.u.append(nn.Linear(conv_out_size, output_dim))
            self.u.append(nn.Linear(conv_out_size, 1))

    def _get_conv_out(self, shape, cid=3):
        o = self.w[0](torch.zeros(1, *shape))
        if cid == 2 or cid == 3:
            o = self.w[1](o)
        if cid == 3:
            o = self.w[2](o)
        return o.size()


class PNN(nn.Module):
    # nlayers is the number of layers in one column
    def __init__(self, nlayers):
        super(PNN, self).__init__()
        self.nlayers = nlayers
        self.columns = nn.ModuleList()

    def forward(self, X):
        # first layer pass
        h = [column[0](X) for column in self.columns]
        # rest layers pass till last layer
        for k in range(1, self.nlayers - 1):
            h = [column[k](h[:i + 1]) for i, column in enumerate(self.columns)]
        h_list = [column[self.nlayers - 1](h) for column in self.columns]

        h_actor = h_list[-1]
        h_critic = self.columns[len(self.columns) - 1][self.nlayers](h)

        # return latest output unless specified
        return h_actor, h_critic, h_list[:-1]

    # sizes contains a list of layers' output size
    # add a column to the neural net
    def add(self, sizes):
        modules = [
            PNNLinear(lid, len(self.columns), sizes[lid], sizes[lid + 1])
            for lid in range(self.nlayers)
        ]
        # adding critic layer
        modules.append(
            PNNLinear(self.nlayers - 1, len(self.columns),
                      sizes[self.nlayers - 1], 1))
        self.columns.append(nn.ModuleList(modules))

    # freeze previous columns
    def freeze(self):
        for column in self.columns:
            for params in column.parameters():
                params.requires_grad = False

    # return parameters of the current column
    def parameters(self, cid=None):
        return super(PNN, self).parameters(
        ) if cid is None else self.columns[cid].parameters()
