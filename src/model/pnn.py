#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
                nn.Parameter(torch.randn(1, dtype=float))
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


class PNN(nn.Module):
    # nlayers is the number of layers in one column
    def __init__(self, nlayers):
        super(PNN, self).__init__()
        self.nlayers = nlayers
        self.columns = nn.ModuleList()

    def forward(self, X, cid=-1):
        # first layer pass
        h = [column[0](X) for column in self.columns]
        # rest layers pass
        for k in range(1, self.nlayers):
            h = [column[k](h[:i + 1]) for i, column in enumerate(self.columns)]

        # return latest output unless specified
        return h[cid]

    # sizes contains a list of layers' output size
    # add a column to the neural net
    def add(self, sizes):
        modules = [
            PNNLinear(lid, len(self.columns), sizes[lid], sizes[lid + 1])
            for lid in range(self.nlayers)
        ]
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
