#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path


class Parameters:
    def __init__(self):
        self.cwd = Path('.')
        # number of layers in PNN (default: 4)
        self.nlayers = 4
        # learning rate (default: 1e-3)
        self.lr = 1e-3
        # discount factor for rewards (default: 0.9)
        self.gamma = 0.9
        # parameter for GAE (default: 1.0)
        self.tau = 1.0
        # entropy coefficient (default: 0.01)
        self.beta = 0.01
        # number of local steps (default: 5)
        self.nlsteps = 5
        # number of global steps (default: 5e6)
        self.ngsteps = 5e6
        # number of processes (default: 6)
        self.nprocesses = 1
        # number of steps between saving (default: 500)
        self.interval = 500
        # maximum repetition steps in test phase (default: 200)
        self.max_actions = 200
        # logging path (default: tensorboard/pnn)
        self.log_path = self.cwd / "tensorboard" / "pnn"
        # saving path (default: trained_models)
        self.save_path = self.cwd / "trained_models"
        # load weight from previous trained stage (default: False)
        self.load = False
        # use gpu (default: False)
        self.gpu = False
        # whether to render the frames or not (default: False)
        self.render = True
        # seed (default: 123)
        self.seed = 123
