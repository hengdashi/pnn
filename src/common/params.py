#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path


class Parameters:
    def __init__(self):
        self.cwd = Path(__file__).absolute().parents[2]
        # number of layers in PNN (default: 4)
        self.nlayers = 4
        # learning rate (default: 1e-4)
        self.lr = 1e-3
        # discount factor for rewards (default: 0.99)
        self.gamma = 0.99
        # parameter for GAE (default: 1.0)
        self.tau = 1.0
        # entropy coefficient (default: 1e-2)
        self.beta = 1e-2
        # critic loss coef (default: 0.5)
        self.critic_loss_coef = 0.5
        # max grad norm (default: 40)
        self.clip = 40
        # number of local steps (default: 4)
        self.nlsteps = 20
        # number of global steps (default: 1e7)
        self.ngsteps = 1e7
        # number of processes (default: 6)
        self.nprocesses = 16
        # number of steps between saving (default: 500)
        self.interval = 500
        # maximum repetition steps in test phase (default: 100)
        self.max_actions = 100
        # logging path (default: tensorboard/pnn)
        self.log_path = self.cwd / "tensorboard" / "pnn"
        # saving path (default: trained_models)
        self.save_path = self.cwd / "trained_models"
        # load weight from previous trained stage (default: False)
        self.load = False
        # use gpu (default: False)
        self.gpu = False
        # whether to render the frames or not (default: False)
        self.render = False
        # seed (default: 123)
        self.seed = 123
