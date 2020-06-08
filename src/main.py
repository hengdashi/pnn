#!/usr/bin/env python
# -*- coding: utf-8 -*-
import shutil

import torch
import torch.multiprocessing as _mp

from model.pnn import PNN
from model.optimizer import GlobalAdam
from model.process import ltrain, ltest
from common.params import Parameters
from model.env import create_env

if __name__ == "__main__":
    opt = Parameters()
    opt.gpu = opt.gpu and torch.cuda.is_available()
    opt.device = torch.device("cuda" if opt.gpu else "cpu")
    print("Using GPU" if opt.gpu else "Using CPU")

    if opt.log_path.exists() and opt.log_path.is_dir():
        shutil.rmtree(opt.log_path)
    opt.log_path.mkdir(parents=True)
    if not opt.save_path.is_dir():
        opt.save_path.mkdir(parents=True)

    # init multiprocessing module
    mp = _mp.get_context("spawn")
    allenvs = create_env()
    gmodel = PNN(nlayers=opt.nlayers)
    # TODO: try one environment for now
    gmodel.add([
        allenvs[0].observation_space.shape[0], 64, 32, 16,
        allenvs[0].action_space.n
    ])

    gmodel.share_memory()
    if opt.load:
        file = opt.save_path / "pnn"
        if file.exists() and file.is_file():
            gmodel.load_state_dict(torch.load(file))

    optimizer = GlobalAdam(gmodel.parameters(), lr=opt.lr)
    processes = []
    for pid in range(opt.nprocesses):
        if pid == 0:
            process = mp.Process(target=ltrain,
                                 args=(pid, opt, gmodel, optimizer, True))
        else:
            process = mp.Process(target=ltrain,
                                 args=(pid, opt, gmodel, optimizer))
        process.start()
        processes.append(process)
    process = mp.Process(target=ltest, args=(opt.nprocesses, opt, gmodel))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()
