#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
from pprint import pformat

from tqdm import auto

import numpy as np

import torch
import torch.multiprocessing as mp

from model.pnn import PNN
from model.test import test
from model.train import train
from model.optimizer import GlobalAdam

from common.env import create_env
from common.params import Parameters

if __name__ == "__main__":
    # need to use spawn method
    # otherwise would be super slow on linux server
    # since the default method is fork
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # load parameters
    opt = Parameters()

    if opt.log_path.exists() and opt.log_path.is_dir():
        shutil.rmtree(opt.log_path)
    opt.log_path.mkdir(parents=True)
    if not opt.save_path.is_dir():
        opt.save_path.mkdir(parents=True)

    # for evaluation use
    allenvs = create_env(opt)
    # keep track of current column
    current = mp.Value('i', 0)
    gmodel = PNN(allenvs, current)

    gmodel.share_memory()
    if opt.load:
        file = opt.save_path / "pnn"
        if file.exists() and file.is_file():
            gmodel.load_state_dict(torch.load(file))

    auto.tqdm.write(f"{pformat(vars(opt), indent=2)}")

    optimizer = GlobalAdam(gmodel.parameters(), lr=opt.lr)

    lock = mp.Lock()
    processes = []

    # spawning training and evaluation model
    #  mp.spawn(fn=train,
    #  args=(opt, current, gmodel, optimizer, lock),
    #  nprocs=opt.nprocesses)
    #  mp.spawn(fn=test, args=(opt, gmodel, lock))

    for pid in range(opt.nprocesses):
        process = mp.Process(target=train,
                             args=(pid, opt, current, gmodel, optimizer, lock))
        process.start()
        processes.append(process)

    process = mp.Process(target=test, args=(opt.nprocesses, opt, gmodel, lock))
    process.start()
    processes.append(process)

    for process in processes:
        process.join()
