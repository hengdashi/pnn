#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil

from tqdm import auto

import torch
import torch.multiprocessing as mp

from model.pnn import PNN
from model.optimizer import GlobalAdam
from model.train import train
from model.test import test
from model.env import create_env
from common.params import Parameters

if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    opt = Parameters()

    if opt.log_path.exists() and opt.log_path.is_dir():
        shutil.rmtree(opt.log_path)
    opt.log_path.mkdir(parents=True)
    if not opt.save_path.is_dir():
        opt.save_path.mkdir(parents=True)

    # for evaluation use
    allenvs = create_env(opt)
    gmodel = PNN(opt.model_type)
    for env in allenvs:
        #  print(env.observation_space)
        if opt.model_type == 'linear':
            gmodel.add(sizes=[
                env.observation_space.shape[0], 64, 32, 16, env.action_space.n
            ])
        elif opt.model_type == 'conv':
            gmodel.add(nchannels=env.observation_space.shape[0],
                       nactions=env.action_space.n)

    gmodel.share_memory()
    if opt.load:
        file = opt.save_path / "pnn"
        if file.exists() and file.is_file():
            gmodel.load_state_dict(torch.load(file))

    auto.tqdm.write(f"{vars(opt)}")

    optimizer = GlobalAdam(gmodel.parameters(), lr=opt.lr)

    lock = mp.Lock()
    processes = []

    process = mp.Process(target=test, args=(opt.nprocesses, opt, gmodel, lock))
    process.start()
    processes.append(process)

    for pid in range(opt.nprocesses):
        process = mp.Process(target=train,
                             args=(pid, opt, gmodel, optimizer,
                                   True if not pid else False))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
