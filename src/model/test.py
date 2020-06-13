#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque

from tqdm import tqdm, auto

import torch
import torch.nn.functional as F

from model.pnn import PNN
from model.env import create_env

from common.utils import gen_stats, get_threshold


def test(pid, opt, gmodel, lock):
    torch.manual_seed(opt.seed + pid)
    allenvs = create_env(opt)
    lmodel = PNN(opt.model_type)
    lmodel.eval()

    for env in allenvs:
        auto.tqdm.write(env.unwrapped.spec.id)
        env.seed(opt.seed + pid)
        # TODO: might need to be changed for non ram version
        if opt.model_type == 'linear':
            lmodel.add(sizes=[
                env.observation_space.shape[0], 64, 32, 16, env.action_space.n
            ])
        elif opt.model_type == 'conv':
            lmodel.add(nchannels=env.observation_space.shape[-1],
                       nactions=env.action_space.n)

    for env in allenvs:
        state = torch.Tensor(env.reset())
        done = True
        step, rewards = 0, []
        actions = deque(maxlen=opt.max_actions)

        iterator = tqdm(range(int(opt.ngsteps)))
        for episode in iterator:
            step += 1
            if done:
                lmodel.load_state_dict(gmodel.state_dict())

            with torch.no_grad():
                _, logits = lmodel(state.permute(2, 0, 1).unsqueeze(0))
            prob = F.softmax(logits, dim=1)
            action = torch.argmax(prob).item()

            state, reward, done, _ = env.step(action)
            state = torch.Tensor(state)
            rewards.append(reward)
            if opt.render:
                env.render()
            actions.append(action)
            if step > opt.ngsteps or \
               actions.count(actions[0]) == actions.maxlen:
                done = True

            rmin, rmax, rmean, rmedian = gen_stats(rewards)
            progress_data = f"min/max/mean/median reward: {rmin:5.1f}/{rmax:5.1f}/{rmean:5.1f}/{rmedian:5.1f}"
            iterator.set_postfix_str(progress_data)

            if done:
                step = 0
                rewards = []
                actions.clear()
                state = torch.Tensor(env.reset())

                # time to move on
                if rmean > get_threshold(env.unwrapped.spec.id):
                    with lock:
                        gmodel.freeze()
                    break
