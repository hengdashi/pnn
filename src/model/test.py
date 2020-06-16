#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque

from tqdm import tqdm, auto

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model.pnn import PNN

from common.env import create_env
from common.utils import get_threshold


def test(pid, opt, gmodel, lock):
    torch.manual_seed(opt.seed + pid)
    writer = SummaryWriter(opt.log_path)
    allenvs = create_env(opt)
    lmodel = PNN(allenvs)

    # turn to eval mode
    lmodel.eval()

    for eid, env in enumerate(allenvs):
        env.seed(opt.seed + pid)
        state = torch.Tensor(env.reset())
        done = True
        step, reward_sum, weighted = 0, 0, 0
        actions = deque(maxlen=opt.max_actions)

        iterator = tqdm(range(int(opt.ngsteps)),
                        desc=env.unwrapped.spec.id,
                        unit='episode')
        for episode in iterator:
            step += 1
            if done:
                lmodel.load_state_dict(gmodel.state_dict())
                if not episode % opt.interval and episode:
                    torch.save(lmodel.state_dict(), opt.save_path / "pnn")

            with torch.no_grad():
                _, logits = lmodel(state.unsqueeze(0))
            prob = F.softmax(logits, dim=-1)
            action = prob.max(1, keepdim=True)[1].numpy()

            state, reward, done, _ = env.step(action[0, 0])
            state = torch.Tensor(state)
            reward_sum += reward

            actions.append(action[0, 0])
            if step > opt.ngsteps or \
               actions.count(actions[0]) == actions.maxlen:
                done = True

            if opt.render:
                env.render()

            if done:
                weighted = opt.discount * weighted + \
                           (1 - opt.discount) * reward_sum
                # only when this episode is done we can collect rewards
                writer.add_scalar(f"Eval Column {eid}/Reward", reward_sum,
                                  episode)
                progress_data = f"step: {step}, reward: {reward_sum:5.1f}, weighted: {weighted:5.1f}"
                iterator.set_postfix_str(progress_data)

                step, reward_sum = 0, 0
                actions.clear()
                state = torch.Tensor(env.reset())

                # time to move on
                # freeze parameters on global model
                if weighted > get_threshold(env.unwrapped.spec.id):
                    with lock:
                        gmodel.freeze()
                    break

        # freeze local model
        lmodel.freeze()

    auto.tqdm.write("Evaluation process terminated")
