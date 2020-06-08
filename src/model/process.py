#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import deque

import torch
import timeit
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from model.pnn import PNN
from model.env import create_env


def local_train(pid, opt, gmodel, optimizer, save=False):
    torch.manual_seed(opt.seed + pid)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(opt.log_path)
    allenvs = create_env()
    lmodel = PNN(nlayers=opt.nlayers)
    # define sizes of each layer and add the columns
    # TODO: might need to be changed for non ram version
    # TODO: try only one environment for now
    lmodel.add([
        allenvs[0].observation_space.shape[0], 128, 64, 32,
        allenvs[0].action_space.n
    ])
    # turn to train mode
    lmodel.train()

    for env in allenvs:
        env.seed(opt.seed + pid)

    for i in range(len(allenvs)):
        # get states
        envs = allenvs[:i + 1]
        states = [torch.Tensor(env.reset()) for env in envs]

        done = True
        step, episode = 0, 0
        while True:
            if save:
                if episode % opt.interval == 0 and episode > 0:
                    torch.save(gmodel.state_dict(), f"{opt.path}/pnn")
                print(f"Process {pid}, Episode {episode}")
            episode += 1
            # 1. worker reset to global network
            lmodel.load_state_dict(gmodel.state_dict())

            log_policies, values, rewards, entropies = [], [], [], []

            for _ in range(opt.nlsteps):
                step += 1
                logits, value = lmodel(states)
                #  print(f"logits: {logits}, value: {value}")
                policy = F.softmax(logits, dim=-1)
                log_policy = F.log_softmax(logits, dim=-1)
                entropy = -(policy * log_policy).sum(1, keepdim=True)

                action = Categorical(policy).sample().item()
                log_policy = log_policy.gather(1, action)

                # 2. worker interacts with the environment
                state, reward, done, _ = envs[-1].step(action)
                state = torch.Tensor(state)

                done = True if step > opt.ngsteps else done

                if done:
                    step = 0
                    state = torch.Tensor(envs[-1].reset())

                values.append(value)
                log_policies.append(log_policy)
                rewards.append(reward)
                entropies.append(entropy)

                if done:
                    break

            R = torch.zeros((1, 1), dtype=torch.float)
            if not done:
                _, R = lmodel(state)

            gae = torch.zeros((1, 1), dtype=torch.float)

            actor_loss, critic_loss, entropy_loss = 0, 0, 0
            next_value = R

            for value, log_policy, reward, entropy in list(
                    zip(values, log_policies, rewards, entropies))[::-1]:
                gae = gae * opt.gamma * opt.tau
                gae = gae + reward + opt.gamma * next_value.detach(
                ) - value.detach()
                next_value = value
                actor_loss += log_policy * gae
                R = R * opt.gamma + reward
                critic_loss += (R - value)**2 / 2
                entropy_loss += entropy

            # 3. worker calculates the value and policy loss
            print(
                f"actor_loss: {actor_loss}\ncritic_loss: {critic_loss}\nentropy_loss: {entropy_loss}"
            )
            loss = -actor_loss + critic_loss - opt.beta * entropy_loss
            #  print(loss, episode)
            #  writer.add_scalar(f"Train_{pid}/Loss", loss, episode)
            optimizer.zero_grad()
            # 4. worker gets gradients from losses
            loss.backward()

            for lparams, gparams in zip(lmodel.parameters(),
                                        gmodel.parameters()):
                if gparams.grad is not None:
                    break
                # 5. worker updates global network with gradients
                gparams._grad = lparams.grad

            optimizer.step()

            if episode == int(opt.ngsteps / opt.nlsteps):
                print("Training process {index} terminated")
                if save:
                    end_time = timeit.default_timer()
                    print(f"The code runs for {(end_time - start_time):.2f}")
                lmodel.freeze()
                break


def local_test(pid, opt, gmodel):
    torch.manual_seed(opt.seed + pid)
    allenvs = create_env()
    lmodel = PNN(nlayers=opt.nlayers)
    # TODO: might need to be changed for non ram version
    lmodel.add([
        allenvs[0].observation_space.shape[0], 128, 64, 32,
        allenvs[0].action_space.n
    ])
    lmodel.eval()

    for env in allenvs:
        env.seed(opt.seed + pid)
    for i in range(len(allenvs)):
        envs = allenvs[:i + 1]
        states = [torch.Tensor(env.reset()) for env in envs]
        done = True
        step = 0
        actions = deque(maxlen=opt.max_actions)
        while True:
            step += 1
            if done:
                lmodel.load_state_dict(gmodel.state_dict())

            logits, value = lmodel(states)
            policy = F.softmax(logits, dim=0)
            action = torch.argmax(policy).item()
            state, reward, done, _ = envs[-1].step(action)
            if opt.render:
                envs[-1].render()
            actions.append(action)
            if step > opt.ngsteps or actions.count(
                    actions[0]) == actions.maxlen:
                done = True
            if done:
                step = 0
                actions.clear()
                state = envs[-1].reset()
            state = torch.Tensor(state)
