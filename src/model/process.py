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


def ltrain(pid, opt, gmodel, optimizer, save=False):
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
        allenvs[0].observation_space.shape[0], 64, 32, 16,
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
                    torch.save(gmodel.state_dict(), opt.save_path / "pnn")
                print(f"Process {pid}, Episode {episode}")
            episode += 1
            # 1. worker reset to global network
            lmodel.load_state_dict(gmodel.state_dict())

            log_policies, values, rewards, entropies = [], [], [], []

            for _ in range(opt.nlsteps):
                step += 1
                logits, value = lmodel(states)
                policy = F.softmax(logits, dim=-1)
                log_policy = F.log_softmax(logits, dim=-1)
                entropy = -(policy * log_policy).sum()

                #  print(
                #  f"logits: {logits}, policy: {policy}, log_policy: {log_policy}"
                #  )

                action = Categorical(policy).sample().item()
                log_policy = log_policy[action]
                # 2. worker interacts with the environment
                state, reward, done, _ = envs[-1].step(action)
                state = torch.Tensor(state)
                reward = max(min(reward, 1), -1)

                done = True if step > opt.ngsteps else done

                if done:
                    step = 0
                    state = torch.Tensor(envs[-1].reset())

                values.append(value)
                log_policies.append(log_policy)
                rewards.append(reward)
                entropies.append(entropy)
                #  print(
                #  f"value: {value}, log_policy: {log_policy}, reward: {reward}, entropy: {entropy}"
                #  )

                if done:
                    break

            R = torch.zeros((1, 1), dtype=torch.float)
            if not done:
                _, R = lmodel(state)
                R = R.detach()
            values.append(R)

            gae = torch.zeros((1, 1), dtype=torch.float)

            actor_loss, critic_loss = 0, 0

            for i in reversed(range(len(rewards))):
                R = R * opt.gamma + rewards[i]
                advantage = R - values[i]
                critic_loss += 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta = rewards[i] + opt.gamma * values[i + 1] - values[i]
                gae = gae * opt.gamma * opt.tau + delta

                actor_loss -= log_policies[i] * gae.detach(
                ) + opt.beta * entropies[i]

            # 3. worker calculates the value and policy loss
            #  print(
            #  f"actor_loss: {actor_loss}\ncritic_loss: {critic_loss}\nentropy_loss: {entropy_loss}"
            #  )
            loss = actor_loss + opt.critic_loss_coef * critic_loss
            writer.add_scalar(f"Train_{pid}/Loss", loss, episode)
            optimizer.zero_grad()
            # 4. worker gets gradients from losses
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(), opt.clip)

            # 5. worker updates global network with gradients
            for lparams, gparams in zip(lmodel.parameters(),
                                        gmodel.parameters()):
                if gparams.grad is not None:
                    break
                gparams._grad = lparams.grad

            optimizer.step()

            if episode == int(opt.ngsteps / opt.nlsteps):
                print("Training process {index} terminated")
                if save:
                    end_time = timeit.default_timer()
                    print(f"The code runs for {(end_time - start_time):.2f}")
                lmodel.freeze()
                break


def ltest(pid, opt, gmodel):
    torch.manual_seed(opt.seed + pid)
    allenvs = create_env()
    lmodel = PNN(nlayers=opt.nlayers)
    # TODO: might need to be changed for non ram version
    lmodel.add([
        allenvs[0].observation_space.shape[0], 64, 32, 16,
        allenvs[0].action_space.n
    ])
    lmodel.eval()

    for env in allenvs:
        env.seed(opt.seed + pid)
    for i in range(len(allenvs)):
        envs = allenvs[:i + 1]
        states = [torch.Tensor(env.reset()) for env in envs]
        done = True
        step, reward_sum = 0, 0
        actions = deque(maxlen=opt.max_actions)
        while True:
            step += 1
            if done:
                lmodel.load_state_dict(gmodel.state_dict())

            with torch.no_grad():
                logits, value = lmodel(states)
            policy = F.softmax(logits, dim=-1)
            action = torch.argmax(policy).item()
            state, reward, done, _ = envs[-1].step(action)
            done = done or step > opt.ngsteps
            reward_sum += reward
            if opt.render:
                envs[-1].render()
            actions.append(action)
            if actions.count(actions[0]) == actions.maxlen:
                done = True
            if done:
                step = 0
                actions.clear()
                state = envs[-1].reset()
            state = torch.Tensor(state)
