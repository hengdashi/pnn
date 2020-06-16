#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from model.pnn import PNN

from common.env import create_env


def train(pid, opt, current, gmodel, optimizer):
    torch.manual_seed(opt.seed + pid)
    writer = SummaryWriter(opt.log_path)
    allenvs = create_env(opt)
    lmodel = PNN(allenvs)

    # turn to train mode
    lmodel.train()

    for cid, env in enumerate(allenvs):
        env.seed(opt.seed + pid)
        # get state
        state = torch.Tensor(env.reset())

        step, done = 0, True
        # number of total steps locally
        for episode in range(int(opt.ngsteps / opt.nlsteps)):
            # 1. worker reset to global network
            lmodel.load_state_dict(gmodel.state_dict())

            log_probs, values, rewards, entropies = [], [], [], []

            # interacting for n local steps
            for _ in range(opt.nlsteps):
                step += 1
                value, logits = lmodel(state.unsqueeze(0))
                prob = F.softmax(logits, dim=-1)
                log_prob = F.log_softmax(logits, dim=-1)
                entropy = -(prob * log_prob).sum(1, keepdim=True)

                action = prob.multinomial(num_samples=1).detach()
                log_prob = log_prob.gather(1, action)
                # 2. worker interacts with the environment
                state, reward, done, _ = env.step(action.numpy())
                state = torch.Tensor(state)

                # clip reward
                reward = max(min(reward, 1), -1)

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)

                done = True if step > opt.ngsteps else done
                if done:
                    step = 0
                    state = torch.Tensor(env.reset())
                    break

            R = torch.zeros((1, 1), dtype=torch.float)
            if not done:
                value, _ = lmodel(state.unsqueeze(0))
                R = value.detach()
            values.append(R)

            actor_loss, critic_loss = 0, 0
            gae = torch.zeros((1, 1), dtype=torch.float)
            for i in reversed(range(len(rewards))):
                R = R * opt.gamma + rewards[i]
                advantage = R - values[i]
                critic_loss += 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation
                delta = rewards[i] + opt.gamma * values[i + 1] - values[i]
                gae = gae * opt.gamma * opt.tau + delta

                actor_loss -= opt.beta * entropies[i] + \
                              log_probs[i] * gae.detach()

            # 3. worker calculates the value and policy loss
            optimizer.zero_grad()
            # 4. worker gets gradients from losses
            loss = actor_loss + opt.critic_loss_coef * critic_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lmodel.parameters(cid), opt.clip)

            # 5. worker updates global network with gradients
            # move to next environment if global model is ahead
            if lmodel.current < current.value:
                break

            for lparams, gparams in zip(lmodel.parameters(cid),
                                        gmodel.parameters(cid)):
                if gparams.grad is not None:
                    break
                gparams._grad = lparams.grad

            optimizer.step()

            # TIME TO LOG DATA
            writer.add_scalar(f"Train_{pid}/Loss", loss, episode)
            #  writer.add_scalar(f"Train_{pid}/Reward", sum(rewards), episode)

        # FREEZE PREVIOUS COLUMNS
        lmodel.freeze()

    print(f"Training process {pid} terminated")
