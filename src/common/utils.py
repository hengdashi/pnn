#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from collections import deque
import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return np.concatenate(states, 0), actions, rewards, np.concatenate(
            next_states, 0), dones

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
