#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from skimage import util
from common.utils import cv2_clipped_zoom


def create_env():
    # TODO: try only one environment, latter can be changed to more
    envs = [gym.make('Pong-ramDeterministic-v4') for _ in range(1)]
    return envs


class PongFlip():
    def __init__(self):
        self.env = gym.make('PongDeterministic-v4')
        self.action_space = self.env.action_space

    def reset(self):
        observation = self.env.reset()
        observation = observation[::-1]
        return observation

    def step(self, action):
        if action == 2 or action == 4:
            action = 3
        elif action == 3 or action == 5:
            action = 4
        observation, reward, done, _ = self.env.step(action)
        observation = observation[::-1]
        return observation, reward, done, _

    def render(self):
        self.env.render()


class PongNoisy():
    def __init__(self):
        self.env = gym.make('PongDeterministic-v4')
        self.action_space = self.env.action_space

    def reset(self):
        observation = self.env.reset()
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        observation = (observation * 255).astype('int')
        return observation

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        observation = (observation * 255).astype('int')
        return observation, reward, done, _

    def render(self):
        self.env.render()


class PongZoom():
    def __init__(self):
        self.env = gym.make('PongDeterministic-v4')
        self.action_space = self.env.action_space

    def reset(self):
        observation = self.env.reset()
        observation = cv2_clipped_zoom(observation, 0.8)
        return observation

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        observation = cv2_clipped_zoom(observation, 0.8)
        return observation, reward, done, _

    def render(self):
        self.env.render()
