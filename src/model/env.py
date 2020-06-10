#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from skimage import util
from common.utils import cv2_clipped_zoom


def create_env():
    # TODO: try only one environment, latter can be changed to more
    envs = [
        NormalizedEnv(gym.make('Pong-ramDeterministic-v4')) for _ in range(1)
    ]
    return envs


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.nsteps = 0

    def observation(self, observation):
        self.nsteps += 1
        self.state_mean = self.state_mean * self.alpha + \
            observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + \
            observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.nsteps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.nsteps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class Pong():
    def __init__(self):
        self.env = gym.make('Pong-ramDeterministic-v4')
        self.action_space = self.env.action_space


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
