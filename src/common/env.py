#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

import gym
from gym.spaces.box import Box

from skimage import util

from common.utils import cv2_clipped_zoom


def create_env(opt):
    envs = []
    if opt.envs == 'pong':
        if opt.ncolumns == 1:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4')))
            ]
        elif opt.ncolumns == 2:
            envs = [
                NormalizedEnv(
                    PongNoisy(AtariRescale(gym.make('PongDeterministic-v4')))),
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4')))
            ]
        elif opt.ncolumns == 3:
            envs = [
                NormalizedEnv(
                    PongNoisy(AtariRescale(gym.make('PongDeterministic-v4')))),
                NormalizedEnv(
                    AtariRescale(PongHFlip(gym.make('PongDeterministic-v4')))),
                NormalizedEnv(
                    AtariRescale(PongZoom(gym.make('PongDeterministic-v4'))))
            ]
    elif opt.envs == 'atari':
        if opt.ncolumns == 1:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('AlienDeterministic-v4')))
            ]
        elif opt.ncolumns == 2:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4'))),
                NormalizedEnv(AtariRescale(gym.make('BoxingDeterministic-v4')))
            ]
        elif opt.ncolumns == 3:
            envs = [
                NormalizedEnv(AtariRescale(gym.make('PongDeterministic-v4'))),
                NormalizedEnv(
                    AtariRescale(gym.make('RiverraidDeterministic-v4'))),
                NormalizedEnv(AtariRescale(gym.make('BoxingDeterministic-v4')))
            ]
    return envs


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        # Box(low, high, shape)
        self.observation_space = Box(0.0, 1.0, [1, 84, 84])

    def observation(self, frame):
        # rescale to 160 x 160
        frame = frame[34:34 + 160, :160]
        # resize to 84 x 84
        frame = cv2.resize(frame, (84, 84))
        # take mean of three rgb values
        # frame would be turned to grayscale
        frame = frame.mean(2, keepdims=True)
        frame = frame.astype(np.float32)
        # normalize to [0, 1] range
        frame *= (1.0 / 255.0)
        # switch to pytorch format
        frame = np.moveaxis(frame, -1, 0)
        return frame


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


class PongHFlip(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.unwrapped.spec.id = 'PongFlipDeterministic-v4'

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


class PongNoisy(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.unwrapped.spec.id = 'PongNoisyDeterministic-v4'

    def reset(self):
        observation = self.env.reset()
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        return observation

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        observation = util.random_noise(observation, mode='gaussian', seed=1)
        return observation, reward, done, _

    def render(self):
        self.env.render()


class PongZoom(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.unwrapped.spec.id = 'PongZoomDeterministic-v4'

    def reset(self):
        observation = self.env.reset()
        observation = cv2_clipped_zoom(observation, 0.75)
        return observation

    def step(self, action):
        observation, reward, done, _ = self.env.step(action)
        observation = cv2_clipped_zoom(observation, 0.75)
        return observation, reward, done, _

    def render(self):
        self.env.render()
