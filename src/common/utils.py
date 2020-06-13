#!/usr/bin/env python
# -*- coding: utf-8 -*-

from statistics import mean, median

import numpy as np
import cv2


def gen_stats(data):
    return min(data), max(data), mean(data), median(data)


def get_threshold(env_name):
    if 'Pong' in env_name:
        return 18
    elif 'Riverraid' in env_name:
        return 14000
    elif 'Boxing' in env_name:
        return 90
    elif 'NameThisGame' in env_name:
        return 14000
    elif 'Alien' in env_name:
        return 2500


def cv2_clipped_zoom(ori, factor):
    h, w = ori.shape[:2]
    h_new, w_new = int(h * factor), int(w * factor)

    y1, x1 = max(0, h_new - h) // 2, max(0, w_new - w) // 2
    y2, x2 = y1 + h, x1 + w
    bbox = np.array([y1, x1, y2, x2])
    bbox = (bbox / factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    crop = ori[y1:y2, x1:x2]

    h_resize, w_resize = min(h_new, h), min(w_new, w)
    h1_pad, w1_pad = (h - h_resize) // 2, (w - w_resize) // 2
    h2_pad, w2_pad = (h - h_resize) - h1_pad, (w - w_resize) - w1_pad
    spec = [(h1_pad, h2_pad), (w1_pad, w2_pad)] + [(0, 0)] * (ori.ndim - 2)

    resize = cv2.resize(crop, (w_resize, h_resize))
    resize = np.pad(resize, spec, mode='constant')
    assert resize.shape[0] == h and resize.shape[1] == w
    return resize
