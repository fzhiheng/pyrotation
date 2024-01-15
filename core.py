#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
import numpy as np

from plum import dispatch


# core.py
# Created by fzhiheng on 2024/1/15
# Copyright (c) 2024 fzhiheng. All rights reserved.
# 2024/1/15 下午4:02

@dispatch
def skew_symmetric(vector: np.ndarray):
    """

    Args:
        vector:  (*,3)

    Returns: (*,3,3)

    """
    skew1 = np.stack([np.zeros_like(vector[..., 0]), -vector[..., 2], vector[..., 1]], axis=-1)  # (*,3)
    skew2 = np.stack([vector[..., 2], np.zeros_like(vector[..., 0]), -vector[..., 0]], axis=-1)  # (*,3)
    skew3 = np.stack([-vector[..., 1], vector[..., 0], np.zeros_like(vector[..., 0])], axis=-1)  # (*,3)
    skew = np.stack([skew1, skew2, skew3], axis=-2)  # (*,3,3)
    return skew


@dispatch
def skew_symmetric(vector: torch.Tensor):
    """

    Args:
        vector:  (*,3)

    Returns: (*,3,3)

    """
    device = vector.device
    skew1 = torch.stack([torch.zeros_like(vector[..., 0]).to(device), -vector[..., 2], vector[..., 1]], axis=-1)
    skew2 = torch.stack([vector[..., 2], torch.zeros_like(vector[..., 0]).to(device), -vector[..., 0]], axis=-1)
    skew3 = torch.stack([-vector[..., 1], vector[..., 0], torch.zeros_like(vector[..., 0]).to(device)], axis=-1)
    skew = torch.stack([skew1, skew2, skew3], axis=-2)
    return skew
