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
    skew1 = torch.stack([torch.zeros_like(vector[..., 0]).to(device), -vector[..., 2], vector[..., 1]], dim=-1)
    skew2 = torch.stack([vector[..., 2], torch.zeros_like(vector[..., 0]).to(device), -vector[..., 0]], dim=-1)
    skew3 = torch.stack([-vector[..., 1], vector[..., 0], torch.zeros_like(vector[..., 0]).to(device)], dim=-1)
    skew = torch.stack([skew1, skew2, skew3], dim=-2)
    return skew


@dispatch
def is_less_then_epsilon_4th_root(x: np.ndarray):
    input_dtype = x.dtype
    assert input_dtype == np.float32 or input_dtype == np.float64
    return x < np.power(np.finfo(input_dtype).eps, 1.0 / 4.0)


@dispatch
def is_less_then_epsilon_4th_root(x: torch.Tensor):
    input_dtype = x.dtype
    device = x.device
    assert input_dtype == torch.float32 or input_dtype == torch.float64
    return x < torch.pow(torch.finfo(input_dtype).eps, torch.tensor(1.0 / 4.0, dtype=input_dtype, device=device))


@dispatch
def arc_sin_x_over_x(x: np.ndarray):
    """

    Args:
        x:  (*,1)

    Returns: (*,1)

    """
    return np.where(is_less_then_epsilon_4th_root(np.abs(x)), 1.0 + x * x * (1.0 / 6.0), np.arcsin(x) / x)


@dispatch
def arc_sin_x_over_x(x: torch.Tensor):
    """

    Args:
        x:  (*,1)

    Returns: (*,1)

    """
    return torch.where(is_less_then_epsilon_4th_root(torch.abs(x)), 1.0 + x * x * (1.0 / 6.0), torch.asin(x) / x)