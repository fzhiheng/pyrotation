# -*- coding: UTF-8 -*-
from typing import Union

import torch
import numpy as np

from plum import dispatch

# File    ：quaternions.py
# Author  ：fzhiheng
# Date    ：2023/10/11 下午8:29

"""
    quaternions are represented as (w,x,y,z) with w being the scalar. 
"""


@dispatch
def weighted_average_quaternions(quaternions: np.ndarray, weights: Union[np.ndarray, None] = None):
    """

    Args:
        quaternions: is a (*,N,4) numpy matrix and contains the quaternions to average in the rows.
            The quaternions are arranged as (w,x,y,z), with w being the scalar
        weights: The weight of the quaternions (*,N,1)

    Returns:the average quaternion of the input. Note that the signs of the output quaternion can be reversed,
        since q and -q describe the same orientation # (*,4)

    Raises: ValueError if all weights are zero

    """

    # Number of quaternions to average
    if weights is None:
        weights = np.ones_like(quaternions[..., 0:1])

    mat_a = quaternions.swapaxes(-1, -2) @ (quaternions * weights)  # (*,4,4)
    weight_sum = np.sum(weights, axis=-2, keepdims=True)  # (*,1,1)
    if np.any(weight_sum <= 0.0):
        raise ValueError("At least one weight must be greater than zero")

    # scale
    mat_a = (1.0 / weight_sum) * mat_a

    eigen_values, eigen_vectors = np.linalg.eig(mat_a)

    index = np.argmax(eigen_values, axis=-1)
    eigen_vectors = np.take_along_axis(eigen_vectors, index[..., None, None], axis=-1)  # (*,4,1)
    eigen_vectors = np.squeeze(eigen_vectors, axis=-1)  # (*,4)

    return np.real((eigen_vectors))


@dispatch
def weighted_average_quaternions(quaternions: torch.Tensor, weights: Union[torch.Tensor, None] = None):
    """

    Args:
        quaternions: is a (*,N,4)  matrix, w,x,y,z
        weights:

    Returns:# (*,4) (*,4)

    """

    if weights is None:
        weights = torch.ones_like(quaternions[..., 0:1])

    weight_sum = torch.sum(weights, dim=-2, keepdim=True)  # (*,1,1)
    if torch.any(weight_sum <= 0.0):
        raise ValueError("At least one weight must be greater than zero")

    mat_a = torch.transpose(quaternions, -1, -2) @ (quaternions * weights)  # (*,4,4)
    mat_a = (1.0 / weight_sum) * mat_a
    eigen_values, eigen_vectors = torch.linalg.eig(mat_a)  # (*,4), # (*,4,4)
    eigen_values = torch.real(eigen_values)
    index = torch.argmax(eigen_values, dim=-1)

    eigen_vectors = torch.take_along_dim(eigen_vectors, index[..., None, None], dim=-1)  # (*,4,1)
    eigen_vectors = torch.squeeze(eigen_vectors, dim=-1)  # (*,4)
    return torch.real((eigen_vectors))


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


@dispatch
def exp_quat(dx: np.ndarray):
    """

    Args:
        dx: (*,3)

    Returns: (*,4) wxyz

    """
    theta = np.linalg.norm(dx, axis=-1, keepdims=True)  # (*,1)
    one_over_48 = 1.0 / 48.0
    na1 = 0.5 + (theta * theta) * one_over_48  # (*,1)
    na2 = np.sin(theta * 0.5) / theta  # (*,1)
    na = np.where(is_less_then_epsilon_4th_root(theta), na1, na2)  # (*,1)
    ct = np.cos(theta * 0.5)  # (*,1)
    return np.concatenate([ct, dx * na], axis=-1)  # (*,4)


@dispatch
def exp_quat(dx: torch.Tensor):
    """

    Args:
        dx: (*,3)

    Returns: (*,4) wxyz

    """
    input_dtype = dx.dtype
    assert input_dtype == torch.float32 or input_dtype == torch.float64

    theta = torch.linalg.norm(dx, dim=-1, keepdim=True)
    one_over_48 = 1.0 / 48.0
    na1 = 0.5 + (theta * theta) * one_over_48
    na2 = torch.sin(theta * 0.5) / theta
    na = torch.where(is_less_then_epsilon_4th_root(theta), na1, na2)
    ct = torch.cos(theta * 0.5)
    return torch.cat([ct, dx * na], dim=-1)

@dispatch
def log_quat(q: np.ndarray):
    """

    Args:
        q:  (*,4) wxyz

    Returns: (*,3)

    """
    q_imagi = q[..., 1:]
    na = np.linalg.norm(q_imagi, axis=-1, keepdims=True)  # (*,1)
    eta = q[..., 0:1]  # (*,1)
    # use eta because it is more precise than na to calculate the scale. No singularities here.
    bool_criterion = np.abs(eta) < na  # (*,1)

    scale1 = np.where(eta >= 0, np.arccos(eta) / na, -np.arccos(-eta) / na)  # (*,1)
    scale2 = np.where(eta > 0, arc_sin_x_over_x(na), -arc_sin_x_over_x(na))  # (*,1)
    scale = np.where(bool_criterion, scale1, scale2)  # (*,1)
    return q_imagi * (2.0 * scale)


@dispatch
def log_quat(q: torch.Tensor):
    """

    Args:
        q:  (*,4) wxyz

    Returns: (*,3)

    """
    input_dtype = q.dtype
    assert input_dtype == torch.float32 or input_dtype == torch.float64

    q_imagi = q[..., 1:]
    na = torch.linalg.norm(q_imagi, dim=-1, keepdim=True)
    eta = q[..., 0:1]
    # use eta because it is more precise than na to calculate the scale. No singularities here.
    bool_criterion = torch.abs(eta) < na

    scale1 = torch.where(eta >= 0, torch.acos(eta) / na, -torch.acos(-eta) / na)
    scale2 = torch.where(eta > 0, arc_sin_x_over_x(na), -arc_sin_x_over_x(na))
    scale = torch.where(bool_criterion, scale1, scale2)
    return q_imagi * (2.0 * scale)


def interploate_quat(q1, q2, ratio):
    """

    Args:
        q1: (*,4) wxyz
        q2: (*,4) wxyz
        ratio: (*,1) 0-1

    Returns: (*,4) wxyz

    """
    q1_conj = torch.cat([q1[..., 0:1], -q1[..., 1:]], dim=-1)  # (*,4) wxyz
    q1_2 = multi_quat(q1_conj, q2)  # (*,4) wxyz
    q_inter = multi_quat(q1, exp_quat(ratio * log_quat(q1_2)))  # (*,4) wxyz
    return q_inter





@dispatch
def multi_quat(q1: np.ndarray, q2: np.ndarray):
    """

    Args:
        q1: (*,4) wxyz
        q2: (*,4) wxyz

    Returns: (*,4) wxyz

    """
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.concatenate([w, x, y, z], axis=-1)


@dispatch
def multi_quat(q1: torch.Tensor, q2: torch.Tensor):
    """

    Args:
        q1: (*,4) wxyz
        q2: (*,4) wxyz

    Returns: (*,4) wxyz

    """
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.cat([w, x, y, z], dim=-1)


if __name__ == "__main__":
    print(pow(0.01, 0.5))
    threshold = np.finfo(np.float64).eps
    print(threshold)
    print(np.finfo(np.float64))
    print(np.finfo(np.float32))
    print(torch.finfo(torch.float32))
    print(torch.finfo(torch.float64))
    print(np.power(threshold, 1.0 / 4.0))
