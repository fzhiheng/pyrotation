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