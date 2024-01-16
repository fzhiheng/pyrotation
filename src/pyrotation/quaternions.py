#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from typing import Union, Optional

import torch
import numpy as np
from plum import dispatch

from .core import is_less_then_epsilon_4th_root, arc_sin_x_over_x
from .conversion import quaternion_from_matrix, quaternion_from_axis_angle, quaternion_from_euler_angle, axis_angle_from_quaternion, \
    matrix_from_quaternion, euler_angle_from_quaternion


# Created by fzhiheng on 2024/1/15
# Copyright (c) 2024 fzhiheng. All rights reserved.
# 2024/1/15 下午7:27


@dispatch
def weighted_average_quaternions(quaternions: np.ndarray, weights: Optional[np.ndarray] = None):
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
def weighted_average_quaternions(quaternions: torch.Tensor, weights: Optional[torch.Tensor] = None):
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


class Quaternion(object):
    def __init__(self, quat: torch.Tensor):
        """

        Args:
            quat: (*,4) wxyz
        """
        self.data = quat
        self.check()
        self.normalize()

    def check(self):
        if self.data.dtype != torch.float32 and self.data.dtype != torch.float64:
            raise TypeError("Quaternion must be float32 or float64")
        if self.data.shape[-1] != 4:
            raise ValueError("Quaternion must be wxyz")

    @property
    def shape(self):
        return self.data.shape

    @property
    def w(self):
        return self.data[..., 0:1]

    @property
    def x(self):
        return self.data[..., 1:2]

    @property
    def y(self):
        return self.data[..., 2:3]

    @property
    def z(self):
        return self.data[..., 3:4]

    def __getitem__(self, item):
        if isinstance(item, tuple) and len(item) == self.data.ndim:
            if isinstance(item[-1], slice):
                if not (item[-1].start is None and item[-1].stop is None and item[-1].step is None):
                    raise IndexError("Last dimension of Quaternion cannot be sliced")
            else:
                raise IndexError("Last dimension of Quaternion cannot be sliced")
        return Quaternion(self.data[item])

    def __add__(self, other):
        return Quaternion(self.data + other.data)

    def __sub__(self, other):
        return Quaternion(self.data - other.data)

    def __mul__(self, other: 'Quaternion') -> 'Quaternion':
        """
        Args:
            other: (*,4) wxyz

        Returns: (*,4) wxyz

        """
        w1, x1, y1, z1 = self.w, self.x, self.y, self.z
        w2, x2, y2, z2 = other.w, other.x, other.y, other.z
        w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return Quaternion(torch.cat([w3, x3, y3, z3], dim=-1))

    def __div__(self, factor: Union[int, float, torch.Tensor]) -> 'Quaternion':
        return Quaternion(self.data / factor)

    __truediv__ = __div__

    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return "{} data:\n{}".format(type(self).__name__, str(self.data))

    def conjugate(self):
        return Quaternion(torch.cat([self.w, -self.x, -self.y, -self.z], dim=-1))

    def norm(self) -> torch.Tensor:
        return torch.linalg.norm(self.data, dim=-1, keepdim=True)

    def normalize(self):
        self.data = self.data / self.norm()

    def inverse(self):
        return self.conjugate() / self.norm()

    def log(self) -> torch.Tensor:
        """

        Returns: (*,3) xyz

        """
        q_imagi = torch.cat([self.x, self.y, self.z], dim=-1)
        na = torch.linalg.norm(q_imagi, dim=-1, keepdim=True)
        eta = self.w
        scale1 = torch.where(eta >= 0, torch.acos(eta) / na, -torch.acos(-eta) / na)
        scale2 = torch.where(eta > 0, arc_sin_x_over_x(na), -arc_sin_x_over_x(na))
        scale = torch.where(torch.abs(eta) < na, scale1, scale2)
        return q_imagi * (2.0 * scale)

    @staticmethod
    def exp(dx: torch.Tensor) -> 'Quaternion':
        input_dtype = dx.dtype
        assert input_dtype == torch.float32 or input_dtype == torch.float64

        theta = torch.linalg.norm(dx, dim=-1, keepdim=True)
        one_over_48 = 1.0 / 48.0
        na1 = 0.5 + (theta * theta) * one_over_48
        na2 = torch.sin(theta * 0.5) / theta
        na = torch.where(is_less_then_epsilon_4th_root(theta), na1, na2)
        ct = torch.cos(theta * 0.5)
        wxyz = torch.cat([ct, dx * na], dim=-1)
        return Quaternion(wxyz)

    def interpolate(self, other: 'Quaternion', ratio: Union[float, torch.Tensor]) -> 'Quaternion':
        """

        Args:
            other: another Quaternion
            ratio: interpolation ratio

        Returns: interpolated Quaternion

        """
        # check the range of ratio
        if isinstance(ratio, float):
            if ratio < 0 or ratio > 1:
                raise ValueError(f"ratio must be in [0,1], but get {ratio}")
        elif isinstance(ratio, torch.Tensor):
            if (ratio < 0).any() or (ratio > 1).any():
                raise ValueError(f"ratio must be in [0,1], but get {ratio}")
        else:
            raise TypeError(f"ratio must be float or torch.Tensor, but get {type(ratio)}")

        q1_2 = self.conjugate() * other
        q_inter = self * Quaternion.exp(ratio * q1_2.log())
        return q_inter

    @staticmethod
    def average(quats: 'Quaternion', weights: Optional[torch.Tensor] = None) -> 'Quaternion':
        """

        Args:
            quats: (*,N,4)  wxyz
            weights: (*,N)

        Returns:# (*,4)

        """
        average_quat = weighted_average_quaternions(quats.data, weights)
        return Quaternion(average_quat)

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor) -> 'Quaternion':
        """

        Args:
            matrix: (*,3,3)

        Returns: (*,4) wxyz

        """
        return cls(quaternion_from_matrix(matrix))

    @classmethod
    def from_axis_angle(cls, axis_angle: torch.Tensor) -> 'Quaternion':
        """

        Args:
            axis_angle: (*,3)

        Returns: (*,4) wxyz

        """
        return cls(quaternion_from_axis_angle(axis_angle))

    @classmethod
    def from_euler_angle(cls, euler_angle: torch.Tensor) -> 'Quaternion':
        """

        Args:
            euler_angle: (*,3)

        Returns: (*,4) wxyz

        """
        return cls(quaternion_from_euler_angle(euler_angle))

    def to_matrix(self) -> torch.Tensor:
        """

        Returns: (*,3,3)

        """
        return matrix_from_quaternion(self.data)

    def to_axis_angle(self) -> torch.Tensor:
        """

        Returns: (*,3)

        """
        return axis_angle_from_quaternion(self.data)

    def to_euler_angle(self) -> torch.Tensor:
        """

        Returns: (*,3)

        """
        return euler_angle_from_quaternion(self.data)


if __name__ == "__main__":
    wxyz1 = torch.rand(2, 3, 4)
    wxyz2 = torch.rand(2, 3, 4)
    q1 = Quaternion(wxyz1)
    q2 = Quaternion(wxyz2)
    q3 = q1 * q2
    print(q1)
    print(q2)
    print(q3)
    print(q1.interpolate(q2, 0.5))
