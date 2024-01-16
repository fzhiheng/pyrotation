# -*- coding: UTF-8 -*-
from typing import Union

import torch
import numpy as np
from plum import dispatch

from .core import skew_symmetric, is_less_then_epsilon_4th_root, get_matrix_y, get_matrix_x, get_matrix_z

# File    ：quaternions.py
# Author  ：fzhiheng
# Date    ：2023/10/11 下午8:29

"""
This module contains functions for working with conversion between different rotational representations.
All functions support batched inputs and outputs. All functions support numpy arrays and torch tensors as input and output.

The following representations are supported:
    quaternions are represented as (w,x,y,z) with w being the scalar.
    matrix are represented as 3x3 matrices.
    axis angle are represented as (x,y,z) with theta being the angle of rotation around the axis.
    euler angles are represented as (yaw,pitch,roll), and yaw is the rotation around the z axis, pitch around the y axis and roll around the x axis.

"""


@dispatch
def check_matrix(matrix: np.ndarray):
    """
    Check if a matrix is a valid rotation matrix in right hand coordinate system.
    Raises a ValueError if this is not the case.

    Args:
        matrix: The matrix to check

    Returns: None

    Raises: ValueError if the matrix is not a valid rotation matrix

    """
    if not np.allclose(np.eye(3), matrix @ np.swapaxes(matrix, -1, -2), rtol=0, atol=1e-6):
        raise ValueError("The matrix is not a valid rotation matrix")
    # 检查是否是右手坐标系
    if not np.allclose(np.linalg.det(matrix), 1.0, rtol=0, atol=1e-6):
        raise ValueError("The matrix is in a right hand coordinate system")


@dispatch
def check_matrix(matrix: torch.Tensor):
    """
    Check if a matrix is a valid rotation matrix in right hand coordinate system.
    Raises a ValueError if this is not the case.

    Args:
        matrix: The matrix to check

    Returns: None

    Raises: ValueError if the matrix is not a valid rotation matrix

    """
    device = matrix.device
    if not torch.allclose(torch.eye(3, dtype=matrix.dtype).to(device), matrix @ matrix.transpose(-1, -2), rtol=0, atol=1e-6):
        raise ValueError("The matrix is not a valid rotation matrix")
    # 检查是否是右手坐标系
    if not torch.allclose(torch.det(matrix), torch.ones(1, dtype=matrix.dtype).to(device), rtol=0, atol=1e-6):
        raise ValueError("The matrix is in a right hand coordinate system")


# 旋转矩阵转四元数
@dispatch
def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """

    Args:
        matrix: (*,3,3)

    Returns: (*,4)

    """
    check_matrix(matrix)
    w = np.sqrt(1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]) / 2
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / (4 * w)
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / (4 * w)
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / (4 * w)
    return np.stack([w, x, y, z], axis=-1)


# 旋转矩阵转四元数
@dispatch
def quaternion_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """

    Args:
        matrix: (*,3,3)

    Returns: (*,4) wxyz

    """
    check_matrix(matrix)
    w = torch.sqrt(1 + matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2]) / 2  # (*,)
    x = (matrix[..., 2, 1] - matrix[..., 1, 2]) / (4 * w)  # (*,)
    y = (matrix[..., 0, 2] - matrix[..., 2, 0]) / (4 * w)  # (*,)
    z = (matrix[..., 1, 0] - matrix[..., 0, 1]) / (4 * w)  # (*,)
    return torch.stack([w, x, y, z], dim=-1)  # (*,4)


# 轴角转四元数
@dispatch
def quaternion_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """

    Args:
        axis_angle: (*,3)

    Returns: wxyz

    """
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)  # (*,1)
    axis = axis_angle / angle  # (*,3)
    sin_result = np.sin(angle / 2)  # (*,1)
    return np.concatenate([np.cos(angle / 2), axis * sin_result], axis=-1)  # (*,4)


# 轴角转四元数
@dispatch
def quaternion_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """

    Args:
        axis_angle: (*,3)

    Returns: wxyz

    """
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle
    sin_result = torch.sin(angle / 2)
    return torch.cat([torch.cos(angle / 2), axis * sin_result], dim=-1)


# 欧拉角转四元数
@dispatch
def quaternion_from_euler_angle(euler: np.ndarray) -> np.ndarray:
    """

    Args:
        euler: (*,3) 默认是ypr,内旋

    Returns: wxyz

    """
    euler = euler / 2
    yaw, pitch, roll = euler[..., 0], euler[..., 1], euler[..., 2]  # (*,)
    w = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(pitch) * np.cos(roll)  # (*,)
    x = -np.sin(yaw) * np.sin(pitch) * np.cos(roll) + np.cos(yaw) * np.cos(pitch) * np.sin(roll)  # (*,)
    y = np.sin(yaw) * np.cos(pitch) * np.sin(roll) + np.cos(yaw) * np.sin(pitch) * np.cos(roll)  # (*,)
    z = -np.cos(yaw) * np.sin(pitch) * np.sin(roll) + np.sin(yaw) * np.cos(pitch) * np.cos(roll)
    return np.stack([w, x, y, z], axis=-1)  # (*,4)


# 欧拉角转四元数
@dispatch
def quaternion_from_euler_angle(euler: torch.Tensor) -> torch.Tensor:
    """

    Args:
        euler: (*,3) 默认是ypr,内旋

    Returns: wxyz

    """
    euler = euler / 2
    yaw, pitch, roll = euler[..., 0], euler[..., 1], euler[..., 2]
    w = torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll) + torch.cos(yaw) * torch.cos(pitch) * torch.cos(roll)
    x = -torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll) + torch.cos(yaw) * torch.cos(pitch) * torch.sin(roll)
    y = torch.sin(yaw) * torch.cos(pitch) * torch.sin(roll) + torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll)
    z = -torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll) + torch.sin(yaw) * torch.cos(pitch) * torch.cos(roll)
    return torch.stack([w, x, y, z], dim=-1)


# 四元数转旋转矩阵
@dispatch
def matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """

    Args:
        quaternion: (*,4) w,x,y,z

    Returns: rotation_matrix (*,3,3)

    """
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    rot_matx = np.zeros((*quaternion.shape[:-1], 3, 3))
    rot_matx[..., 0, 0] = np.power(w, 2) + np.power(x, 2) - np.power(y, 2) - np.power(z, 2)
    rot_matx[..., 0, 1] = 2 * x * y - 2 * w * z
    rot_matx[..., 0, 2] = 2 * x * z + 2 * w * y
    rot_matx[..., 1, 0] = 2 * x * y + 2 * w * z
    rot_matx[..., 1, 1] = np.power(w, 2) - np.power(x, 2) + np.power(y, 2) - np.power(z, 2)
    rot_matx[..., 1, 2] = 2 * y * z - 2 * w * x
    rot_matx[..., 2, 0] = 2 * x * z - 2 * w * y
    rot_matx[..., 2, 1] = 2 * y * z + 2 * w * x
    rot_matx[..., 2, 2] = np.power(w, 2) - np.power(x, 2) - np.power(y, 2) + np.power(z, 2)
    return rot_matx


# 四元数转旋转矩阵
@dispatch
def matrix_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """

    Args:
        quaternion: (*,4) wxyz

    Returns: rotation_matrix (*,3,3)

    """
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    rot_matx = torch.zeros((*quaternion.shape[:-1], 3, 3), device=quaternion.device)
    rot_matx[..., 0, 0] = torch.pow(w, 2) + torch.pow(x, 2) - torch.pow(y, 2) - torch.pow(z, 2)
    rot_matx[..., 0, 1] = 2 * x * y - 2 * w * z
    rot_matx[..., 0, 2] = 2 * x * z + 2 * w * y
    rot_matx[..., 1, 0] = 2 * x * y + 2 * w * z
    rot_matx[..., 1, 1] = torch.pow(w, 2) - torch.pow(x, 2) + torch.pow(y, 2) - torch.pow(z, 2)
    rot_matx[..., 1, 2] = 2 * y * z - 2 * w * x
    rot_matx[..., 2, 0] = 2 * x * z - 2 * w * y
    rot_matx[..., 2, 1] = 2 * y * z + 2 * w * x
    rot_matx[..., 2, 2] = torch.pow(w, 2) - torch.pow(x, 2) - torch.pow(y, 2) + torch.pow(z, 2)
    return rot_matx


def matrix_from_euler_angle(euler_angle: Union[torch.Tensor, np.ndarray], axes=("z", "y", "x")) -> Union[torch.Tensor, np.ndarray]:
    """ get rotation matrix from euler angle,default order is yaw, pitch, roll

    Args:
        euler_angle (Union[torch.Tensor, np.ndarray]): (*,3), (yaw, pitch, roll)

    Returns:
            Union[torch.Tensor, np.ndarray]: (*,3,3)
    """
    yaw = euler_angle[..., 0]
    pitch = euler_angle[..., 1]
    roll = euler_angle[..., 2]
    R_z = get_matrix_z(yaw)
    R_y = get_matrix_y(pitch)
    R_x = get_matrix_x(roll)
    axis_rotation = {"z": R_z, "y": R_y, "x": R_x}
    matrix = axis_rotation[axes[0]] @ axis_rotation[axes[1]] @ axis_rotation[axes[2]]
    return matrix


# 轴角转旋转矩阵
@dispatch
def matrix_from_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """

    Args:
        axis_angle: (*,3)

    Returns: (*,3,3)

    """
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)  # (*,1)
    axis = axis_angle / angle  # (*,3)
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    axis_skew = skew_symmetric(axis)  # (*,3,3)
    angle_cos = np.cos(angle)[..., None]  # (*,1,1)
    angle_sin = np.sin(angle)[..., None]  # (*,1,1)

    R = angle_cos * np.eye(3) + angle_sin * axis_skew + (1 - angle_cos) * axis[..., None] @ axis[..., None, :]

    return R


@dispatch
def matrix_from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """

    Args:
        axis_angle: (*,3)

    Returns: (*,3,3)

    """
    device = axis_angle.device
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / angle
    # 使用罗德里格旋转公式
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    axis_skew = skew_symmetric(axis)
    angle_cos = torch.cos(angle)[..., None]
    angle_sin = torch.sin(angle)[..., None]

    R = angle_cos * torch.eye(3).to(device) + angle_sin * axis_skew + (1 - angle_cos) * axis[..., None] @ axis[..., None, :]
    return R


# 四元数转轴角
@dispatch
def axis_angle_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """

    Args:
        quaternion: (*,4)

    Returns: (*,3)

    """
    angle = 2 * np.arccos(quaternion[..., 0:1])
    axis = quaternion[..., 1:] / np.sin(angle / 2)
    return angle * axis


@dispatch
def axis_angle_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """四元数转轴角

    Args:
        quaternion: (*,4)

    Returns: (*,3)

    """
    angle = 2 * torch.arccos(quaternion[..., 0:1])
    axis = quaternion[..., 1:] / torch.sin(angle / 2)
    return angle * axis


# 旋转矩阵转轴角
@dispatch
def axis_angle_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵转轴角

    Args:
        matrix: (*,3,3)

    Returns: (*,3)

    """
    check_matrix(matrix)
    angle = np.arccos((matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2] - 1) / 2)  # (*,)
    angle = angle[..., None]
    axis = np.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        axis=-1,
    )
    axis = axis / (2 * np.sin(angle))
    return angle * axis


@dispatch
def axis_angle_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵转轴角

    Args:
        matrix: (*,3,3)

    Returns: (*,3)

    """
    check_matrix(matrix)
    angle = torch.arccos((matrix[..., 0, 0] + matrix[..., 1, 1] + matrix[..., 2, 2] - 1) / 2)  # (*,)
    angle = angle[..., None]
    axis = torch.stack(
        [
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ],
        dim=-1,
    )  # (*,3)
    axis = axis / (2 * torch.sin(angle))
    return angle * axis


# 欧拉角转轴角
def axis_angle_from_euler(euler_angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """欧拉角转轴角

    Args:
        euler_angle: (*,3)

    Returns: (*,3)

    """
    matrix = matrix_from_euler_angle(euler_angle)
    axis_angle = axis_angle_from_matrix(matrix)
    return axis_angle


# 四元数转欧拉角
@dispatch
def euler_angle_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """四元数转欧拉角
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    Args:
        quaternion: (*,4)

    Returns: (*,3)

    """
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.stack([yaw, pitch, roll], axis=-1)


@dispatch
def euler_angle_from_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """四元数转欧拉角
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    Args:
        quaternion: (*,4)

    Returns: (*,3)

    """
    w, x, y, z = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.sign(sinp) * np.pi / 2, torch.arcsin(sinp))
    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return torch.stack([yaw, pitch, roll], dim=-1)


# 旋转矩阵转欧拉角
@dispatch
def euler_angle_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵转欧拉角
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    Args:
        matrix: (*,3,3)

    Returns: (*,3)

    """
    check_matrix(matrix)
    sy = np.sqrt(matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0])
    singular = sy < 1e-6
    roll = np.where(singular, np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1]),
                    np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]))
    pitch = np.where(singular, np.arctan2(-matrix[..., 2, 0], sy), np.arctan2(-matrix[..., 2, 0], sy))
    yaw = np.where(singular, 0, np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))
    return np.stack([yaw, pitch, roll], axis=-1)


@dispatch
def euler_angle_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
    """旋转矩阵转欧拉角
    https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    Args:
        matrix: (*,3,3)

    Returns: (*,3)

    """
    check_matrix(matrix)
    sy = torch.sqrt(matrix[..., 0, 0] * matrix[..., 0, 0] + matrix[..., 1, 0] * matrix[..., 1, 0])
    singular = sy < 1e-6
    roll = torch.where(singular, torch.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1]),
                       torch.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]))
    pitch = torch.where(singular, torch.arctan2(-matrix[..., 2, 0], sy), torch.arctan2(-matrix[..., 2, 0], sy))
    yaw = torch.where(singular, 0, torch.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))
    return torch.stack([yaw, pitch, roll], dim=-1)


# 轴角转欧拉角
def axis_angle_from_euler_angle(euler_angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """欧拉角转轴角
    Args:
        euler_angle: (*,3)

    Returns: (*,3)

    """
    return axis_angle_from_quaternion(quaternion_from_euler_angle(euler_angle))


def euler_angle_from_axis_angle(axis_angle: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """轴角转欧拉角
    Args:
        axis_angle: (*,3)

    Returns: (*,3)

    """
    return euler_angle_from_quaternion(quaternion_from_axis_angle(axis_angle))


# other functions
@dispatch
def fill_matrix(matrix: np.ndarray, t: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        matrix (np.ndarray): (*,3,3) matrix
        t (np.ndarray): (*,3)

    Returns:
        np.ndarray: (*,4,4)
    """
    full_matrix = np.concatenate([matrix, t[..., None]], axis=-1)
    padding = np.zeros_like(full_matrix[..., :1, :])
    padding[..., 0, -1] = 1
    full_matrix = np.concatenate([full_matrix, padding], axis=-2)
    return full_matrix


@dispatch
def fill_matrix(matrix: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        matrix (torch.Tensor): (*,3,3) matrix
        t (torch.Tensor): (*,3)

    Returns:
        torch.Tensor: (*,4,4)
    """
    full_matrix = torch.cat([matrix, t[..., None]], dim=-1)
    padding = torch.zeros_like(full_matrix[..., :1, :])
    padding[..., 0, -1] = 1
    full_matrix = torch.cat([full_matrix, padding], dim=-2)
    return full_matrix


def full_matrix_from_qt(q: Union[np.ndarray, torch.Tensor], t: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """_summary_

    Args:
        q (Union[np.ndarray, torch.Tensor]): (*,4) wxyz
        t (Union[np.ndarray, torch.Tensor]): (*,3)

    Returns:
        Union[np.ndarray, torch.Tensor]: (*,4,4)
    """
    matrix = matrix_from_quaternion(q)  # (*,3,3)
    full_matrix = fill_matrix(matrix, t)
    return full_matrix


def SO3_from_so3(phi: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """_summary_

    Args:
        so3 (Union[np.ndarray, torch.Tensor]): (*,3)

    Returns:
        Union[np.ndarray, torch.Tensor]: (*,3,3)
    """
    return matrix_from_axis_angle(phi)


def so3_from_SO3(SO3: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """_summary_

    Args:
        SO3 (Union[np.ndarray, torch.Tensor]): (*,3,3)

    Returns:
        Union[np.ndarray, torch.Tensor]: (*,3)
    """
    return axis_angle_from_matrix(SO3)


@dispatch
def SE3_from_se3(se3: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        se3 (Union[np.ndarray, torch.Tensor]): (*,6) (phi, rho) phi: rotation, rho: translation

    Returns:
        Union[np.ndarray, torch.Tensor]: (*,4,4)
    """
    phi = se3[..., :3]  # (*,3)
    rho = se3[..., 3:]  # (*,3)
    angle = np.linalg.norm(phi, axis=-1, keepdims=True)
    a = phi / angle
    skew = skew_symmetric(a)
    angle_cos = np.cos(angle)
    angle_sin = np.sin(angle)
    R = angle_cos[..., None] * np.eye(3) + angle_sin[..., None] * skew + (1 - angle_cos[..., None]) * a[..., None] @ a[..., None, :]
    tmp1 = np.where(is_less_then_epsilon_4th_root(np.abs(angle)), 1.0 - angle * angle * (1.0 / 6.0), angle_sin / angle)
    tmp2 = np.where(is_less_then_epsilon_4th_root(np.abs(angle)), angle * (1.0 / 2.0) - angle * angle * angle * (1.0 / 24.0), (1 - angle_cos) / angle)
    tmp1 = tmp1[..., None]
    tmp2 = tmp2[..., None]
    jacobi = tmp1 * np.eye(3) + (1 - tmp1) * a[..., None] @ a[..., None, :] + tmp2 * skew  # (*,3,3)
    t = jacobi @ rho[..., None]
    SE3 = fill_matrix(R, t[..., 0])
    return SE3


@dispatch
def SE3_from_se3(se3: torch.Tensor) -> torch.Tensor:
    """_summary_

    Args:
        se3 (Union[np.ndarray, torch.Tensor]): (*,6) (phi, rho) phi: rotation, rho: translation

    Returns:
        Union[np.ndarray, torch.Tensor]: (*,4,4)
    """
    phi = se3[..., :3]  # (*,3)
    rho = se3[..., 3:]  # (*,3)
    device = se3.device
    angle = torch.linalg.norm(phi, dim=-1, keepdim=True)
    a = phi / angle
    skew = skew_symmetric(a)
    angle_cos = torch.cos(angle)
    angle_sin = torch.sin(angle)
    R = angle_cos[..., None] * torch.eye(3).to(device) + angle_sin[..., None] * skew + (1 - angle_cos[..., None]) * a[..., None] @ a[..., None, :]
    tmp1 = torch.where(is_less_then_epsilon_4th_root(torch.abs(angle)), 1.0 - angle * angle * (1.0 / 6.0), angle_sin / angle)
    tmp2 = torch.where(is_less_then_epsilon_4th_root(torch.abs(angle)), angle * (1.0 / 2.0) - angle * angle * angle * (1.0 / 24.0),
                       (1 - angle_cos) / angle)
    tmp1 = tmp1[..., None]
    tmp2 = tmp2[..., None]
    jacobi = tmp1 * torch.eye(3).to(device) + (1 - tmp1) * a[..., None] @ a[..., None, :] + tmp2 * skew  # (*,3,3)
    t = jacobi @ rho[..., None]
    SE3 = fill_matrix(R, t[..., 0])
    return SE3


@dispatch
def se3_from_SE3(SE3: np.ndarray) -> np.ndarray:
    """

    Args:
        SE3: (*,4,4)

    Returns:

    """
    R = SE3[..., :3, :3]
    t = SE3[..., :3, 3:4]
    phi = so3_from_SO3(R)  # (*,3)
    angle = np.linalg.norm(phi, axis=-1, keepdims=True)
    skew = skew_symmetric(phi)  # (*,3,3)
    angle_cos = np.cos(angle)
    angle_sin = np.sin(angle)
    tmp1 = np.where(is_less_then_epsilon_4th_root(np.abs(angle)), 1.0 - angle * angle * (1.0 / 6.0), angle_sin / angle)
    tmp2 = np.where(is_less_then_epsilon_4th_root(np.abs(angle)), 0.5 - angle * angle * (1.0 / 24.0), (1 - angle_cos) / angle ** 2)
    tmp = (1 - tmp1 / (2 * tmp2)) / (angle ** 2)

    jacobi_inv = np.ones_like(tmp[..., None]) * np.eye(3) - 0.5 * skew + tmp[..., None] * skew @ skew  # (*,3,3)
    rho = (jacobi_inv @ t)[..., 0]

    se3 = np.concatenate([phi, rho], -1)
    return se3


@dispatch
def se3_from_SE3(SE3: torch.Tensor) -> torch.Tensor:
    """

    Args:
        SE3: (*,4,4)

    Returns:

    """
    R = SE3[..., :3, :3]
    t = SE3[..., :3, 3:4]
    device = SE3.device
    phi = so3_from_SO3(R)  # (*,3)
    angle = torch.linalg.norm(phi, axis=-1, keepdims=True)
    skew = skew_symmetric(phi)  # (*,3,3)
    angle_cos = torch.cos(angle)
    angle_sin = torch.sin(angle)
    tmp1 = torch.where(is_less_then_epsilon_4th_root(np.abs(angle)), 1.0 - angle * angle * (1.0 / 6.0), angle_sin / angle)
    tmp2 = torch.where(is_less_then_epsilon_4th_root(np.abs(angle)), 0.5 - angle * angle * (1.0 / 24.0), (1 - angle_cos) / angle ** 2)
    tmp = (1 - tmp1 / (2 * tmp2)) / (angle ** 2)

    jacobi_inv = torch.ones_like(tmp[..., None]).to(device) * torch.eye(3, dtype=SE3.dtype).to(device) - 0.5 * skew + tmp[..., None] * skew @ skew
    rho = (jacobi_inv @ t)[..., 0]

    se3 = torch.cat([phi, rho], -1)
    return se3
