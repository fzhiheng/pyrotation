# -*- coding: UTF-8 -*-
from typing import Union

import torch
import numpy as np

from plum import dispatch

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
    if not torch.allclose(torch.eye(3).to(device), matrix @ matrix.transpose(-1, -2), rtol=0, atol=1e-6):
        raise ValueError("The matrix is not a valid rotation matrix")
    # 检查是否是右手坐标系
    if not torch.allclose(torch.det(matrix), torch.ones(1).to(device), rtol=0, atol=1e-6):
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

    Returns: (*,4)

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
    euler = euler/2
    yaw, pitch, roll = euler[..., 0], euler[..., 1], euler[..., 2]  # (*,)
    w = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(pitch) * np.cos(roll)  # (*,)
    x = np.sin(yaw) * np.cos(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(pitch) * np.sin(roll)  # (*,)
    y = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.cos(pitch) * np.sin(roll)  # (*,)
    z = np.cos(yaw) * np.cos(pitch) * np.sin(roll) - np.sin(yaw) * np.sin(pitch) * np.cos(roll)  # (*,)
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
    x = torch.sin(yaw) * torch.cos(pitch) * torch.cos(roll) - torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    y = torch.cos(yaw) * torch.sin(pitch) * torch.cos(roll) + torch.sin(yaw) * torch.cos(pitch) * torch.sin(roll)
    z = torch.cos(yaw) * torch.cos(pitch) * torch.sin(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.cos(roll)
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


# 欧拉角转旋转矩阵
@dispatch
def get_matrix_z(yaw: np.ndarray) -> np.ndarray:
    """ roation matrix around z-axis

    Args:
        yaw (np.ndarray): (*,)

    Returns:
        np.ndarray: (*,3,3)
    """
    Rz = np.stack(
        [
            np.stack([np.cos(yaw), -np.sin(yaw), np.zeros_like(yaw)], axis=-1),
            np.stack([np.sin(yaw), np.cos(yaw), np.zeros_like(yaw)], axis=-1),
            np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.ones_like(yaw)], axis=-1),
        ],
        axis=-2,
    )

    return Rz


@dispatch
def get_matrix_y(pitch: np.ndarray) -> np.ndarray:
    """ roation matrix around y-axis

    Args:
        pitch (np.ndarray): (*,)

    Returns:
        np.ndarray: (*,3,3)
    """
    Ry = np.stack(
        [
            np.stack([np.cos(pitch), np.zeros_like(pitch), np.sin(pitch)], axis=-1),
            np.stack([np.zeros_like(pitch), np.ones_like(pitch), np.zeros_like(pitch)], axis=-1),
            np.stack([-np.sin(pitch), np.zeros_like(pitch), np.cos(pitch)], axis=-1),
        ],
        axis=-2,
    )

    return Ry


@dispatch
def get_matrix_x(roll: np.ndarray) -> np.ndarray:
    """ roation matrix around x-axis

    Args:
        roll (np.ndarray): (*,)
    Returns:
        np.ndarray: (*,3,3)
    """

    Rx = np.stack(
        [
            np.stack([np.ones_like(roll), np.zeros_like(roll), np.zeros_like(roll)], axis=-1),
            np.stack([np.zeros_like(roll), np.cos(roll), -np.sin(roll)], axis=-1),
            np.stack([np.zeros_like(roll), np.sin(roll), np.cos(roll)], axis=-1),
        ],
        axis=-2,
    )
    return Rx


@dispatch
def get_matrix_z(yaw: torch.Tensor) -> torch.Tensor:
    """ roation matrix around z-axis

    Args:
        yaw (torch.Tensor): (*,)
    Returns:
        torch.Tensor: (*,3,3)
    """
    device = yaw.device
    Rz = torch.stack(
        [
            torch.stack([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw).to(device)], dim=-1),
            torch.stack([torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw).to(device)], dim=-1),
            torch.stack([torch.zeros_like(yaw).to(device),
                         torch.zeros_like(yaw).to(device),
                         torch.ones_like(yaw).to(device)], dim=-1),
        ],
        dim=-2,
    )
    return Rz


@dispatch
def get_matrix_y(pitch: torch.Tensor) -> torch.Tensor:
    """ roation matrix around y-axis

    Args:
        pitch (torch.Tensor): (*,)
    Returns:
        torch.Tensor: (*,3,3)
    """
    device = pitch.device
    Ry = torch.stack(
        [
            torch.stack([torch.cos(pitch), torch.zeros_like(pitch).to(device), torch.sin(pitch)], dim=-1),
            torch.stack([torch.zeros_like(pitch).to(device),
                         torch.ones_like(pitch).to(device),
                         torch.zeros_like(pitch).to(device)], dim=-1),
            torch.stack(
                [-torch.sin(pitch), torch.zeros_like(pitch).to(device), torch.cos(pitch)], dim=-1),
        ],
        dim=-2,
    )
    return Ry


@dispatch
def get_matrix_x(roll: torch.Tensor) -> torch.Tensor:
    """ roation matrix around x-axis

    Args:
        roll (torch.Tensor): (*,)
    Returns:
        torch.Tensor: (*,3,3)
    """
    device = roll.device
    Rx = torch.stack(
        [
            torch.stack([torch.ones_like(roll).to(device),
                         torch.zeros_like(roll).to(device),
                         torch.zeros_like(roll).to(device)], dim=-1),
            torch.stack([torch.zeros_like(roll).to(device), torch.cos(roll), -torch.sin(roll)], dim=-1),
            torch.stack([torch.zeros_like(roll).to(device), torch.sin(roll), torch.cos(roll)], dim=-1),
        ],
        dim=-2,
    )
    return Rx


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
    # 使用罗德里格旋转公式
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    axis_skew1 = np.stack([np.zeros_like(axis[..., 0]), -axis[..., 2], axis[..., 1]], axis=-1)  # (*,3)
    axis_skew2 = np.stack([axis[..., 2], np.zeros_like(axis[..., 0]), -axis[..., 0]], axis=-1)  # (*,3)
    axis_skew3 = np.stack([-axis[..., 1], axis[..., 0], np.zeros_like(axis[..., 0])], axis=-1)  # (*,3)
    axis_skew = np.stack([axis_skew1, axis_skew2, axis_skew3], axis=-2)  # (*,3,3)

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
    axis_skew1 = torch.stack([torch.zeros_like(axis[..., 0]).to(device), -axis[..., 2], axis[..., 1]], dim=-1)
    axis_skew2 = torch.stack([axis[..., 2], torch.zeros_like(axis[..., 0]).to(device), -axis[..., 0]], dim=-1)
    axis_skew3 = torch.stack([-axis[..., 1], axis[..., 0], torch.zeros_like(axis[..., 0]).to(device)], dim=-1)
    axis_skew = torch.stack([axis_skew1, axis_skew2, axis_skew3], dim=-2)

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


if __name__ == "__main__":
    shape = (1, 1, 3)
    axis_angle = np.ones(shape)

    matrix = matrix_from_axis_angle(axis_angle)
    quat = quaternion_from_axis_angle(axis_angle)
    euler = euler_angle_from_matrix(matrix)

    # check all functions
    assert np.allclose(axis_angle, axis_angle_from_quaternion(quat))
    assert np.allclose(axis_angle, axis_angle_from_matrix(matrix))
    assert np.allclose(axis_angle, axis_angle_from_euler(euler))

    assert np.allclose(matrix, matrix_from_quaternion(quat))
    assert np.allclose(matrix, matrix_from_euler_angle(euler))

    assert np.allclose(euler, euler_angle_from_quaternion(quat))
    assert np.allclose(euler, euler_angle_from_axis_angle(axis_angle))

    assert np.allclose(quat, quaternion_from_matrix(matrix))
    assert np.allclose(quat, quaternion_from_euler_angle(euler))

    axis_angle = torch.ones(shape)
    matrix = matrix_from_axis_angle(axis_angle)
    quat = quaternion_from_axis_angle(axis_angle)
    euler = euler_angle_from_matrix(matrix)

    # check all functions
    assert torch.allclose(axis_angle, axis_angle_from_quaternion(quat))
    assert torch.allclose(axis_angle, axis_angle_from_matrix(matrix))
    assert torch.allclose(axis_angle, axis_angle_from_euler(euler))

    assert torch.allclose(matrix, matrix_from_quaternion(quat))
    assert torch.allclose(matrix, matrix_from_euler_angle(euler))

    assert torch.allclose(euler, euler_angle_from_quaternion(quat))
    assert torch.allclose(euler, euler_angle_from_axis_angle(axis_angle))

    assert torch.allclose(quat, quaternion_from_matrix(matrix))
    assert torch.allclose(quat, quaternion_from_euler_angle(euler))
