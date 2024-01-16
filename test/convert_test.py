#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
import numpy as np

from src.pyrotation.conversion import *

# convert_test.py
# Created by fzhiheng on 2024/1/16
# Copyright (c) 2024 fzhiheng. All rights reserved.
# 2024/1/16 下午5:10

if __name__ == "__main__":

    shape = (1, 1, 3)
    axis_angle = np.random.random(shape)

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

    axis_angle = torch.randn(shape)
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

    se3 = np.random.rand(2, 2, 6)
    assert np.allclose(se3, se3_from_SE3(SE3_from_se3(se3)))

    se3 = torch.randn(2, 2, 6)
    assert torch.allclose(se3, se3_from_SE3(SE3_from_se3(se3)))

    print("test success!")


