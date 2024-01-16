# pyrotation
Interconversion of Euler angles, axis angles, quaternions, and rotation matrix.
Supports `numpy.ndarray` and `torch.Tensor`, and uses `plum-dispatch` to decorate functions of the same name.
In addition, it supports **arbitrary** shapes.

## Installation
```bash
pip install pyrotation
```

## conversion
we use `plum-dispatch` to decorate functions of the same name, so you can use `pyrotation.conversion` to convert between different representations.
We support `numpy.ndarray` and `torch.Tensor`, and the input can be of arbitrary shape.


|      | quaternion | matrix                 | euler_angle | axis_angle |
| ---- | ---------- |------------------------| ----------- | ---------- |
| quaternion | - | `quaternion_from_matrix` | `quaternion_from_euler_angle` | `quaternion_from_axis_angle` |
| matrix | `matrix_from_quaternion` | - | `matrix_from_euler_angle` | `matrix_from_axis_angle` |
| euler_angle | `euler_angle_from_quaternion` | `euler_angle_from_matrix` | - | `euler_angle_from_axis_angle` |
| axis_angle | `axis_angle_from_quaternion` | `axis_angle_from_matrix` | `axis_angle_from_euler_angle` | - |
 
```python
import torch
from pyrotation.conversion import quaternion_from_matrix, euler_angle_from_matrix

# convert a 10x3x3 rotation matrix to a quaternion
R = torch.rand(10, 3, 3)
quat = quaternion_from_matrix(R)
euler_angle = euler_angle_from_matrix(R)

print(quat.shape) # torch.Size([10, 4])
print(euler_angle.shape) # torch.Size([10, 3])
```

We also support conversions between `so3` and `SO3`, `se3` and `SE3`. You can use `SO3_from_so3` and `SE3_from_se3` to convert `so3` and `se3` to `SO3` and `SE3`, 
respectively. Similarly, you can use `so3_from_SO3` and `se3_from_SE3` to convert `SO3` and `SE3` to `so3` and `se3`, respectively.
