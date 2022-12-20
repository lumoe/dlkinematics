import numpy as np
import tensorflow as tf
from dlkinematics.training_utils import phi5_loss
from numpy import cos, sin


def homogenous_transformation_matrix(x, y, z, phi_x, phi_y, phi_z):
    rx = tf.convert_to_tensor(
        [
            [1, 0, 0, 0],
            [0, cos(phi_x), sin(phi_x), 0],
            [0, -sin(phi_x), cos(phi_x), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )
    ry = tf.convert_to_tensor(
        [
            [cos(phi_y), 0, -sin(phi_y), 0],
            [0, 1, 0, 0],
            [sin(phi_y), 0, cos(phi_y), 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )
    rz = tf.convert_to_tensor(
        [
            [cos(phi_z), -sin(phi_z), 0, 0],
            [sin(phi_z), cos(phi_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=tf.float32,
    )
    M = tf.convert_to_tensor(
        [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]], dtype=tf.float32
    )
    return rz @ ry @ rx @ M


x, y, z = 1, 2, 3
# create transformation matrices and also add batch dimension
target = tf.expand_dims(homogenous_transformation_matrix(x, y, z, 0, np.pi, 0), axis=0)
result = tf.expand_dims(homogenous_transformation_matrix(x, y, z, 0, 0, 0), axis=0)
error = phi5_loss(result, target)

print(f"Distance between target and result: {error}")