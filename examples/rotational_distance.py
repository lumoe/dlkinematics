import numpy as np
import tensorflow as tf
from numpy import cos, sin

from dlkinematics.training_utils import phi2_loss, phi3_loss, phi4_loss, phi5_loss


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
phi_x, phi_y, phi_z = 0, np.pi, 0

# create transformation matrices and also add batch dimension
target = tf.expand_dims(
    homogenous_transformation_matrix(x, y, z, phi_x, phi_y, phi_z), axis=0)
target_phis = tf.expand_dims(tf.convert_to_tensor(
    [phi_x, phi_y, phi_z, 0, 0, 0]), axis=0)
result = tf.expand_dims(
    homogenous_transformation_matrix(x, y, z, 0, 0, 0), axis=0)

print(
    f"Distance between target and result (phi2): {phi2_loss(target_phis, result)}")
print(
    f"Distance between target and result (phi3): {phi3_loss(target_phis, result)}")
print(
    f"Distance between target and result (phi4): {phi4_loss(target_phis, result)}")
print(
    f"Distance between target and result (phi5): {phi5_loss(target, result)}")
