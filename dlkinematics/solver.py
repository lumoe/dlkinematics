from __future__ import print_function, absolute_import
import sys

import numpy as np
import tensorflow as tf

from .tf_transformations import tf_homogeneous_transformation

from transformations import quaternion_matrix, quaternion_from_euler, translation_matrix


def solve_static(chain):
    matrices = []
    for _, joint in enumerate(chain):
        r = rotation(joint.rotation)
        t = translation_matrix(joint.offset)
        matrices.append(np.matmul(t, r))

    return matrices


def rotation(angles):
    q = quaternion_from_euler(angles[0], angles[1], angles[2], 'sxyz')
    return quaternion_matrix(q)


@tf.function
def solve_forward(forward_matrices, thetas, theta_indices, thetas_shape, debug):
    thetas = tf.scatter_nd(theta_indices, thetas, thetas_shape)
    sin = tf.math.sin(thetas[:, :, :3])
    cos = tf.math.cos(thetas[:, :, :3])
    translation = thetas[:, :, 3:]
    transformation_matrices = tf_homogeneous_transformation(
        sin, cos, translation)

    transformation_matrices = tf.linalg.matmul(
        forward_matrices, transformation_matrices
    )

    r = tf.scan(lambda x, y: tf.linalg.matmul(x, y), transformation_matrices)
    if debug:
        return r
    else:
        return r[-1]
