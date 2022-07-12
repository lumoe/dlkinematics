from __future__ import print_function, division, unicode_literals
import sys

import numpy as np
import tensorflow as tf
"""

Matrix([
    [1.0*cy*cz, -1.0*cx*sz + cz*sx*sy, cx*cz*sy + 1.0*sx*sz,   0],
    [1.0*cy*sz,  1.0*cx*cz + sx*sy*sz, cx*sy*sz - 1.0*cz*sx,   0],
    [-1.0*sy,             1.0*cy*sx,            1.0*cx*cy,   0],
    [0,                     0,                    0, 1.0]])
"""


def tf_homogeneous_transformation(sin, cos, translation):
    sx, sy, sz = tf.unstack(sin, num=3, axis=-1)
    cx, cy, cz = tf.unstack(cos, num=3, axis=-1)
    x, y, z = tf.unstack(translation, num=3, axis=-1)
    zeros = tf.zeros(tf.shape(sx), dtype=tf.float32)
    ones = tf.ones(tf.shape(sx), dtype=tf.float32)
    r00 = cy * cz
    r01 = -1.0 * cx * sz + cz * sx * sy
    r02 = cx * cz * sy + 1.0 * sx * sz
    r10 = cy * sz
    r11 = cx * cz + sx * sy * sz
    r12 = cx * sy * sz - 1.0 * cz * sx
    r20 = -1.0 * sy
    r21 = cy * sx
    r22 = cx * cy

    transformation_matrix = tf.stack((
        r00, r01, r02, x,
        r10, r11, r12, y,
        r20, r21, r22, z,
        zeros, zeros, zeros, ones
    ), axis=-1)
    output_shape = tf.concat(
        (tf.shape(sin)[:-1], (4, 4)), axis=-1)
    return tf.reshape(
        transformation_matrix, shape=output_shape)
