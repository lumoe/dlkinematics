import tensorflow.keras.backend as K
from .urdf import chain_from_urdf_file
from .dlkinematics import DLKinematics

from dlkinematics.urdf_parser.urdf import Link, Joint, Robot, Pose

import tensorflow_graphics.geometry.transformation.quaternion as tfg_quaternion
import tensorflow_graphics.geometry.transformation.rotation_matrix_3d as tfg_rotation_matrix_3d
import tensorflow_graphics.geometry.transformation.axis_angle as tfg_axis_angle
import tensorflow_graphics.geometry.transformation.euler as tfg_euler
import tensorflow_graphics as tfg
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import numpy as np
print('tensorflow.graphics.__version__', tfg.__version__)


def get_loss_function(fn_rotation, fn_translation):
    fn_rotation = rotation_losses[fn_rotation]
    fn_translation = translation_losses[fn_translation]

    def SO3_loss(y_actual, y_estimated):
        loss_1 = fn_translation(y_actual, y_estimated)
        loss_2 = fn_rotation(y_actual, y_estimated)
        return tf.reduce_mean(tf.cast([loss_1, loss_2], tf.float32))
    return SO3_loss


@tf.custom_gradient
def l2_norm(x):
    y = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))

    def grad(dy):
        return dy*(x/K.maximum(y, K.epsilon()))
    return y, grad


def translation_with_rmse(y_actual, y_estimated):
    translation_actual = y_actual[:, 3:]
    translation_estimated = y_estimated[:, :3, 3:]
    translation_estimated = tf.reshape(
        translation_estimated, shape=(tf.shape(translation_estimated)[0], -1))

    x = tf.math.subtract(translation_actual, translation_estimated)
    y = l2_norm(x)
    return tf.reduce_mean(y)


def rotation_with_rmse(y_actual, y_estimated):
    angles_actual = y_actual[:, :3]
    angles_estimated = y_estimated[:, :3, :3]
    angles_estimated = tfg_euler.from_rotation_matrix(angles_estimated)
    x = tf.subtract(angles_actual, angles_estimated)
    y = l2_norm(x)
    return tf.reduce_mean(y)


"""
    Implemented different loss functions for the rotation vector according to this paper.
    https://www.researchgate.net/profile/Du_Huynh/publication/220146411_Metrics_for_3D_Rotations_Comparison_and_Analysis/links/00b7d51f39f278a9d2000000/Metrics-for-3D-Rotations-Comparison-and-Analysis.pdf
    
    All losses are metrics in SO3. 
"""


@tf.custom_gradient
def phi2_norm(x):
    y = K.sqrt(K.sum(K.square(x), axis=-1, keepdims=True))

    def grad(dy):
        res = dy*(x/K.maximum(y, K.epsilon()))
        return res
    return y, grad


def phi2_loss(y_actual, y_estimated):
    """
        Norm of the Difference of Quaternions
    """
    angles_actual = y_actual[:, :3]
    angles_estimated = y_estimated[:, :3, :3]
    tfg.geometry.transformation.rotation_matrix_3d.assert_rotation_matrix_normalized(
        angles_estimated, eps=0.0001)
    angles_actual = tfg_quaternion.from_euler(angles_actual)

    angles_estimated = tfg_euler.from_rotation_matrix(angles_estimated)
    angles_estimated = tfg_quaternion.from_euler(angles_estimated)

    minimum = tf.math.minimum(
        phi2_norm(tf.math.add(angles_actual, angles_estimated)), phi2_norm(tf.math.subtract(angles_actual, angles_estimated)))
    e = tf.reduce_mean(minimum)
    return e


def phi3_loss(y_actual, y_estimated):
    """
        Arccosine of the Inner Product of Unit Quaternions
    """
    angles_actual = y_actual[:, :3]
    angles_estimated = y_estimated[:, :3, :3]
    angles_actual = tfg_quaternion.from_euler(angles_actual)
    # Drop outer most dimension
    angles_actual = tf.squeeze(angles_actual)
    angles_estimated = tfg_euler.from_rotation_matrix(angles_estimated)
    angles_estimated = tfg_quaternion.from_euler(angles_estimated)
    angles_actual = tf.reshape(angles_actual, tf.shape(angles_estimated))

    res = tf.math.abs(tf.keras.backend.batch_dot(
        angles_actual, angles_estimated, 1))

    res = tf.math.acos(K.clip(res, -1, 1))
    res = tf.reduce_mean(res)
    return res


def phi4_loss(y_actual, y_estimated):
    """
        Inner Product of Unit Quaternions
    """
    angles_actual = y_actual[:, :3]
    angles_estimated = y_estimated[:, :3, :3]
    angles_actual = tfg_quaternion.from_euler(angles_actual)
    angles_estimated = tfg_euler.from_rotation_matrix(angles_estimated)
    angles_estimated = tfg_quaternion.from_euler(angles_estimated)
    angles_actual = tf.reshape(angles_actual, tf.shape(angles_estimated))
    e = 1. - \
        (tf.math.abs(tf.keras.backend.batch_dot(angles_actual, angles_estimated, 1)))
    e = tf.reduce_mean(e)
    return e


def phi5_loss(y_actual, y_estimated):
    """
        Deviation from the Identity Matrix
    Args:
        y_actual (Tensor): shape(None, 6), dtype=float32 or float64
        y_estimated (Tensor): shape(None, 4, 4), dtype=float64 or float64
    Returns:
        Tensor: scalar value. 0 <= error <= 2*sqrt(2)
    """
    angles_actual = y_actual[:, :3]
    angles_estimated = y_estimated[:, :3, :3]
    angles_actual = tfg.geometry.transformation.rotation_matrix_3d.from_euler(
        angles_actual)
    angles_actual = tf.cast(angles_actual, dtype=tf.float32)
    angles_estimated = tf.cast(angles_estimated, dtype=tf.float32)
    angles_error = tf.math.reduce_euclidean_norm(tf.eye(3, dtype=tf.float32) - tf.linalg.matmul(angles_actual,
                                                                                                angles_estimated, transpose_b=True), axis=(-2, -1))
    return angles_error[-1]


rotation_losses = {
    'phi2': phi2_loss,
    'phi3': phi3_loss,
    'phi4': phi4_loss,
    'phi5': phi5_loss,
    'rmse': rotation_with_rmse,
    'euclidean': rotation_with_rmse
}


translation_losses = {
    'euclidian': translation_with_rmse,
    'rmse': translation_with_rmse
}


class ForwardKinematics(layers.Layer):
    def __init__(self, urdf_file, base_link, end_link, batch_size, **kwargs):
        self.urdf_file = urdf_file
        self.base_link = base_link
        self.end_link = end_link
        self.batch_size = batch_size
        super(ForwardKinematics, self).__init__(**kwargs)

    def build(self, input_shape):
        dlkinematics_chain = chain_from_urdf_file(self.urdf_file)
        self.kernel = DLKinematics(
            dlkinematics_chain, self.base_link, self.end_link, self.batch_size)

    def call(self, input):
        return self.kernel.forward(tf.reshape(input, [-1]))

    def compute_output_shape(self, input_shape):
        return (self.batch_size, 4, 4)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'urdf_file': self.urdf_file,
            'base_link': self.base_link,
            'end_link': self.end_link,
            'batch_size': self.batch_size,
        })
        return config


def num_joints_for_urdf(urdf_file, root=None, tip=None):
    dlkinematics_chain = chain_from_urdf_file(urdf_file)
    if not root:
        root = dlkinematics_chain.links[0].name

    if not tip:
        tip = dlkinematics_chain.links[-1].name

    dl_kin = DLKinematics(dlkinematics_chain, root, tip)
    return dl_kin.num_joints


def subsitute_link_with_joint(urdf, target_link, root, last, xyz=[0, 0, 0], rpy=[0, 0, 0]):
    new_chain = Robot()
    for idx, element in enumerate(urdf.get_chain(root, last, joints=True, links=True, fixed=True)):
        if element == target_link:
            prev_joint_name = urdf.get_chain(root, last)[idx-1]
            prev_joint = urdf.joint_map.get(prev_joint_name)

            next_joint_name = urdf.get_chain(root, last)[idx+1]
            next_joint = urdf.joint_map.get(next_joint_name)

            print(next_joint.origin.xyz)
            print(next_joint.origin.rpy)

            new_chain.add_link(Link(name='imputed_link_1'))
            new_chain.add_link(Link(name='imputed_link_2'))
            new_chain.add_joint(
                Joint(axis=[1, 1, 1],
                      parent='imputed_link_1',
                      child='imputed_link_2',
                      name='imputed_joint',
                      joint_type='floating',
                      origin=Pose(xyz=[0, 0, 0], rpy=[0, 0, 0]
                                  #   origin=Pose(xyz=next_joint.origin.xyz, rpy=next_joint.origin.rpy
                                  )),
            )

            next_joint.parent = 'imputed_link_2'
            next_joint.origin = Pose(xyz=[0., 0., 0.], rpy=[0., 0., 0.])
            new_chain.add_joint(next_joint)

            prev_joint.child = 'imputed_link_1'
            new_chain.add_joint(prev_joint)

        else:
            if link := urdf.link_map.get(element):
                new_chain.add_link(link)
            elif joint := urdf.joint_map.get(element):
                new_chain.add_joint(joint)
    return new_chain
