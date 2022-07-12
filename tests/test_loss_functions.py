import pytest
import numpy as np
from transformations import rotation_matrix, concatenate_matrices
import tensorflow as tf
from dlkinematics.training_utils import translation_with_rmse, phi5_loss, phi3_loss, phi4_loss, phi2_loss, get_loss_function

import sys

module = sys.modules['tensorflow_graphics.util.shape']
def _get_dim(tensor, axis):
    """Returns dimensionality of a tensor for a given axis."""
    return tf.compat.v1.dimension_value(tensor.shape[axis])

module._get_dim = _get_dim
sys.modules['tensorflow_graphics.util.shape'] = module


def test_translation_euclidian_error_to_zero_batch_size_1(batch_size=1):
    translations = np.array([[0., 0., 0., 1., 2., 3.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[1., 0., 0., 1.],
          [0., 1., 0., 2.],
          [0., 0., 1., 3.],
          [0., 0., 0., 1.]]]
    )
    e = translation_with_rmse(translations, homogeneous_transformation)
    assert np.allclose(e, 0.)


def test_translation_euclidian_error_to_one_batch_size_1():
    translations = np.array([[0., 0., 0., 0., 2., 3.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[1., 0., 0., 1.],
          [0., 1., 0., 2.],
          [0., 0., 1., 3.],
          [0., 0., 0., 1.]]]
    )
    e = translation_with_rmse(translations, homogeneous_transformation)
    assert np.allclose(e, 1.)


def test_translation_euclidian_error_to_zero_batch_size_2(batch_size=1):
    translations = np.array(
        [[0., 0., 0., 1., 2., 3.], [0., 0., 0., 1., 2., 3.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[1., 0., 0., 1.],
          [0., 1., 0., 2.],
          [0., 0., 1., 3.],
          [0., 0., 0., 1.]],
         [[1., 0., 0., 1.],
          [0., 1., 0., 2.],
          [0., 0., 1., 3.],
          [0., 0., 0., 1.]]]
    )
    e = translation_with_rmse(translations, homogeneous_transformation)
    assert np.allclose(e, 0.)


def test_translation_euclidian_error_to_one_batch_size_2():
    translations = np.array(
        [[0., 0., 0., 0., 2., 3.], [0., 0., 0., 0., 2., 3.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[1., 0., 0., 1.],
          [0., 1., 0., 2.],
          [0., 0., 1., 3.],
          [0., 0., 0., 1.]],
         [[1., 0., 0., 1.],
          [0., 1., 0., 2.],
          [0., 0., 1., 3.],
          [0., 0., 0., 1.]]]
    )
    e = translation_with_rmse(translations, homogeneous_transformation)
    assert np.allclose(e, 1.)


def _get_rotation_matrix(alpha, beta, gamma):
    from transformations import rotation_matrix, concatenate_matrices
    xaxis, yaxis, zaxis = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    Rx = rotation_matrix(alpha, xaxis)
    Ry = rotation_matrix(beta, yaxis)
    Rz = rotation_matrix(gamma, zaxis)
    R = concatenate_matrices(Rz, Ry, Rx)
    return R


def test_phi5_loss_error_to_zero_batch_size_1():
    rotations = np.array([[1., 2., 3., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi5_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi5_loss_error_to_max_batch_size_1():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4)])
    e = phi5_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, 2.*np.sqrt(2), atol=1e-5)


def test_phi5_loss_error_to_zero_batch_size_2():
    rotations = np.array(
        [[1., 2., 3., 0., 0., 0.], [3., 2., 1., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0., 0., 0., 1.]],
         [[-0.2248451,  0.90238159, -0.36763046,  0.],
          [-0.35017549, -0.42691762, -0.83373765,  0.],
          [-0.90929743, -0.05872664,  0.41198225,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi5_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi5_loss_error_to_max_batch_size_2():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.],
                          [0., 0., np.pi, 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4)] * 2)
    e = phi5_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, 2.*np.sqrt(2), atol=1e-5)


def test_phi3_loss_error_to_zero_batch_size_1():
    rotations = np.array([[1., 2., 3., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi3_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi3_loss_error_to_max_batch_size_1():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4).astype(np.float32)])
    e = phi3_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, np.pi/2, atol=1e-5)


def test_phi3_loss_error_to_zero_batch_size_2():
    rotations = np.array(
        [[1., 2., 3., 0., 0., 0.], [3., 2., 1., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0., 0., 0., 1.]],
         [[-0.2248451,  0.90238159, -0.36763046,  0.],
          [-0.35017549, -0.42691762, -0.83373765,  0.],
          [-0.90929743, -0.05872664,  0.41198225,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi3_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi3_loss_error_to_max_batch_size_2():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.],
                          [0., 0., np.pi, 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4).astype(np.float32)] * 2)
    e = phi3_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, np.pi/2, atol=1e-5)


def test_phi4_loss_error_to_zero_batch_size_1():
    rotations = np.array([[1., 2., 3., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi4_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi4_loss_error_to_max_batch_size_1():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4).astype(np.float32)])
    e = phi4_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, 1., atol=1e-5)


def test_phi4_loss_error_to_zero_batch_size_2():
    rotations = np.array(
        [[1., 2., 3., 0., 0., 0.], [3., 2., 1., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0., 0., 0., 1.]],
         [[-0.2248451,  0.90238159, -0.36763046,  0.],
          [-0.35017549, -0.42691762, -0.83373765,  0.],
          [-0.90929743, -0.05872664,  0.41198225,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi4_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi4_loss_error_to_max_batch_size_2():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.],
                          [0., 0., np.pi, 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4).astype(np.float32)] * 2)
    e = phi4_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, 1., atol=1e-5)


def test_phi2_loss_error_to_zero_batch_size_1():
    rotations = np.array([[1., 2., 3., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi2_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi2_loss_error_to_max_batch_size_1():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4).astype(np.float32)])
    e = phi2_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, np.sqrt(2), atol=1e-5)


def test_phi2_loss_error_to_zero_batch_size_2():
    rotations = np.array(
        [[1., 2., 3., 0., 0., 0.], [3., 2., 1., 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  0.],
          [-0.05872664, -0.42691762,  0.90238159,  0.],
          [-0.90929743, -0.35017549, -0.2248451,  0.],
          [0., 0., 0., 1.]],
         [[-0.2248451,  0.90238159, -0.36763046,  0.],
          [-0.35017549, -0.42691762, -0.83373765,  0.],
          [-0.90929743, -0.05872664,  0.41198225,  0.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = phi2_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_phi2_loss_error_to_max_batch_size_2():
    rotations = np.array([[0., np.pi, 0., 0., 0., 0.],
                          [0., 0., np.pi, 0., 0., 0.]], dtype=np.float32)
    homogeneous_transformation = np.array([np.eye(4).astype(np.float32)] * 2)
    e = phi2_loss(rotations, homogeneous_transformation)
    print(e)
    assert np.allclose(e, np.sqrt(2), atol=1e-5)


def test_get_loss_function_batch_size_2_():
    SO3_loss = get_loss_function('phi2', 'rmse')
    rotations = np.array(
        [[1., 2., 3., 1., 2., 3.], [1., 2., 3., 1., 2., 3.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  1.],
          [-0.05872664, -0.42691762,  0.90238159,  2.],
          [-0.90929743, -0.35017549, -0.2248451,  3.],
          [0., 0., 0., 1.]],
         [[0.41198225, -0.83373765, -0.36763046,  1.],
          [-0.05872664, -0.42691762,  0.90238159,  2.],
          [-0.90929743, -0.35017549, -0.2248451,  3.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = SO3_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)


def test_get_loss_function_batch_size_1():
    SO3_loss = get_loss_function('phi2', 'euclidian')
    rotations = np.array([[1., 2., 3., 1., 2., 3.]], dtype=np.float32)
    homogeneous_transformation = np.array(
        [[[0.41198225, -0.83373765, -0.36763046,  1.],
          [-0.05872664, -0.42691762,  0.90238159,  2.],
          [-0.90929743, -0.35017549, -0.2248451,  3.],
          [0.,  0.,  0.,  1.]]], dtype=np.float32)
    e = SO3_loss(rotations, homogeneous_transformation)
    assert np.allclose(e, 0., atol=1e-5)
