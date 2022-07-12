import pytest
import os
import random

import numpy as np


from .helper import solve_ours_complete_chain, solve_ros_complete_chain, get_num_joints


def _test_kinematic_from_urdf_file(urdf_file_location, thetas, batch_size=1):
    robot_string = open(urdf_file_location).read()

    dl_kin_results = solve_ours_complete_chain(
        robot_string, thetas, batch_size=batch_size)

    num_joints = get_num_joints(robot_string)

    for i, dl_kin_result in enumerate(dl_kin_results):
        ROS_result = solve_ros_complete_chain(
            robot_string, thetas[num_joints*i:num_joints + num_joints*i])
        assert np.allclose(dl_kin_result.numpy(), ROS_result, atol=1e-5)


def test_kinematic_from_urdf_file_human(file='data/human.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_b21(file='data/b21.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_sample(file='data/sample.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_r2d2(file='data/r2d2.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_pi_robot(file='data/pi_robot.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_simple(file='data/simple.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_iis_robot(file='data/iis_robot.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_random_chain_1(file='data/random_chain_1.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_random_chain_6(file='data/random_chain_6.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_random_chain_12(file='data/random_chain_12.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)


def test_kinematic_from_urdf_file_random_chain_24(file='data/random_chain_24.urdf'):
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(open(file).read()))]
    _test_kinematic_from_urdf_file(file, thetas)
