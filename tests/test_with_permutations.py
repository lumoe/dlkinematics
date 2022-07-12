import os
import pytest
import itertools
import numpy as np
import random

from .helper import solve_ours_complete_chain, solve_ros_complete_chain, get_num_joints, generate_random_urdf

# Imports for dlkinematics
from dlkinematics.urdf import chain_from_urdf_string
from dlkinematics.dlkinematics import DLKinematics


def _test(thetas, robot_string, batch_size=1):
    dl_kin_results = solve_ours_complete_chain(
        robot_string, thetas, batch_size=batch_size)

    num_joints = get_num_joints(robot_string)

    for i, dl_kin_result in enumerate(dl_kin_results):
        ROS_result = solve_ros_complete_chain(
            robot_string, thetas[num_joints*i:num_joints + num_joints*i])
        assert np.allclose(dl_kin_result.numpy(), ROS_result, atol=1e-5)


def test_random_urdf():
    num_joints = 1
    batch_size = 64
    robot_string = generate_random_urdf(num_joints)
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(robot_string) * batch_size)]
    _test(thetas, robot_string, batch_size=batch_size)


# def test_robot_size_and_batch_size_permutations():
#     allbatchsizes = [64, 512, 1024, 2048, 8192]
#     alljoints = [1, 2, 7, 20, 100, 200, 500]
#     combinations = list(itertools.product(allbatchsizes, alljoints))
#     for batch_size, num_joints in combinations:
#         robot_string = generate_random_urdf(num_joints)
#         thetas = [random.uniform(-10, 10)
#                   for _ in range(get_num_joints(robot_string) * batch_size)]
#         _test(thetas, robot_string, batch_size=batch_size)
