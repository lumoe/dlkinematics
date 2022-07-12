import random
import os
import pytest
import numpy as np
import random


from .helper import solve_ours_complete_chain, solve_ros_complete_chain, get_num_joints

# Imports for dlkinematics
from dlkinematics.urdf import chain_from_urdf_string
from dlkinematics.dlkinematics import DLKinematics

robot_string = '''
<robot name="simple_robot">
    <link name="l1">
    </link>
    <link name="l2">
    </link>
    <link name="l3">
    </link>
    <link name="l4">
    </link>

    <joint name="j1" type="prismatic">
        <parent link="l1"/>
        <child link="l2"/>
        <origin xyz="1.0 2.0 3.0" rpy="1.0 0.2 0.0"/>
        <axis xyz="1.0 0.0 0.0"/>
    </joint>

    <joint name="j2" type="prismatic">
        <parent link="l2"/>
        <child link="l3"/>
        <origin xyz="3.0 2.0 1.0" rpy="1.0 2.0 0.0"/>
        <axis xyz="0.0 1.0 0.0"/>
    </joint>
    <joint name="j3" type="prismatic">
        <parent link="l3"/>
        <child link="l4"/>
        <origin xyz="3.0 2.0 1.0" rpy="2.0 1.0 0.0"/>
        <axis xyz="0.0 0.0 1.0"/>
    </joint>
</robot>
'''


def _test(thetas, batch_size=1):
    dl_kin_results = solve_ours_complete_chain(
        robot_string, thetas, batch_size=batch_size)

    num_joints = get_num_joints(robot_string)

    for i, dl_kin_result in enumerate(dl_kin_results):
        ROS_result = solve_ros_complete_chain(
            robot_string, thetas[num_joints*i:num_joints + num_joints*i])
        assert np.allclose(dl_kin_result.numpy(), ROS_result, atol=1e-5)


def test_forward_kinematic_for_complete_chain():
    thetas = [0.] * get_num_joints(robot_string)
    _test(thetas)


def test_forward_kinematic_for_complete_chain_with_theta():
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(robot_string))]
    _test(thetas)


def test_forward_kinematic_for_complete_chain_with_small_theta():
    thetas = [0.0001] * get_num_joints(robot_string)
    _test(thetas)


def test_forward_kinematic_for_complete_chain_with_high_theta():
    thetas = [1000.] * get_num_joints(robot_string)
    _test(thetas)


def test_forward_kinematic_for_complete_chain_batch_size_16():
    thetas = [random.uniform(-10, 10)
              for _ in range(get_num_joints(robot_string) * 16)]
    _test(thetas, batch_size=16)


def test_generated_indices():
    ROS_chain = chain_from_urdf_string(robot_string)
    root, last = ('l1', 'l3')
    # ROS implementation
    KDL_kinematic = DLKinematics(ROS_chain, root, last)
    assert np.allclose(KDL_kinematic.theta_indices, [[0, 0, 3], [1, 0, 4]])


if __name__ == '__main__':
    pytest.main()
