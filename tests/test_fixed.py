import os
import pytest
import numpy as np
import random

from .helper import solve_ours_complete_chain, solve_ros_complete_chain, get_num_joints

robot_string = '''
<robot name="simple_robot">
    <link name="l1">
    </link>
    <link name="l2">
    </link>
    <link name="l3">
    </link>

    <joint name="j1" type="fixed">
        <parent link="l1"/>
        <child link="l2"/>
        <origin xyz="1.0 2.0 3.0" rpy="1.0 0.0 0.0"/>
    </joint>

    <joint name="j2" type="fixed">
        <parent link="l2"/>
        <child link="l3"/>
        <origin xyz="3.0 2.0 1.0" rpy="1.0 2.0 0.0"/>
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
    thetas = []
    _test(thetas)


def test_forward_kinematic_for_complete_chain_batch_size_16():
    thetas = []
    _test(thetas, batch_size=16)


if __name__ == '__main__':
    pytest.main()
