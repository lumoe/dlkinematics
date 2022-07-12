from dlkinematics.dlkinematics import DLKinematics
from dlkinematics.urdf_parser.urdf import URDF
import numpy as np

robot_string = '''
<robot name="simple_robot">
    <link name="l1">
    </link>
    <link name="l2">
    </link>
    <link name="l3">
    </link>

    <joint name="j1" type="continuous">
        <parent link="l1"/>
        <child link="l2"/>
        <origin xyz="1.0 2.0 3.0" rpy="1.0 0.2 0.0"/>
        <axis xyz="1.0 0.0 0.0"/>
    </joint>

    <joint name="j2" type="continuous">
        <parent link="l2"/>
        <child link="l3"/>
        <origin xyz="3.0 2.0 1.0" rpy="1.0 2.0 0.0"/>
        <axis xyz="0.0 1.0 0.0"/>
    </joint>
</robot>
'''


def test_generated_indices():
    ROS_chain = URDF.from_xml_string(robot_string)
    root, last = ('l1', 'l3')
    # ROS implementation
    KDL_kinematic = DLKinematics(ROS_chain, root, last)
    assert np.allclose(KDL_kinematic.theta_indices, [[0, 0, 0], [1, 0, 1]])


if __name__ == '__main__':
    test_generated_indices()
