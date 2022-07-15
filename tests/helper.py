import os
import io
import random
import pytest
import numpy as np


# Imports for ROS
from dlkinematics.urdf_parser.urdf import URDF
from .pykdl_utils.kdl_kinematics import KDLKinematics

# Imports for dlkinematics
from dlkinematics.urdf import chain_from_urdf_string
from dlkinematics.dlkinematics import DLKinematics


def solve_ros_complete_chain(robot_string, thetas):
    ROS_chain = URDF.from_xml_string(robot_string)

    root, last = get_link_names(robot_string)

    # ROS implementation
    KDL_kinematic = KDLKinematics(ROS_chain, root, last)
    return KDL_kinematic.forward(thetas)


def solve_ours_complete_chain(robot_string, thetas, batch_size):
    dlkinematics_chain = chain_from_urdf_string(robot_string)

    root, last = get_link_names(robot_string)

    # Our implementation
    dl_kin = DLKinematics(dlkinematics_chain, root, last, batch_size)
    return dl_kin.forward(thetas)


def get_num_joints(robot_string, root=None, last=None):
    dlkinematics_chain = chain_from_urdf_string(robot_string)
    if not root:
        root = dlkinematics_chain.links[0].name

    if not last:
        last = dlkinematics_chain.links[-1].name

    dl_kin = DLKinematics(dlkinematics_chain, root, last)
    return dl_kin.num_joints


def get_link_names(robot_string):
    dlkinematics_chain = chain_from_urdf_string(robot_string)

    return (dlkinematics_chain.links[0].name,
            dlkinematics_chain.links[-1].name)


def generate_random_urdf(num_joints):
    assert num_joints > 0
    urdf_s = io.StringIO()
    urdf_s.write(u"""<robot name="simple_robot">\n""")
    for i in range(num_joints + 1):
        urdf_s.write(u"""<link name="l%d"></link>\n""" % (i,))
    for i in range(num_joints):
        urdf_s.write(
            u"""
            <joint name="j{NUM}" type="{JOINT_TYPE}">
                <parent link="l{NUM}"/>
                <child link="l{NUMpp}"/>
                <origin xyz="{XYZ}" rpy="{RPY}"/>
                <axis xyz="{AXIS}"/>
            </joint > \n""".format(NUM=i, NUMpp=i + 1, JOINT_TYPE=_get_random_joint(),
                                   XYZ=_get_random_offset(), RPY=_get_random_offset(),
                                   AXIS=_get_random_axis()))
    urdf_s.write(u"</robot>\n")
    urdf_s.seek(0)
    return urdf_s.read()


def _get_random_joint():
    return ['fixed',
            'revolute',
            'continuous',
            'prismatic'][random.randint(0, 3)]


def _get_random_offset():
    return "{: .2f} {: .2f} {: .2f}".format(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))


def _get_random_axis():
    axis = np.array([0., 0., 1.])
    random.shuffle(axis)
    return "{}".format(axis)[1:-1]
