import os
import pytest
import numpy as np

# Imports for dlkinematics
from dlkinematics.urdf import chain_from_urdf_file
from dlkinematics.dlkinematics import DLKinematics

# Imports for ROS
from dlkinematics.urdf_parser.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics

def _test_chain_from_urdf_file(urdf_file_location):
    # ROS implementation 
    ROS_chain = URDF.from_xml_string(open(urdf_file_location).read())
    root = ROS_chain.links[0].name
    last = ROS_chain.links[-1].name

    KDL_kinematic = KDLKinematics(ROS_chain, root, last)

    # Our implementation
    dlkinematics_chain = chain_from_urdf_file(urdf_file_location)
    root = dlkinematics_chain.links[0].name
    last = dlkinematics_chain.links[-1].name

    dl_kin = DLKinematics(dlkinematics_chain, root, last)

    # dl_kin should return a chain without static joints
    assert [a.joint.name for a in dl_kin.get_chain()] == KDL_kinematic.get_joint_names()


def test_chain_from_urdf_file_human(file='data/human.urdf'):
    _test_chain_from_urdf_file(file)

def test_chain_from_urdf_file_b21(file='data/b21.urdf'):
    _test_chain_from_urdf_file(file)

def test_chain_from_urdf_file_sample(file='data/sample.urdf'):
    _test_chain_from_urdf_file(file)

def test_chain_from_urdf_file_r2d2(file='data/r2d2.urdf'):
    _test_chain_from_urdf_file(file)

def test_chain_from_urdf_file_pi_robot(file='data/pi_robot.urdf'):
    _test_chain_from_urdf_file(file)

def test_chain_from_urdf_file_simple(file='data/simple.urdf'):
    _test_chain_from_urdf_file(file)
