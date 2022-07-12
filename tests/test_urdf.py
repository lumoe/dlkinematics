import pytest
import os

from dlkinematics.urdf import chain_from_urdf_string
from dlkinematics.urdf import chain_from_urdf_file


def _test_string_file_robot_chain_equal(urdf_file_location):
    file_robot = chain_from_urdf_file(urdf_file_location)
    string_robot = chain_from_urdf_string(open(urdf_file_location).read())
    assert file_robot.get_chain(file_robot.links[0].name, file_robot.links[-1].name) == string_robot.get_chain(
        string_robot.links[0].name, string_robot.links[-1].name)


def test_load_human_urdf_file(file='data/human.urdf'):
    _test_string_file_robot_chain_equal(file)


def test_load_b21_urdf_file(file='data/b21.urdf'):
    _test_string_file_robot_chain_equal(file)


def test_load_sample_urdf_file(file='data/sample.urdf'):
    _test_string_file_robot_chain_equal(file)


def test_load_r2d2_urdf_file(file='data/r2d2.urdf'):
    _test_string_file_robot_chain_equal(file)


def test_load_pi_robot_urdf_file(file='data/pi_robot.urdf'):
    _test_string_file_robot_chain_equal(file)


def test__load_simple_urdf_file(file='data/simple.urdf'):
    _test_string_file_robot_chain_equal(file)


def test__load_iis_robot_urdf_file(file='data/iis_robot.urdf'):
    _test_string_file_robot_chain_equal(file)


if __name__ == '__main__':
    pytest.main()
