from __future__ import absolute_import

from .urdf_parser.urdf import Robot

from lxml import etree

SUPPORTED_JOINT_TYPES = [
    'fixed',
    'revolute',
    'continuous',
    'prismatic',
    'floating',
    'planar'
]


def chain_from_urdf_file(filepath):
    with open(filepath, 'r') as input_file:
        ch = chain_from_urdf_string(input_file.read())

    return ch


def chain_from_urdf_string(urdf_string):
    return _build_chain(urdf_string)


def _build_chain(urdf_string):
    urdf_string = urdf_string.encode()
    robot = Robot.from_xml_string(urdf_string)
    for joint in robot.joints:
        if joint.type not in SUPPORTED_JOINT_TYPES:
            raise NotImplementedError(
                'Joint type {0} is not supported'.format(joint.type))
    return robot


class UrdfFileError(Exception):
    def __init__(self, *args):
        if args:
            self.error_message = args[0]
        else:
            self.error_message = None

    def __str__(self):
        if self.error_message:
            return 'UrdfNotValidError: {0}'.format(self.error_message)
        else:
            return 'UrdfNotValidError: URDF file is not valid'
