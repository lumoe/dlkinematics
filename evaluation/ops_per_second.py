try:
    from pykdl_utils.kdl_kinematics import KDLKinematics
except ImportError as e:
    print('Unable to use pykdl, make sure to run it in the docker environment')
    pass
from dlkinematics.dlkinematics import DLKinematics
from dlkinematics.urdf import chain_from_urdf_string

import sys
import time
import random

import tensorflow as tf


def get_link_names(robot_string):
    dlkinematics_chain = chain_from_urdf_string(robot_string)

    return (dlkinematics_chain.links[0].name,
            dlkinematics_chain.links[-1].name)


def get_num_joints(chain, root=None, last=None):
    dl_kin = DLKinematics(chain, root, last)
    return dl_kin.num_joints


def get_link_names(chain):
    return (chain.links[0].name,
            chain.links[-1].name)


class catchtime():
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.perf_counter() - self.t


robot_string = open('data/iis_robot.urdf', 'r').read()
chain = chain_from_urdf_string(robot_string)
root, last = get_link_names(chain)

thetas = [random.uniform(-10, 10)
          for _ in range(get_num_joints(chain, root, last))]

batch_sizes = [1, 256, 1024, 4096]
iterations = 2 ** 20

type = sys.argv[1]

if type == 'kdl' and 'KDLKinematics' in dir():
    KDL_kinematic = KDLKinematics(chain, root, last)
    with catchtime() as ct:
        for i in range(iterations):
            KDL_kinematic.forward(thetas, root, last)

    print(f'KDLKinematics took {ct.t}')

if type == 'gpu':
    with tf.device('/gpu:0'):
        for batch_size in batch_sizes:
            dl_kin = DLKinematics(chain, root, last, batch_size)
            thetas = [random.uniform(-10, 10)
                      for _ in range(get_num_joints(chain, root, last) * batch_size)]

            # Measure time for all passes
            with catchtime() as ct:
                for i in range(int(iterations / batch_size)):
                    dl_kin.forward(thetas)

            print(f'KDLKinematics took {ct.t} for batch size: {batch_size}')

if type == 'cpu':
    with tf.device('/cpu:0'):
        for batch_size in batch_sizes:
            dl_kin = DLKinematics(chain, root, last, batch_size)
            thetas = [random.uniform(-10, 10)
                      for _ in range(get_num_joints(chain, root, last) * batch_size)]

            # Measure time for all passes
            with catchtime() as ct:
                for i in range(int(iterations / batch_size)):
                    dl_kin.forward(thetas)

            print(f'KDLKinematics took {ct.t} for batch size: {batch_size}')
