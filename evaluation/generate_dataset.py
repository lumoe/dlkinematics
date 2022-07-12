from transformations import euler_from_matrix, translation_from_matrix
import itertools
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import progressbar

from dlkinematics.urdf import chain_from_urdf_string  # noqa
from dlkinematics.dlkinematics import DLKinematics  # noqa
from tensorflow_graphics.geometry.transformation.euler import from_rotation_matrix as euler_from_rotation_matrix
from tensorflow.keras.utils import Sequence

RADIANS = True


class Generator(object):
    def __init__(self, urdf_file, root_link, end_link, batch_size):
        self.batch_size = batch_size
        self.end_link = end_link
        with open(urdf_file) as inputfile:
            self.robot = chain_from_urdf_string(inputfile.read())

        self.chain = self.robot.get_chain(
            root_link, end_link, joints=True, links=False, fixed=True)

        self.chain = [self.robot.joint_map.get(c) for c in self.chain]
        self.dlkin = DLKinematics(
            self.robot, root_link, end_link, batch_size=batch_size)

    def _get_data(self):
        data = list()
        for joint in self.chain:
            lower, upper = (0, 0)
            if joint.type == 'continuous':
                data.append(
                    np.random.uniform(0, 2*np.pi, size=self.batch_size)
                )
            elif joint.type == 'revolute':
                lower, upper = (joint.limit.lower, joint.limit.upper)
                data.append(
                    np.random.uniform(lower, upper, size=self.batch_size)
                )
            elif joint.type == 'static':
                pass
            elif joint.type == 'prismatic':
                data.append(
                    np.random.uniform(joint.limit.lower,
                                      joint.limit.upper, size=self.batch_size)
                )
        data = np.stack(data, axis=1)
        return data

    def generate_samples(self):
        while 1:
            data = self._get_data()

            fk = self.dlkin.forward(data.flatten())
            angles = fk[:, :3, :3]
            angles = euler_from_rotation_matrix(angles)
            translations = tf.reshape(fk[:, :3, 3:], (self.batch_size, 3))
            xyzrpy = tf.reshape(
                tf.stack([angles, translations], axis=1), (self.batch_size, 6))
            X = tf.cast(xyzrpy, tf.float32)
            y = tf.cast(data, tf.float32)
            yield X, y

    def generate_samples_fk(self):
        while 1:
            data = self._get_data()

            fk = self.dlkin.forward(data.flatten())
            angles = fk[:, :3, :3]
            angles = euler_from_rotation_matrix(angles)
            translations = tf.reshape(fk[:, :3, 3:], (self.batch_size, 3))

            xyzrpy = tf.reshape(
                tf.stack([angles, translations], axis=1), (self.batch_size, 6))
            X = tf.cast(xyzrpy, tf.float32)
            yield X, X

    def generate_samples_complete(self):
        while 1:
            data = self._get_data()
            fk = self.dlkin.forward(data.flatten())
            angles = fk[:, :3, :3]
            angles = euler_from_rotation_matrix(angles)
            translations = tf.reshape(fk[:, :3, 3:], (self.batch_size, 3))
            xyzrpy = tf.reshape(
                tf.stack([angles, translations], axis=1), (self.batch_size, 6))
            yield data, fk, angles, translations, xyzrpy

    def joint_samples(self):
        while 1:
            data = self._get_data()
            yield data


def generate_dataset(infile, outdir, num_samples, root_link, end_link, save_files=True):
    """
    Generate sample data given sample URDF
    """

    with open(infile) as inputfile:
        robot = chain_from_urdf_string(inputfile.read())

    """
    Generates joint configuraton for each joint dependig on type
    """
    chain = robot.get_chain(root_link, end_link,
                            joints=True, links=False, fixed=True)
    chain = [robot.joint_map.get(c) for c in chain]
    data = list()
    for joint in chain:
        lower, upper = (0, 0)
        if joint.type == 'continuous':
            # 360deg
            data.append(
                np.random.uniform(0, 2*np.pi, size=num_samples)
            )
        elif joint.type == 'revolute':
            lower, upper = (joint.limit.lower, joint.limit.upper)
            data.append(
                np.random.uniform(lower, upper, size=num_samples)
            )
        elif joint.type == 'static':
            pass
        elif joint.type == 'prismatic':
            data.append(
                np.random.uniform(joint.limit.lower,
                                  joint.limit.upper, size=num_samples)
            )

    """
    Creates cartesian product for each joint configuration
    """
    crossproduct = itertools.product(*data)

    # Create DataFrame for joint configuration
    df = pd.DataFrame(crossproduct, columns=[j.name for j in chain])

    fk_euler = list()
    fk_translation = list()
    root, end = (root_link,
                 end_link)
    print('Generating chain from {} to {}'.format(root, end))

    dlkin = DLKinematics(robot, root, end, batch_size=1)

    import tensorflow as tf
    tf.executing_eagerly()
    for _, row in progressbar.progressbar(df.iterrows()):
        thetas = row.values
        forward_matrix = dlkin.forward(thetas)
        forward_matrix = np.array(forward_matrix)

        euler = euler_from_matrix(forward_matrix[0])
        translation = translation_from_matrix(forward_matrix[0])

        fk_euler.append(euler)
        fk_translation.append(translation)

    # Set values and convert them to positive angles
    df['rot_x'] = [r[0] for r in fk_euler]
    df['rot_y'] = [r[1] for r in fk_euler]
    df['rot_z'] = [r[2] for r in fk_euler]

    df['translate_x'] = [t[0] for t in fk_translation]
    df['translate_y'] = [t[1] for t in fk_translation]
    df['translate_z'] = [t[2] for t in fk_translation]

    # Convert to degrees
    if not RADIANS:
        df['rot_x'] = np.degrees(df['rot_x'])
        df['rot_y'] = np.degrees(df['rot_y'])
        df['rot_z'] = np.degrees(df['rot_z'])
        for j in chain.joints:
            df[j.name] = np.degrees(df[j.name])

    # df = df.sample(frac=1).reset_index(drop=True)
    joint_columns = [j.name for j in chain]
    X_train, X_test, y_train, y_test = train_test_split(
        df[df.columns[-6:]], df[df.columns[0:len(joint_columns)]], test_size=0.20, random_state=1337)

    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train, y_train, test_size=0.20, random_state=1337)

    if save_files:
        X_train.to_csv(os.path.join(outdir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(outdir, 'X_test.csv'), index=False)
        X_validation.to_csv(os.path.join(
            outdir, 'X_validation.csv'), index=False)

        y_train.to_csv(os.path.join(outdir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(outdir, 'y_test.csv'), index=False)
        y_validation.to_csv(os.path.join(
            outdir, 'y_validation.csv'), index=False)
    else:
        return X_test, y_test


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate dataset based on URDF file')
    parser.add_argument('infile', type=str, help='URDF input file')
    parser.add_argument('outdir', type=str,
                        help='Directory to write results to')
    parser.add_argument('num_samples', type=int,
                        help='Number of samples to generate for each joint')
    parser.add_argument('root_link', type=str,
                        help='Name of first link in kinematic chain')
    parser.add_argument('end_link', type=str,
                        help='Name of last link in kinematic chain')
    args = parser.parse_args()
    # infile, outdir, num_samples = ("data/simple.urdf", "out", "10")
    generate_dataset(args.infile, args.outdir, args.num_samples,
                     args.root_link, args.end_link)
