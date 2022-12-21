from dlkinematics.dlkinematics import DLKinematics
from dlkinematics.urdf import chain_from_urdf_file, chain_from_urdf_string
from dlkinematics.training_utils import l2_norm

from dlkinematics.training_utils import subsitute_link_with_joint

import random
import numpy as np
import tensorflow as tf

from dlkinematics.evaluation.generate_dataset import Generator
from tqdm import trange


class ParamEstimator:
    def __init__(self, dlkin, dlkin_new, learning_rate=1e-3, iterations=50000, batch_size=1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.dlkin = dlkin
        self.dlkin_new = dlkin_new

        self.qs = tf.Variable(tf.reshape(
            np.array([0, 0, 0, 0, 0, 0]*self.batch_size, dtype=np.float32), (self.batch_size, 6)), dtype=tf.float32)

        for idx, joint in enumerate(self.dlkin_new.urdf.joints):
            if 'imputed' in joint.name:
                break

        self.imputed_index = idx
        self.batch_size = self.dlkin_new.batch_size
        self.num_joints = self.dlkin_new.num_joints

        self.splite_indices = ([self.imputed_index, (self.num_joints - 1) -
                                self.imputed_index] * (self.batch_size))

    def optimize(self, x):
        with tf.GradientTape() as g:
            g.watch(self.target)
            g.watch(self.qs)

            x = tf.cast(x, tf.float32)
            x = tf.split(x, self.splite_indices, axis=-1)

            # Add joint parameters of imputed universal joint into the list
            t = [None] * (self.batch_size * 3)
            t[::3] = x[::2]
            t[1::3] = self.qs
            t[2::3] = x[1::2]
            t = tf.concat(t, axis=-1)

            fk = self.dlkin_new.forward(t)
            loss = l2_norm(self.target - fk)
            loss = tf.reduce_mean(loss)

        grad = g.gradient(loss, self.qs)
        self.qs.assign_sub(self.learning_rate*grad)
        return loss

    # @tf.function
    def step(self, q_in):
        self.target = self.dlkin.forward(q_in)
        with trange(self.iterations, unit="iteration", disable=False) as pbar:
            for _ in pbar:
                loss = self.optimize(q_in)
                pbar.set_postfix(loss=loss.numpy())
                if loss < 1e-3 and self.learning_rate > 1e-5:
                    self.learning_rate /= 10
                if loss < 1e-4:
                    break
        return self.qs, pbar.n


if __name__ == "__main__":
    chain = chain_from_urdf_file('data/iis_robot.urdf')
    root, last = 'world', 'head_realsense_camera_color_optical_frame'

    dl_kin = DLKinematics(chain, root, last)

    new_chain = subsitute_link_with_joint(
        urdf=dl_kin.urdf,
        target_link='base_link',
        root=root,
        last=last
    )

    dl_kin_new = DLKinematics(new_chain, root, last)

    thetas_orig = [1, 2, 3, 4]
    thetas_new = [0, 0, 0, -0.169, 0.0, 0.437, 1, 2, 3, 4]
    # thetas_new = [0, 0, 0, 0, 0, 0, 1, 2, 3, 4]

    # orig = dl_kin.forward(thetas_orig)
    # new = dl_kin_new.forward(thetas_new)

    # print(orig)
    # print(new)
    # print(np.allclose(orig, new))

    expected = np.array([0, 0, 0, -0.169, 0.0, 0.437])

    batch_size = 1
    num_samples = 100

    sample_generator = Generator(
        urdf_file='data/iis_robot.urdf',
        root_link=root,
        end_link=last,
        batch_size=batch_size)

    estimator = ParamEstimator(dl_kin, dl_kin_new, batch_size=batch_size)

    for idx, q_sample in enumerate(sample_generator.joint_samples()):
        res, iterations = estimator.step(q_sample.flatten())
        print(f'{np.round(expected, 3)}')
        print(f'{np.round(estimator.qs.numpy(), 3)}')
        if idx % num_samples == 0:
            break
