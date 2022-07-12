import numpy as np
import tensorflow as tf
from dlkinematics.dlkinematics import DLKinematics
from dlkinematics.urdf import chain_from_urdf_file
from generate_dataset import Generator
from tqdm import trange
import pandas as pd
from dlkinematics.training_utils import l2_norm


class IK:
    def __init__(self, urdf, root, tip, learning_rate=1e-3, iterations=50000, batch_size=1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        chain = chain_from_urdf_file(urdf_file)
        self.dlkin = DLKinematics(chain, root, tip, batch_size=self.batch_size)
        self.qs = tf.Variable(np.random.uniform(
            0, 2*np.pi, size=self.dlkin.num_joints*self.batch_size), dtype=tf.float32)

    def optimize(self, x):
        with tf.GradientTape() as g:
            g.watch(self.target)
            g.watch(self.qs)
            fk = self.dlkin.forward(x)
            loss = l2_norm(self.target - fk)
            loss = tf.reduce_mean(loss)

        grad = g.gradient(loss, self.qs)
        self.qs.assign_sub(self.learning_rate*grad)
        return loss

    # @tf.function
    def ik(self, target):
        self.target = self.dlkin.forward(target)
        with trange(self.iterations, unit="iteration", disable=False) as pbar:
            for _ in pbar:
                loss = self.optimize(self.qs)
                pbar.set_postfix(loss=loss.numpy())
                if loss < 1e-3 and self.learning_rate > 1e-5:
                    self.learning_rate /= 10
                if loss < 1e-4:
                    break
        return self.qs, pbar.n


urdf_file, root, tip = '../data/random_chain_24.urdf', 'link0', 'link24'

batch_size = 4
results = None
generator = Generator(urdf_file, root, tip, batch_size)

num_evaluations = 100
cur = 1

for data, fk, angles, translations, xyzrpy in generator.generate_samples_complete():
    ik = IK(urdf_file, root, tip, batch_size=batch_size)
    res, iterations = ik.ik(data.flatten())
    print(f'{data.flatten()}, {res.numpy()}, {iterations}')
    if cur % num_evaluations == 0:
        break
    cur += 1
