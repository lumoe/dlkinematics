# run this as python -m examples.model_identification
import numpy as np
import tensorflow as tf

from dlkinematics.dlkinematics import DLKinematics
from dlkinematics.tf_transformations import pose_from_matrix
from dlkinematics.training_utils import subsitute_link_with_joint
from dlkinematics.urdf import chain_from_urdf_file
from evaluation.estimate_model_params import ParamEstimator
from evaluation.generate_dataset import Generator

# Loading kinematic chain ..
path_to_urdf = "data/random_chain_6.urdf"
chain = chain_from_urdf_file(path_to_urdf)
root = "link0"
last = "link6"
target_link = "link3"
batch_size = 128
num_samples = 50

dl_kin = DLKinematics(chain, root, last)

# Replace link with universal joint
new_chain = subsitute_link_with_joint(
    urdf=dl_kin.urdf, target_link=target_link, root=root, last=last
)
dl_kin_new = DLKinematics(new_chain, root, last)

sample_generator = Generator(
    urdf_file=path_to_urdf, root_link=root, end_link=last, batch_size=batch_size
)

estimator = ParamEstimator(dl_kin, dl_kin_new, batch_size=batch_size)

# Optimization loop
for idx, q_sample in enumerate(sample_generator.joint_samples()):
    res, iterations = estimator.step(q_sample.flatten())
    if idx % num_samples == 0:
        break

print(f"{np.round(estimator.qs.numpy(), 3)}")
