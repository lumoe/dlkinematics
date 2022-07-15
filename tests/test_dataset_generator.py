import numpy as np


# Imports for ROS
from dlkinematics.urdf_parser.urdf import URDF
from .pykdl_utils.kdl_kinematics import KDLKinematics  # noqa
# from evaluation.generate_dataset import generate_dataset


from transformations import euler_from_matrix, translation_from_matrix

# Needs rewrite of dataset generator to be used as a module


# def _test_simple_urdf():
#     X_test = pd.read_csv('evaluation/simple_urdf_small_new/X_test.csv')
#     y_test = pd.read_csv('evaluation/simple_urdf_small_new/y_test.csv')
#     urdf_file = 'data/simple.urdf'

#     with open(urdf_file, 'r') as urdf:
#         ROS_chain = URDF.from_xml_string(urdf.read())

#     root, last = ('base_link', 'third_link')

#     # ROS implementation
#     KDL_kinematic = KDLKinematics(ROS_chain, root, last)
#     for idx, row in X_test.iterrows():
#         matrix = KDL_kinematic.forward(y_test.iloc[idx].values)
#         rot = euler_from_matrix(matrix)
#         tra = translation_from_matrix(matrix)

#         assert np.allclose(rot, X_test.iloc[idx].values[:3], atol=1e-4)
#         assert np.allclose(tra, X_test.iloc[idx].values[3:], atol=1e-4)
