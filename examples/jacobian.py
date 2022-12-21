import tensorflow as tf

from dlkinematics.dlkinematics import DLKinematics
from dlkinematics.tf_transformations import pose_from_matrix
from dlkinematics.urdf import chain_from_urdf_file

# Load URDF
chain = chain_from_urdf_file("data/random_chain_6.urdf")
# Create DLKinematics
dlkinematics = DLKinematics(chain, base_link="link0", end_link="link3", batch_size=2)
# Joint configuartion
thetas = tf.convert_to_tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], dtype=tf.float32)

# Forward pass
with tf.GradientTape(persistent=True) as tape:
    tape.watch(
        thetas
    )  # this is important to get the gradient with respect to thetas which is not a variable.
    fk_res = dlkinematics.forward(tf.reshape(thetas, (-1,)))
    poses = pose_from_matrix(fk_res)
jacobians = tape.batch_jacobian(poses, thetas)

for theta, pose, jacobian in zip(thetas, poses, jacobians):
    print(f"Thetas: {theta}")
    print(f"Pose: (x, y, z, alpha, beta, gamma): {pose}")
    print(f"Jacobian: {jacobian}")
