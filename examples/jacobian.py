import tensorflow as tf
from dlkinematics.urdf import chain_from_urdf_file
from dlkinematics.dlkinematics import DLKinematics
from dlkinematics.training_utils import pose_from_matrix

# Load URDF
chain = chain_from_urdf_file("../data/human.urdf")
# Create DLKinematics
dlkinematics = DLKinematics(chain, base_link="link0", end_link="link3", batch_size=2)
# Joint configuartion
thetas = tf.convert_to_tensor([[1., 2., 3.], [3., 4., 6.]], dtype=tf.float32)
thetas = tf.reshape(thetas, shape=-1)
# Forward pass
with tf.GradientTape(persistent=True) as tape:
    tape.watch(thetas)
fk_res = dlkinematics.forward(thetas)
p = pose_from_matrix(fk_res)
jacobian = tape.batch_jacobian(p, thetas)
