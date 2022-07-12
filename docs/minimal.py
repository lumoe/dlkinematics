import tensorflow as tf
from dlkinematics.urdf import chain_from_urdf_file
from dlkinematics.dlkinematics import DLKinematics

# Load URDF
chain = chain_from_urdf_file('data/human.urdf')

# Get first and last link from chain
root_link, last_link, batch_size = "human_base", "human_spine_2", 2

# Create DLKinematics
dlkinematics = DLKinematics(chain, root_link, last_link, batch_size)

# Joint configuartion
thetas = tf.Variable([1., 2., 3., 4.], dtype=tf.float32)

# Forward pass
with tf.GradientTape() as tape:
    result = dlkinematics.forward(thetas)

print(result)
print(tape.gradient(result, thetas))


# Output
# tf.Tensor(
# tf.Tensor(
#     [[[-0.4161468   0.          0.9092974   0.]
#       [0.7651474   0.5403023   0.35017547  0.]
#       [-0.49129546  0.84147096 - 0.22484507  0.07262489]
#       [0.          0.          0.          1.]]

#       [[-0.6536436   0. - 0.7568025   0.]
#        [-0.10679997 - 0.9899925   0.09224219  0.]
#        [-0.7492288   0.14112     0.6471023   0.07262489]
#        [0.          0.          0.          1.]]], shape=(2, 4, 4), dtype=float32)
# tf.Tensor([1.5302947 - 1.1769228 - 1.0435438 -1.4922143], shape=(4,), dtype=float32)
