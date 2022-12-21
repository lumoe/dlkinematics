import argparse
import sys
import os
import json
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.python.framework import ops
from tensorflow.keras.callbacks import ModelCheckpoint

from datetime import datetime

from dlkinematics.urdf import chain_from_urdf_file  # noqa
from dlkinematics.dlkinematics import DLKinematics  # noqa
from dlkinematics.training_utils import ForwardKinematics, get_loss_function, num_joints_for_urdf
from generate_dataset import Generator


ops.reset_default_graph()
tf.keras.backend.clear_session()

# tf.debugging.enable_check_numerics()
tf.debugging.disable_check_numerics()

# class IsFinite(layers.Layer):
#     def __init__(self, label, **kwargs):
#         self.label = label
#         super(IsFinite, self).__init__(**kwargs)

#     def call(self, inp):
#         # tf.debugging.assert_all_finite(inp, self.label)
#         # tf.debugging.is_numeric_tensor(tf.shape(inp))
#         # tf.debugging.assert_shapes
#         # tf.autograph.trace(inp)
#         # tf.print(inp)
#         return inp

#     def get_config(self):
#         config = super().get_config().copy()
#         return config

MODEL_PATH_PREFIX = 'models'
def run_exp(name, epochs, steps, width, depth, urdf, root, tip, rloss, tloss):
    steps_per_epoch = steps
    # Hyperparameter definition
    learning_rate = 1e-5
    hidden_layers = [2 ** width] * depth
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = get_loss_function(rloss, tloss)
    # loss = My_SO3_loss
    batch_size = 32
    base_link, end_link = (root, tip)
    num_joints = num_joints_for_urdf(urdf, root=root, tip=tip)

    generator = Generator(urdf, base_link, end_link, batch_size)

    dataset = tf.data.Dataset.from_generator(
        generator.generate_samples_fk,
        (tf.float32, tf.float32),
        (tf.TensorShape([batch_size, 6]), tf.TensorShape([batch_size, 6])),
    )

    validation_data =  tf.data.Dataset.from_generator(
        generator.generate_samples_fk,
        (tf.float32, tf.float32),
        (tf.TensorShape([batch_size, 6]), tf.TensorShape([batch_size, 6])),
    )


    logdir = f"logs/exp/{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


    model = keras.Sequential()
    input_layer = layers.Input(shape=(6,))
    model.add(input_layer)


    for i in range(len(hidden_layers)):
        layer = layers.Dense(
            units=hidden_layers[i],
            kernel_initializer='random_normal',
            use_bias=True,
            bias_initializer='random_normal',
            activation='relu'
        )
        model.add(layer)

    output_layer = layers.Dense(
        units=num_joints,
        kernel_initializer='random_normal',
        use_bias=True,
        bias_initializer='random_normal',
        activation='linear'
    )
    model.add(output_layer)

    FK_layer = ForwardKinematics(urdf, base_link, end_link, batch_size)
    model.add(FK_layer)

    model.compile(loss=loss, optimizer=optimizer, run_eagerly=False)
    model.summary()


    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        histogram_freq=1, update_freq='batch', log_dir=logdir)


    checkpoint = ModelCheckpoint(f'{MODEL_PATH_PREFIX}/{name}-checkpoint.h5', monitor='loss', verbose=1,
        save_best_only=True, mode='auto', period=1)

    _ = model.fit(
        x=dataset,
        validation_data=validation_data,
        validation_steps=10,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
        callbacks=[tensorboard_callback, checkpoint],
    )


    model.save(os.path.join(f'{MODEL_PATH_PREFIX}/{name}-trained.h5'))

# def run_exp(name, epochs, steps_per_epoch, width, depth, urdf, root, tip):
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Randomly generate kinematic chain and create URDF')
    parser.add_argument('name', type=str,
                        help='Name of experiment')
    parser.add_argument('--width', type=int,
                        help='Width of the hidden layer')
    parser.add_argument('--depth', type=int,
                        help='Number of hidden layers')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs')
    parser.add_argument('--steps', type=int,
                        help='Numer of steps per epoch')
    parser.add_argument('--urdf', type=str,
                        help='Location of URDF file')
    parser.add_argument('--root', type=str,
                        help='Name of root link')
    parser.add_argument('--tip', type=str,
                        help='Name of tip link')
    parser.add_argument('--rloss', type=str,
                        help='Function for rotation loss')
    parser.add_argument('--tloss', type=str,
                        help='Function for translation loss')
    args = parser.parse_args()
    if os.path.exists(f'{MODEL_PATH_PREFIX}/{args.name}.json'):
        raise Exception('Experiment already exists')
    with open(f'{MODEL_PATH_PREFIX}/{args.name}.json', 'w') as summary:
        summary.write(json.dumps(vars(args)))
    run_exp(**vars(args))
    