import argparse

import tensorflow_graphics as tfg
from tensorflow import keras
import tensorflow as tf
import tensorflow_graphics.geometry.transformation.rotation_matrix_3d as tfg_rotation_matrix_3d
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import progressbar

from dlkinematics.urdf import chain_from_urdf_file  # noqa
from dlkinematics.dlkinematics import DLKinematics  # noqa
from evaluation.generate_dataset import Generator
from dlkinematics.training_utils import ForwardKinematics, get_loss_function


def visualise(urdf, model, samples, root, tip, batch_size=10):
    modelname = model
    SO3_loss = get_loss_function('rmse', 'rmse')
    model = keras.models.load_model(model, custom_objects={'ForwardKinematics': ForwardKinematics, 'SO3_loss': SO3_loss})
    if 'forward_kinematics' in model.layers[-1].name:
        model.pop()

    generator = Generator(urdf, root, tip, batch_size)

    predictions = list()
    angles = list()
    translations = list()
    xyzrpys = list()
    for idx, sample in progressbar.progressbar(enumerate(generator.generate_samples_complete())):
        data, fk, angle, translation, xyzrpy = sample
        predicted = model.predict(tf.cast(xyzrpy, dtype=tf.float32))
        predictions.append(predicted)
        angles.append(angle)
        xyzrpys.append(xyzrpys)
        translations.append(translation)
        if idx % samples == 0 and idx != 0:
            break

    translations = tf.cast(translations, dtype=tf.float32)
    angles = tf.cast(angles, dtype=tf.float32)

    robot = chain_from_urdf_file(urdf)
    dl_kin = DLKinematics(robot, root, tip, batch_size=batch_size)

    FK_result = list()

    for y in progressbar.progressbar(predictions):
        FK_result.append(dl_kin.forward(y.flatten()))

    FK_result = np.array(FK_result)
    # print(np.reshape(FK_result.flatten(), [(samples+1)*batch_size, 4,4]))
    # return
    FK_result = np.reshape(FK_result, [(samples+1)*batch_size, 4, 4])
    angles_estimated = tfg.geometry.transformation.euler.from_rotation_matrix(
        FK_result[:, :3, :3])



    angles = np.reshape(angles, [batch_size*(samples+1), 3])
    rot_x_diff = angles[:, 0].flatten() - angles_estimated[:, 0]
    rot_y_diff = angles[:, 1].flatten() - angles_estimated[:, 1]
    rot_z_diff = angles[:, 2].flatten() - angles_estimated[:, 2]

    tra_x = translations[:, :, 0].numpy().flatten()
    tra_y = translations[:, :, 1].numpy().flatten()
    tra_z = translations[:, :, 2].numpy().flatten()

    tra_x_diff = translations[:, :, 0].numpy().flatten() - FK_result[:, :3, 3:].reshape(FK_result.shape[0], 3)[:, :1].flatten()
    tra_y_diff = translations[:, :, 1].numpy().flatten() - FK_result[:, :3, 3:].reshape(FK_result.shape[0], 3)[:, 1:2].flatten()
    tra_z_diff = translations[:, :, 2].numpy().flatten() - FK_result[:, :3, 3:].reshape(FK_result.shape[0], 3)[:, 2:3].flatten()


        # angles = np.reshape(angles, )
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.quiver(translations[:, :, 0].numpy().flatten(), translations[:, :, 1].numpy().flatten(), translations[:, :, 2].numpy().flatten(), angles[:, 0].flatten(), angles[:, 1].flatten(), angles[:, 2].flatten(), length=0.1, normalize=True)
    ax.quiver(FK_result[:, :3, 3:].reshape(FK_result.shape[0], 3)[:, :1].flatten(),
            FK_result[:, :3, 3:].reshape(FK_result.shape[0], 3)[:, 1:2].flatten(),
            FK_result[:, :3, 3:].reshape(FK_result.shape[0], 3)[:, 2:3].flatten(), 
            angles_estimated[:, 0], angles_estimated.numpy()[:, 1],
            angles_estimated[:, 2], length=0.1, normalize=True, color='red')
    plt.show()
    # fig = plt.figure()
    # ax = fig.gca()
    # ax.plot(rot_x_diff, label='Rotation X')
    # ax.plot(rot_y_diff, label='Rotation Y')
    # ax.plot(rot_z_diff, label='Rotation Z')
    # ax.legend()

    # fig = plt.figure()
    # ax = fig.gca()
    # ax.plot(tra_x_diff, label='X')
    # ax.plot(tra_y_diff, label='Y')
    # ax.plot(tra_z_diff, label='Z')
    # ax.legend()
    # plt.show()

    rot_x_diff = np.where(rot_x_diff > np.pi, rot_x_diff - np.pi*2, rot_x_diff)
    rot_x_diff = np.where(rot_x_diff < -np.pi, rot_x_diff + np.pi*2, rot_x_diff)

    rot_y_diff = np.where(rot_y_diff > np.pi, rot_y_diff - np.pi*2, rot_y_diff)
    rot_y_diff = np.where(rot_y_diff < -np.pi, rot_y_diff + np.pi*2, rot_y_diff)

    rot_z_diff = np.where(rot_z_diff > np.pi, rot_z_diff - np.pi*2, rot_z_diff)
    rot_z_diff = np.where(rot_z_diff < -np.pi, rot_z_diff + np.pi*2, rot_z_diff)

    rot_x_diff = np.nan_to_num(rot_x_diff)
    rot_y_diff = np.nan_to_num(rot_y_diff)
    rot_z_diff = np.nan_to_num(rot_z_diff)

    # print(f'Mean error for rotation x: {np.mean(rot_x_diff)}rad = {np.rad2deg(np.mean(rot_x_diff))}°')
    # print(f'Mean error for rotation y: {np.mean(rot_y_diff)}rad = {np.rad2deg(np.mean(rot_y_diff))}°')
    # print(f'Mean error for rotation z: {np.mean(rot_z_diff)}rad = {np.rad2deg(np.mean(rot_z_diff))}°')

    # print(f'Standard Deviation for rotation error x: {np.std(rot_x_diff)}rad = {np.rad2deg(np.std(rot_x_diff))}°')
    # print(f'Standard Deviation for rotation error y: {np.std(rot_y_diff)}rad = {np.rad2deg(np.std(rot_y_diff))}°')
    # print(f'Standard Deviation for rotation error z: {np.std(rot_z_diff)}rad = {np.rad2deg(np.std(rot_z_diff))}°')

    # print(f'Max error for rotation x: {np.amax(np.abs(rot_x_diff))}rad = {np.rad2deg(np.amax(np.abs(rot_x_diff)))}°')
    # print(f'Max error for rotation y: {np.amax(np.abs(rot_y_diff))}rad = {np.rad2deg(np.amax(np.abs(rot_y_diff)))}°')
    # print(f'Max error for rotation z: {np.amax(np.abs(rot_z_diff))}rad = {np.rad2deg(np.amax(np.abs(rot_z_diff)))}°')

    # print(f'Mean error for translation x: {np.mean(tra_x_diff)} => {(np.mean(tra_x_diff) / np.amax(tra_x))*100}%')
    # print(f'Mean error for translation y: {np.mean(tra_y_diff)} => {(np.mean(tra_y_diff) / np.amax(tra_y))*100}%')
    # print(f'Mean error for translation z: {np.mean(tra_z_diff)} => {(np.mean(tra_z_diff) / np.amax(tra_z))*100}%')

    # print(f'Standard Deviation for translation error x: {np.std(tra_x_diff)} => {(np.std(tra_x_diff) / np.amax(tra_x))*100}%')
    # print(f'Standard Deviation for translation error y: {np.std(tra_y_diff)} => {(np.std(tra_y_diff) / np.amax(tra_y))*100}%')
    # print(f'Standard Deviation for translation error z: {np.std(tra_z_diff)} => {(np.std(tra_z_diff) / np.amax(tra_z))*100}%')
    
    # print(f'Max error for translation x: {np.amax(np.abs(tra_x_diff))} => {(np.amax(np.abs(tra_x_diff)) / np.amax(tra_y))*100}%')
    # print(f'Max error for translation y: {np.amax(np.abs(tra_y_diff))} => {(np.amax(np.abs(tra_y_diff)) / np.amax(tra_y))*100}%')
    # print(f'Max error for translation z: {np.amax(np.abs(tra_z_diff))} => {(np.amax(np.abs(tra_z_diff)) / np.amax(tra_z))*100}%')


    print('\n'*2)
    print(modelname)
    print('rotation')
    print(f'{np.rad2deg(np.mean(rot_x_diff)):.2e}')
    print(f'{np.rad2deg(np.std(rot_x_diff)):.2e}')
    print(f'{np.rad2deg(np.mean(rot_y_diff)):.2e}')
    print(f'{np.rad2deg(np.std(rot_y_diff)):.2e}')
    print(f'{np.rad2deg(np.mean(rot_z_diff)):.2e}')
    print(f'{np.rad2deg(np.std(rot_z_diff)):.2e}')
    print(f'{np.rad2deg(np.amax(np.abs(rot_x_diff))):.2e}')
    print(f'{np.rad2deg(np.amax(np.abs(rot_y_diff))):.2e}')
    print(f'{np.rad2deg(np.amax(np.abs(rot_z_diff))):.2e}')

    print()
    print('translation')
    print(f'{np.mean(tra_x_diff):.2e}')
    print(f'{np.mean(tra_y_diff):.2e}')
    print(f'{np.mean(tra_z_diff):.2e}')
    print(f'{np.std(tra_x_diff):.2e}')
    print(f'{np.std(tra_y_diff):.2e}')
    print(f'{np.std(tra_z_diff):.2e}')
    print(f'{np.amax(np.abs(tra_x_diff)):.2e}')
    print(f'{np.amax(np.abs(tra_y_diff)):.2e}')
    print(f'{np.amax(np.abs(tra_z_diff)):.2e}')
    print('len', len(tra_x_diff))


    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(FK_result[:, :, :3, 3:].reshape(FK_result.shape[0], 3)[:, :1].flatten(),
    #         FK_result[:, :, :3, 3:].reshape(FK_result.shape[0], 3)[:, 1:2].flatten(),  # noqa
    #         FK_result[:, :, :3, 3:].reshape(FK_result.shape[0], 3)[:, 2:3].flatten(), label='estimated', color='red')
    # ax = fig.gca(projection='3d')
    # ax.scatter(translations[:, :, 0], translations[:, :, 1],
    #         translations[:, :, 2], label='truth', color='blue')
    # ax.legend()

    mean = np.mean(rot_x_diff)
    standard_deviation = np.std(rot_x_diff)
    distance_from_mean = abs(rot_x_diff - mean)
    not_outlier = distance_from_mean < 3 * standard_deviation
    no_outliers = rot_x_diff[not_outlier]

    fig = plt.hist(no_outliers, density=True, bins=100)
    # plt.title('Histogram for rotation error')
    plt.xlabel('Rotation error for x-axis [rad]')
    plt.ylabel('Samples')
    # plt.savefig(f'./{modelname.split("-")[0]}-hist.pdf')
    plt.savefig(f'{modelname.split("-")[0].split("/")[-1]}-hist.pdf')
    # plt.show()  

    plt.clf()
    plt.cla()
    mean = np.mean(tra_x_diff)
    standard_deviation = np.std(tra_x_diff)
    distance_from_mean = abs(tra_x_diff - mean)
    not_outlier = distance_from_mean < 3 * standard_deviation
    no_outliers = tra_x_diff[not_outlier]

    fig = plt.hist(no_outliers, density=True, bins=100, color='C1')
    # plt.title('Histogram for rotation error')
    plt.xlabel('Translation error for x-axis [m]')
    plt.ylabel('Samples')
    # plt.savefig(f'./{modelname.split("-")[0]}-hist.pdf')
    plt.savefig(f'{modelname.split("-")[0].split("/")[-1]}-tra-hist.pdf')

    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(angles_estimated.numpy()[:, :, 0].flatten(), angles_estimated.numpy()[:, :, 1].flatten(),
    #         angles_estimated.numpy()[:, :, 2].flatten(), label='estimated', color='red')
    # ax.legend()
    # ax = fig.gca(projection='3d')
    # ax.scatter(angles[:, :, 0], angles[:, :, 1],
    #         angles[:, :, 2], label='truth', color='blue')
    # ax.legend()
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise difference of learned IK and ground truth')
    parser.add_argument('--urdf', type=str,
                        help='Path to URDF file')
    parser.add_argument('--model', type=str,
                        help='Path to saved model (*.h5)')
    parser.add_argument('--root', type=str,
                        help='Root link')
    parser.add_argument('--tip', type=str,
                        help='tip link')
    parser.add_argument('--samples', type=int,
                        help='Number of samples')
    args = parser.parse_args()
    visualise(**vars(args))
