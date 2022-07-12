import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow_graphics as tfg
import tensorflow_graphics.geometry.transformation.rotation_matrix_3d as tfg_rotation_matrix_3d


from dlkinematics.urdf import chain_from_urdf_file
from dlkinematics.dlkinematics import DLKinematics


def visualise(urdf, root, tip, data):
    with open(data, 'r') as f:
        res = f.read()
    true = []
    estimated = []
    iterations = []
    for row in res.split('\n\n'):
        row = row.replace('\n', '')
        true.append(eval(row.split(',')[0].strip().replace(' ', ',')))
        estimated.append(eval(row.split(',')[1].strip().replace(' ', ',')))
        iterations.append(int(row.split(',')[2].strip()))

    estimated = np.array(estimated)
    true = np.array(true)
    iterations = np.array(iterations)
    robot = chain_from_urdf_file(urdf)
    dl_kin = DLKinematics(robot, root, tip, batch_size=len(estimated))

    fk_true = dl_kin.forward(true.flatten())
    fk_estimated = dl_kin.forward(estimated.flatten())

    angles_true = tfg.geometry.transformation.euler.from_rotation_matrix(
        fk_true[:, :3, :3])

    angles_estimated = tfg.geometry.transformation.euler.from_rotation_matrix(
        fk_estimated[:, :3, :3])

    angles_diff = angles_true - angles_estimated
    translation_diff = fk_true[:, :3, 3:] - fk_estimated[:, :3, 3:]

    rot_x_diff = angles_diff[:, 0].numpy()
    rot_y_diff = angles_diff[:, 1].numpy()
    rot_z_diff = angles_diff[:, 2].numpy()

    tra_x_diff = translation_diff[:, 0]
    tra_y_diff = translation_diff[:, 1]
    tra_z_diff = translation_diff[:, 2]

    rot_x_diff = np.where(rot_x_diff > np.pi, rot_x_diff - np.pi*2, rot_x_diff)
    rot_x_diff = np.where(rot_x_diff < -np.pi,
                          rot_x_diff + np.pi*2, rot_x_diff)

    rot_y_diff = np.where(rot_y_diff > np.pi, rot_y_diff - np.pi*2, rot_y_diff)
    rot_y_diff = np.where(rot_y_diff < -np.pi,
                          rot_y_diff + np.pi*2, rot_y_diff)

    rot_z_diff = np.where(rot_z_diff > np.pi, rot_z_diff - np.pi*2, rot_z_diff)
    rot_z_diff = np.where(rot_z_diff < -np.pi,
                          rot_z_diff + np.pi*2, rot_z_diff)

    rot_x_diff = np.nan_to_num(rot_x_diff)
    rot_y_diff = np.nan_to_num(rot_y_diff)
    rot_z_diff = np.nan_to_num(rot_z_diff)

    print('\n' * 2)
    print(data)
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

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    c1 = pd.read_csv('../evaluation/2.csv').iloc[:, 1]
    plt.plot(c1, label='$chain_{2}$')
    c3 = pd.read_csv('../evaluation/24.csv').iloc[:, 1]
    plt.plot(c3, label='$chain_{24}$')
    plt.legend(loc="lower right")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualise difference of estimated IK form gradient descent')
    parser.add_argument('--urdf', type=str,
                        help='Path to URDF file')
    parser.add_argument('--data', type=str,
                        help='Path to saved data (*.csv)')
    parser.add_argument('--root', type=str,
                        help='Root link')
    parser.add_argument('--tip', type=str,
                        help='tip link')
    args = parser.parse_args()
    visualise(**vars(args))
