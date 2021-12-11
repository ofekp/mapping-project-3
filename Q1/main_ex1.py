import os.path

import read_ex1_data
from ParticlesFilter import *
import numpy as np
import matplotlib.pyplot as plt
import math
import graphs

np.random.seed(12345)


def main():
    """
    This function in my implementation for Particles filter
    """
    """ Load data """
    print("Reading ground truth landmarks")
    trueLandmarks = np.array(read_ex1_data.read_landmarks("./LastID_5.csv"))

    print("Reading ground truth odometry")
    trueOdometry = read_ex1_data.read_odometry("./odometry.dat")

    """ Calculate true trajectory """
    trueTrajectory = np.zeros((trueOdometry.__len__(), 3))
    for i in range(1, trueOdometry.__len__()):
        dr1 = trueOdometry[i - 1]['r1']
        dt = trueOdometry[i - 1]['t']
        dr2 = trueOdometry[i - 1]['r2']
        theta = trueTrajectory[i - 1, 2]
        dMotion = np.expand_dims(np.array([dt * math.cos(theta + dr1), dt * math.sin(theta + dr1), dr1 + dr2]), 0)
        trueTrajectory[i, :] = trueTrajectory[i-1, :] + dMotion

    """ Plot Q1 """
    plt.figure('Q1 - Ground trues trajectory and landmarks')
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.grid()
    plt.xlabel("X [m]", fontsize=20)
    plt.ylabel("Y [m]", fontsize=20)
    plt.legend(['Ground Truth', 'Landmarks'], prop={"size": 20}, loc="best")
    plt.title('Q1.a - Ground trues trajectory and landmarks')
    graphs.show_graphs("../../../Results/Particle Filter/", "ground_truth_trajectory_and_landmarks")

    """ Generate measurement odometry """
    sigma_r1 = 0.01
    sigma_t =  0.1
    sigma_r2 =  0.01
    measurmentOdometry = dict()
    measured_trajectory = np.zeros((trueOdometry.__len__() + 1, 3))
    for i, timestamp in enumerate(range(trueOdometry.__len__())):
        dr1 = trueOdometry[timestamp]['r1'] + float(np.random.normal(0, sigma_r1, 1))
        dt = trueOdometry[timestamp]['t'] + float(np.random.normal(0, sigma_t, 1))
        dr2 = trueOdometry[timestamp]['r2'] + float(np.random.normal(0, sigma_r2, 1))
        measurmentOdometry[timestamp] = {'r1': dr1,
                                         't': dt,
                                         'r2': dr2}
        theta = measured_trajectory[i, 2]
        dMotion = np.expand_dims(np.array([dt * math.cos(theta + dr1), dt * math.sin(theta + dr1), dr1 + dr2]), 0)
        measured_trajectory[i + 1, :] = measured_trajectory[i, :] + dMotion

    plt.figure('Q1 - Ground trues trajectory and landmarks and noisy trajectory')
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.grid()
    plt.xlabel("X [m]", fontsize=20)
    plt.ylabel("Y [m]", fontsize=20)
    plt.legend(['Ground Truth', 'Landmarks'], prop={"size": 20}, loc="best")
    plt.title('Q1.a - Ground trues trajectory and landmarks')
    graphs.show_graphs("../../../Results/Particle Filter/", "ground_truth_trajectory_noisy_trajectory_and_landmarks")

    """ Create ParticlesFilter object """
    pf = ParticlesFilter(trueLandmarks, sigma_r1, sigma_t, sigma_r2)
    for i, timestamp in enumerate(range(measurmentOdometry.__len__() - 1)):
        # graphs.draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles)
        # graphs.show_graphs()
        # calculate Zt - the range and bearing to the closest landmark as seen from the current true position of the robot
        closest_landmark_id = np.argmin(np.linalg.norm(trueLandmarks - trueTrajectory[i + 1, 0:2], axis=1))
        dist_xy = trueLandmarks[closest_landmark_id] - trueTrajectory[i + 1, 0:2]
        # r = np.linalg.norm(dist_xy) + np.random.normal(0, 1.0)
        r = np.linalg.norm(dist_xy)
        # phi = ParticlesFilter.normalize_angle(np.arctan2(dist_xy[1], dist_xy[0]) + np.random.normal(0, 0.1))
        phi = np.arctan2(dist_xy[1], dist_xy[0])
        Zt = np.array([r, phi])
        pf.apply(Zt, measurmentOdometry[timestamp])
        # if i % 10 == 0:
        #     graphs.draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles)
        #     graphs.show_graphs()
    graphs.draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles)
    graphs.show_graphs()


if __name__ == "__main__":
    main()
