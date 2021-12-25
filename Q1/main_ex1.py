import read_ex1_data
from ParticlesFilter import *
import numpy as np
import matplotlib.pyplot as plt
import math
import graphs
import itertools
import pandas as pd

# np.random.seed(12345)
# np.random.seed(17)
np.random.seed(11)


def calculate_mse(X_Y_GT, X_Y_est, start_frame=50):
    """
    calculate MSE

    Args:
        X_Y_GT (np.ndarray): ground truth values of x and y
        X_Y_est (np.ndarray): estimated values of x and y

    Returns:
        float: MSE
    """
    e_x = X_Y_GT[:, 0].squeeze() - X_Y_est[:, 0].squeeze()  # e_x dim is [1, -1]
    e_y = X_Y_GT[:, 1].squeeze() - X_Y_est[:, 1].squeeze()  # e_y dim is [1, -1]
    e_x = e_x[start_frame:]
    e_y = e_y[start_frame:]
    MSE = np.sqrt((1 / (e_x.shape[0] - start_frame)) * (np.dot(e_x, e_x.T) + np.dot(e_y, e_y.T)))
    return float(MSE)

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
    plt.figure()
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.grid()
    plt.xlabel("X [m]", fontsize=20)
    plt.ylabel("Y [m]", fontsize=20)
    plt.legend(['Ground Truth', 'Landmarks'], prop={"size": 20}, loc="best")
    plt.title('Ground trues trajectory and landmarks', fontsize=20)
    graphs.show_graphs("../../../Results/Particle Filter/", "ground_truth_trajectory_and_landmarks")

    """ Generate measurement odometry """
    sigma_r1 = 0.01
    sigma_t = 0.1
    sigma_r2 = 0.01
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

    plt.figure()
    plt.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    plt.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')
    plt.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.grid()
    plt.xlabel("X [m]", fontsize=20)
    plt.ylabel("Y [m]", fontsize=20)
    plt.legend(['Ground truth trajectory', 'Trajectory with gaussian noise in the odometry data', 'Landmarks'], prop={"size": 20}, loc="best")
    plt.title('Ground trues trajectory, landmarks and noisy trajectory', fontsize=20)
    graphs.show_graphs("../../../Results/Particle Filter/", "ground_truth_trajectory_noisy_trajectory_and_landmarks")

    """ Create ParticlesFilter object """
    sigma_range = 1.0
    sigma_bearing = 0.1
    num_particles_arr = [1, 10, 20, 30, 50, 100, 500, 1000, 10000]
    add_sensor_noise_arr = [True, False]
    # num_particles_arr = [500]
    # add_sensor_noise_arr = [True]
    mse_results = pd.DataFrame(np.array([[0] * len(num_particles_arr)] * len(add_sensor_noise_arr)),
                               columns=num_particles_arr, index=[str(s) for s in add_sensor_noise_arr])
    for add_sensor_noise, num_particles in itertools.product(add_sensor_noise_arr, num_particles_arr):
        pf = ParticlesFilter(trueLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfPaticles=num_particles)
        for i, timestamp in enumerate(range(trueOdometry.__len__() - 1)):
            # calculate Zt - the range and bearing to the closest landmark as seen from the current true position of the robot
            if i == 0 and num_particles == 1000 and add_sensor_noise:
                # just to show the initial distribution of particles
                graphs.draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles, "Ground truth trajectory and landmarks, showing the initial distribution of 1000 particles")
                graphs.show_graphs("../../../Results/Particle Filter/", "pf_gt_trajectory_landmarks_1000_particles_initial_particle_distribution")
            closest_landmark_id = np.argmin(np.linalg.norm((trueLandmarks - trueTrajectory[i + 1, 0:2]), axis=1))
            dist_xy = trueLandmarks[closest_landmark_id] - trueTrajectory[i + 1, 0:2]
            r = np.linalg.norm(dist_xy)
            phi = np.arctan2(dist_xy[1], dist_xy[0]) - trueTrajectory[i + 1, 2]
            if add_sensor_noise:
                r += np.random.normal(0, sigma_range)
                phi += np.random.normal(0, sigma_bearing)
            phi = ParticlesFilter.normalize_angle(phi)
            Zt = np.array([r, phi])
            # graphs.draw_pf_frame_with_closes_landmark(trueTrajectory, pf.history, trueLandmarks, pf.particles, trueTrajectory[i + 1], trueLandmarks[closest_landmark_id], r, phi)
            # graphs.show_graphs()
            pf.apply(Zt, trueOdometry[timestamp])
        title = "pf_estimation_{}_{}_particles".format("with_sensor_noise" if add_sensor_noise else "without_sensor_noise", num_particles)
        graphs.draw_pf_frame(trueTrajectory, pf.history, trueLandmarks, pf.particles, title.replace("_", " ").replace("pf", "Particle filter"))
        graphs.show_graphs("../../../Results/Particle Filter/", title)
        # graphs.show_graphs()
        mse = round(calculate_mse(trueTrajectory, pf.history), 5)
        mse_results.loc[str(add_sensor_noise), num_particles] = mse
        if num_particles >= 1000:
            title = "pf_animation_{}_{}_particles".format("with_sensor_noise" if add_sensor_noise else "without_sensor_noise", num_particles)
            anim = graphs.build_animation(trueTrajectory, pf.history, trueLandmarks, pf.particles_history, title.replace("_", " ").replace("pf", "Particle filter"), "x [meters]", "y [meters]", "Ground truth trajectory", "Particle filter estimated trajectory", "Landmarks", "Particles and their heading")
            graphs.save_animation(anim, "../../../Results/Particle Filter/", title)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.expand_frame_repr', False):
        print(mse_results)
    graphs.draw_table(mse_results)
    graphs.show_graphs("../../../Results/Particle Filter/", "pf_mse_summary_table")


if __name__ == "__main__":
    main()
