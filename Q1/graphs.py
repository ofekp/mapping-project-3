import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from pandas.plotting import table
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import os


def save_animation(ani, basedir, file_name):
    print("Saving animation")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=50, metadata=dict(artist='pearlofe'), bitrate=1800)
    ani.save(os.path.join(basedir, f'{file_name}.mp4'), writer=writer, dpi=100)
    print("Animation saved")


def build_animation(gt_x_y_phi, est_x_y_phi, landmarks_x_y, particles_x_y_phi, title, xlabel, ylabel, gt_label, est_label, landmarks_label, particles_label):
    """
    particles_x_y_phi dim is [num_frames, 3, number_of_particles]
    """
    frames = []

    fig = plt.figure()
    fig.set_size_inches(18, 18)
    ax = fig.add_subplot(1, 1, 1)
    print("Creating animation")

    gt_x, gt_y, est_x, est_y = [], [], [], []
    val0, = plt.plot([], [], 'b-', animated=True, label=gt_label)
    val1 = LineCollection([], color='c', alpha=0.01)
    val2 = plt.scatter([], [], s=10, facecolors='none', edgecolors='g', alpha=0.2, label=particles_label)
    val3, = plt.plot([], [], 'r:', animated=True, label=est_label)

    ax.add_collection(val1)
    plt.legend()

    # expand the particles, we will have a total of num_frames rows, and 3 * num_particles cols
    particles_x_y_phi_expended = np.hstack(particles_x_y_phi.transpose(1, 0, 2))
    values = np.hstack((gt_x_y_phi[:, 0:2], est_x_y_phi[:, 0:2], particles_x_y_phi_expended))

    def init():
        margin = 10
        x_min = np.min(gt_x_y_phi[:, 0]) - margin
        x_max = np.max(gt_x_y_phi[:, 0]) + margin
        y_min = np.min(gt_x_y_phi[:, 1]) - margin
        y_max = np.max(gt_x_y_phi[:, 1]) + margin
        if (x_max - x_min) > (y_max - y_min):
            h = (margin + x_max - x_min) / 2
            c = (y_max + y_min) / 2
            y_min = c - h
            y_max = c + h
        else:
            w = (margin + y_max - y_min) / 2
            c = (x_max + x_min) / 2
            x_min = c - w
            x_max = c + w
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.scatter(landmarks_x_y[:, 0], landmarks_x_y[:, 1], s=80, facecolors='none', edgecolors='b',
                   label=landmarks_label)
        val0.set_data([], [])
        val1.set_segments([])
        val2.set_offsets(np.array([[]]))
        val3.set_data([], [])

        return val0, val1, val2, val3

    def update(frame):
        gt_x.append(frame[0])
        gt_y.append(frame[1])
        est_x.append(frame[2])
        est_y.append(frame[3])
        val0.set_data(gt_x, gt_y)

        particles_x = frame[np.arange(4, len(frame), 3)]
        particles_y = frame[np.arange(5, len(frame), 3)]
        particles_phi = frame[np.arange(6, len(frame), 3)]
        line_segments = []
        for i in range(particles_x.shape[0]):
            x = particles_x[i]
            y = particles_y[i]
            phi = particles_phi[i]
            heading_line_len = 0.5
            endx = x + heading_line_len * np.cos(phi)
            endy = y + heading_line_len * np.sin(phi)
            line_segments.append(np.array([[x, y], [endx, endy]]))
        val1.set_segments(line_segments)

        val2.set_offsets(np.concatenate((particles_x.reshape(-1, 1), particles_y.reshape(-1, 1)), axis=1))
        val3.set_data(est_x, est_y)

        return val0, val1, val2, val3

    anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, blit=True)
    return anim


def draw_table(df):
    fig, ax = plt.subplots()
    ax.set_frame_on(False)
    ax.axis('off')
    t = table(ax, df, loc='center', cellLoc='center', colWidths=[0.08] * len(df.columns))
    # cellDict = t.get_celld()
    # for i in range(len(df.columns)):
    #     cellDict[(0, i)].set_height(.02)
    #     for j in range(1, len(df) + 1):
    #         cellDict[(j, i)].set_height(.02)
    t.scale(1, 1)
    t.set_fontsize(25)


def draw_pf_frame_with_closes_landmark(trueTrajectory, measured_trajectory, trueLandmarks, particles, pos, closest_landmark, r, phi):
    fig, ax = plt.subplots()
    ax.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    ax.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
    line_segments = []
    for particle in particles:
        x = particle[0]
        y = particle[1]
        heading_line_len = 0.5
        endx = x + heading_line_len * np.cos(particle[2])
        endy = y + heading_line_len * np.sin(particle[2])
        line_segments.append(np.array([[x, y], [endx, endy]]))
    line_collection = LineCollection(line_segments, color='c', alpha=0.05)
    ax.scatter(particles[:, 0], particles[:, 1], s=10, facecolors='none', edgecolors='g', alpha=0.2)
    ax.add_collection(line_collection)
    ax.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')

    # mark closest landmark
    ax.scatter(pos[0], pos[1], s=20, facecolors='none', edgecolors='orange')
    ax.scatter(closest_landmark[0], closest_landmark[1], s=60, facecolors='none', edgecolors='orange')
    x = pos[0]
    y = pos[1]
    endx = pos[0] + r * np.cos(pos[2] + phi)
    endy = pos[1] + r * np.sin(pos[2] + phi)
    ax.plot([x, endx], [y, endy], color='orange', alpha=0.5)

    ax.grid()
    ax.set_title('Q1 - Ground trues trajectory and landmarks and noisy trajectory')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X [m]", fontsize=20)
    ax.set_ylabel("Y [m]", fontsize=20)
    ax.legend(['Ground Truth', 'Landmarks'], prop={"size": 20}, loc="best")


def draw_pf_frame(trueTrajectory, measured_trajectory, trueLandmarks, particles, title):  #, pos, closest_landmark, r, phi):
    fig, ax = plt.subplots()
    ax.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    ax.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
    line_segments = []
    for particle in particles:
        x = particle[0]
        y = particle[1]
        heading_line_len = 0.5
        endx = x + heading_line_len * np.cos(particle[2])
        endy = y + heading_line_len * np.sin(particle[2])
        line_segments.append(np.array([[x, y], [endx, endy]]))
    line_collection = LineCollection(line_segments, color='c', alpha=0.05)
    ax.scatter(particles[:, 0], particles[:, 1], s=10, facecolors='none', edgecolors='g', alpha=0.2)
    ax.add_collection(line_collection)
    ax.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')

    ax.grid()
    ax.set_title(title, fontsize=20)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X [m]", fontsize=20)
    ax.set_ylabel("Y [m]", fontsize=20)
    ax.legend(['Ground Truth', 'Landmarks'], prop={"size": 20}, loc="best")


def show_graphs(folder=None, file_name=None):
    """
    Saves or shows the current plot
    Args:
        folder: optional, must be provided with file_name too, the image will be saved to this path with the given file name
        file_name: optional, must be provided with folder too, the image will be saved to this path with the given file name
    """
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    if not folder or not file_name:
        plt.show()
    else:
        file_name = "{}/{}.png".format(folder, file_name)
        figure = plt.gcf()  # get current figure
        number_of_subplots_in_figure = len(plt.gcf().get_axes())
        figure.set_size_inches(number_of_subplots_in_figure * 18, 18)
        ram = io.BytesIO()
        plt.savefig(ram, format='png', dpi=100)
        ram.seek(0)
        im = Image.open(ram)
        im2 = im.convert('RGB').convert('P', palette=Image.ADAPTIVE)
        im2.save(file_name, format='PNG')
        plt.close(figure)
    plt.close('all')