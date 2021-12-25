import numpy as np
from PIL import Image
import io
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import itertools
import cv2


def save_animation(ani, basedir, file_name):
    """
    Saves the given animation 'ani' to file with file name 'file_name' in the dir 'basedir'
    """
    print("Saving animation")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='pearlofe'), bitrate=1800)
    ani.save(os.path.join(basedir, f'{file_name}.mp4'), writer=writer, dpi=100)
    print("Animation saved")


def build_animation(gt_x_y, est_x_y, key_points_history, vo_data, title, xlabel, ylabel, gt_label, est_label):
    """
    Creates the animation with the ground truth trajectory, the estimated trajectory, the landmarks and the particles
    and their heading.

    Parameters:
        gt_x_y - dim is [num_frames, 2]
        est_x_y - dim is [num_frames, 2]
        key_points_history - list([num_key_points_in_frame, 2]) the list size is num_frames
        vo_data - the loaded visual odometry data, used to show the images from camera 0
        title - graph title
        xlabel, ylabel - the axes labels of the graph
        gt_label, est_label - legend labels
    """
    frames = []

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.set_size_inches(18, 18)
    print("Creating animation")

    gt_x, gt_y, est_x, est_y = [], [], [], []

    val0 = ax1.imshow(next(itertools.islice(vo_data.images, 0, None)), cmap=plt.get_cmap('gray'), animated=True)
    val1 = ax1.scatter([], [], s=30, facecolors='none', edgecolors='g', alpha=0.7)
    val2, = ax2.plot([], [], 'b-', animated=True, label=gt_label)
    val3, = ax2.plot([], [], 'r:', animated=True, label=est_label)

    plt.legend()

    values = np.hstack((gt_x_y[:, 0:2], est_x_y[:, 0:2], np.arange(gt_x_y.shape[0]).reshape(-1, 1)))

    def init():
        margin = 10
        x_min = np.min(gt_x_y[:, 0]) - margin
        x_max = np.max(gt_x_y[:, 0]) + margin
        y_min = np.min(gt_x_y[:, 1]) - margin
        y_max = np.max(gt_x_y[:, 1]) + margin
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

        ax1.set_axis_off()

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_title(title)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)

        val0.set_data(next(itertools.islice(vo_data.images, 0, None)))
        val1.set_offsets(np.array([[]]))
        val2.set_data([], [])
        val3.set_data([], [])

        return val0, val1, val2, val3

    def update(frame):
        gt_x.append(frame[0])
        gt_y.append(frame[1])
        est_x.append(frame[2])
        est_y.append(frame[3])

        frame_idx = int(frame[4])
        if frame_idx % 50 == 0:
            print(frame_idx)

        file = os.path.join(vo_data.img_dir, "{:06d}.png".format(frame_idx))
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        val0.set_array(img)
        key_points = key_points_history[frame_idx]
        val1.set_offsets(key_points)
        val2.set_data(gt_x, gt_y)
        val3.set_data(est_x, est_y)

        return val0, val1, val2, val3

    anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, blit=True)
    return anim


def plot_gt_trajectory(gt_trajectory):
    """
    Plots a comparison between the ground trajectory and the predicted one
    Args:
        enu: ground truth enu data
        enu_predicted: predicted enu data
    """
    fig, ax = plt.subplots()
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Comparison of the ground truth trajectory and the predicted trajectory", fontsize=20)
    ax.set_xlabel("East [meters]", fontsize=20)
    ax.set_ylabel("North [meters]", fontsize=20)


def plot_trajectory_comparison(gt_trajectory, measured_trajectory):
    """
    Plots a comparison between the ground trajectory and the predicted one
    Args:
        enu: ground truth trajectory
        enu_predicted: predicted trajectory with visual odometry
    """
    fig, ax = plt.subplots()
    ax.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b')
    ax.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], 'g')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Comparison of the ground truth trajectory and the predicted trajectory", fontsize=20)
    ax.set_xlabel("East [meters]", fontsize=20)
    ax.set_ylabel("North [meters]", fontsize=20)
    ax.legend(["Ground truth trajectory", "Predicted trajectory"], prop={"size": 20}, loc="best")


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