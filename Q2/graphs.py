import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


def show_image(img):
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.imshow(img, cmap=plt.get_cmap('gray'))


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
        enu: ground truth enu data
        enu_predicted: predicted enu data
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