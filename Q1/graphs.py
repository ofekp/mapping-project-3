import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt


# def build_animation(X_Y0, X_Y1, X_Y2, x_xy_xy_y, title, xlabel, ylabel, label0, label1, label2):
#     frames = []
#
#     fig = plt.figure()
#     fig.set_size_inches(18, 18)
#     ax = fig.add_subplot(1, 1, 1)
#     print("Creating animation")
#
#     x0, y0, x1, y1, x2, y2 = [], [], [], [], [], []
#     val0, = plt.plot([], [], 'b-', animated=True, label=label0)
#     val1, = plt.plot([], [], 'g:', animated=True, label=label1)
#     val2, = plt.plot([], [], 'r--', animated=True, label=label2)
#     val3 = Ellipse(xy=(0, 0), width=0, height=0, angle=0, animated=True)
#
#     ax.add_patch(val3)
#     plt.legend()
#
#     values = np.hstack((X_Y0, X_Y1, X_Y2, x_xy_xy_y))
#
#     def init():
#         margin = 10
#         x_min = np.min(X_Y0[:, 0]) - margin
#         x_max = np.max(X_Y0[:, 0]) + margin
#         y_min = np.min(X_Y0[:, 1]) - margin
#         y_max = np.max(X_Y0[:, 1]) + margin
#         if (x_max - x_min) > (y_max - y_min):
#             h = (margin + x_max - x_min) / 2
#             c = (y_max + y_min) / 2
#             y_min = c - h
#             y_max = c + h
#         else:
#             w = (margin + y_max - y_min) / 2
#             c = (x_max + x_min) / 2
#             x_min = c - w
#             x_max = c + w
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)
#         ax.set_title(title)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         val0.set_data([], [])
#         val1.set_data([], [])
#         val2.set_data([], [])
#
#         return val0, val1, val2, val3
#
#     def update(frame):
#         x0.append(frame[0])
#         y0.append(frame[1])
#         x1.append(frame[2])
#         y1.append(frame[3])
#         x2.append(frame[4])
#         y2.append(frame[5])
#         val0.set_data(x0, y0)
#         val1.set_data(x1, y1)
#         val2.set_data(x2, y2)
#
#         cov_mat = frame[6:].reshape(2, -1)
#         ellipse = error_ellipse(np.array([frame[4], frame[5]]), cov_mat)
#
#         val3.angle = ellipse.angle
#         val3.center = ellipse.center
#         val3.width = ellipse.width
#         val3.height = ellipse.height
#         val3._alpha = ellipse._alpha
#
#         return val0, val1, val2, val3
#
#     anim = animation.FuncAnimation(fig, update, frames=values, init_func=init, interval=1, blit=True)
#     return anim


def draw_pf_frame(trueTrajectory, measured_trajectory, trueLandmarks, particles):
    fig, ax = plt.subplots()
    ax.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
    ax.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')
    ax.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
    ax.scatter(particles[:, 0], particles[:, 1], s=10, facecolors='none', edgecolors='g', alpha=0.1)
    for particle in particles:
        x = particle[0]
        y = particle[1]
        heading_line_len = 0.5
        endx = particle[0] + heading_line_len * np.cos(particle[2])
        endy = particle[1] + heading_line_len * np.sin(particle[2])
        ax.plot([x, endx], [y, endy], color='g', alpha=0.1)
    ax.grid()
    ax.set_title('Q1 - Ground trues trajectory and landmarks and noisy trajectory')
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