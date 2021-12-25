import os
from visual_odometry import VisualOdometry
from data_loader import DataLoader
import graphs
import numpy as np


class ProjectQuestions:
    def __init__(self, vo_data):
        """
        Initialization of Q2 with the loaded data
        """
        assert type(vo_data) is dict, "vo_data should be a dictionary"
        assert all([val in list(vo_data.keys()) for val in ['sequence', 'dir']]), "vo_data must contain keys: ['sequence', 'dir']"
        assert type(vo_data['sequence']) is int and (0 <= vo_data['sequence'] <= 10), "sequence must be an integer value between 0-10"
        assert type(vo_data['dir']) is str and os.path.isdir(vo_data['dir']), "dir should be a directory"
        self.vo_data = vo_data

    def Q2(self):
        """
        prepares the data for visual odometry and runs the algorithm we learned in class
        in this question we find the estimated trajectory by performing monocular visual odometry (single camera)
        """
        vo_data = DataLoader(self.vo_data)
        vo = VisualOdometry(vo_data)
        gt_trajectory, measured_trajectory, key_points_history = vo.calc_trajectory()
        graphs.plot_trajectory_comparison(gt_trajectory, measured_trajectory)
        graphs.show_graphs("../../../Results/Visual Odometry/", "vo_gt_trajectory_estimated_trajectory")
        title = "Visual Odometry Ground Truth and Predicted Trajectories"
        anim = graphs.build_animation(gt_trajectory, measured_trajectory, key_points_history, vo_data,
                                      title, "x [meters]", "z (optical axis) [meters]", "Ground truth trajectory",
                                      "Estimated trajectory with visual odometry")
        graphs.save_animation(anim, "../../../Results/Visual Odometry/", "vo_animation")

        # find the max deviation from the ground truth trajectory
        frames_to_measure = 500
        diff = gt_trajectory - measured_trajectory
        max_distance = np.max(np.linalg.norm(diff[:frames_to_measure], axis=1))
        print(f"The maximum Euclidean distance between the ground truth trajectory and "
              f"the estimated trajectory in the first [{frames_to_measure}] frames is [{max_distance}] meters")

    def run(self):
        """
        run Q2
        """
        self.Q2()
    