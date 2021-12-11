import os
from visual_odometry import VisualOdometry
from data_loader import DataLoader
import graphs


class ProjectQuestions:
    def __init__(self,vo_data):
        assert type(vo_data) is dict, "vo_data should be a dictionary"
        assert all([val in list(vo_data.keys()) for val in ['sequence', 'dir']]), "vo_data must contain keys: ['sequence', 'dir']"
        assert type(vo_data['sequence']) is int and (0 <= vo_data['sequence'] <= 10), "sequence must be an integer value between 0-10"
        assert type(vo_data['dir']) is str and os.path.isdir(vo_data['dir']), "dir should be a directory"
        self.vo_data = vo_data

    def Q2(self):
        vo_data = DataLoader(self.vo_data)
        vo = VisualOdometry(vo_data)
        gt_trajectory, measured_trajectory = vo.calc_trajectory()
        graphs.plot_trajectory_comparison(gt_trajectory, measured_trajectory)
        graphs.show_graphs()
        # gt_trajectory = vo.get_gt_trajectory()
        # graphs.plot_gt_trajectory(gt_trajectory)
        # graphs.show_graphs()
    
    def run(self):
        self.Q2()
    