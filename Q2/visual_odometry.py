import numpy as np
import cv2
from data_loader import DataLoader
from camera import Camera
import matplotlib.pyplot as plt
import graphs


class VisualOdometry:
    def __init__(self, vo_data):
        self.vo_data = vo_data
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher()

        # initial camera pose
        self.camera_rotation = np.eye(3)
        self.camera_translation = np.zeros((3,1))

        # self.num_of_frames = self.vo_data.N
        # print(self.vo_data.cam.intrinsics)
        # print(self.vo_data.cam.extrinsics)

    def get_gt_trajectory(self):
        gt_trajectory = np.array([]).reshape(0, 2)
        for curr_gt_pose in self.vo_data.gt_poses:
            gt_trajectory = np.concatenate((gt_trajectory, np.array([[curr_gt_pose[0, 3], curr_gt_pose[2, 3]]])), axis=0)
        return gt_trajectory

    def calc_trajectory(self):
        gt_trajectory = np.array([]).reshape(0, 2)
        measured_trajectory = np.array([]).reshape(0, 2)
        prev_img = None
        prev_gt_pose = None
        i = 0
        for curr_img, curr_gt_pose in zip(self.vo_data.images, self.vo_data.gt_poses):
            if prev_img is None:
                prev_img = curr_img
                prev_gt_pose = curr_gt_pose
                continue

            # graphs.show_image(curr_img)
            # graphs.show_graphs()

            # feature detection
            key_points_1, descriptors_1 = self.sift.detectAndCompute(prev_img, None)
            key_points_2, descriptors_2 = self.sift.detectAndCompute(curr_img, None)

            matches = self.bf.match(descriptors_1, descriptors_2)
            prev_points = np.array([]).reshape(0, 2)
            curr_points = np.array([]).reshape(0, 2)
            for match in matches:
                point = key_points_1[match.queryIdx]
                prev_points = np.concatenate((prev_points, np.array([[point.pt[0], point.pt[1]]])), axis=0)
                point = key_points_2[match.trainIdx]
                curr_points = np.concatenate((curr_points, np.array([[point.pt[0], point.pt[1]]])), axis=0)

            E, _ = cv2.findEssentialMat(curr_points, prev_points, self.vo_data.cam.intrinsics, cv2.RANSAC, 0.99, 1.0, None)
            _, R, t, _ = cv2.recoverPose(E, curr_points, prev_points, self.vo_data.cam.intrinsics)

            # calculate the scale
            scale = np.linalg.norm(prev_gt_pose[:, 3] - curr_gt_pose[:, 3])

            self.camera_translation = self.camera_translation + scale * self.camera_rotation.dot(t)
            self.camera_rotation = R.dot(self.camera_rotation)

            gt_trajectory = np.concatenate((gt_trajectory, np.array([[curr_gt_pose[0, 3], curr_gt_pose[2, 3]]])), axis=0)
            measured_trajectory = np.concatenate((measured_trajectory, np.array([[float(self.camera_translation[0]), float(self.camera_translation[2])]])), axis=0)

            prev_img = curr_img
            prev_gt_pose = curr_gt_pose

            i += 1
            if i % 50 == 0:
                print(f"Frame {i}")

        return gt_trajectory, measured_trajectory


