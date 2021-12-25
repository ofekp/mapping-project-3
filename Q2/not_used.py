# def show_image(img):
#     fig, ax = plt.subplots()
#     ax.axis('off')
#     ax.imshow(img, cmap=plt.get_cmap('gray'))


# def get_gt_trajectory(self):
#     gt_trajectory = np.array([]).reshape(0, 2)
#     for curr_gt_pose in self.vo_data.gt_poses:
#         gt_trajectory = np.concatenate((gt_trajectory, np.array([[curr_gt_pose[0, 3], curr_gt_pose[2, 3]]])), axis=0)
#     return gt_trajectory