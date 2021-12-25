# def resampleParticles(self):
#     N = len(self.weights)
#
#     positions = (np.arange(N) + np.random.uniform()) / N
#
#     indexes = np.zeros(N, dtype=int)
#     cumulative_sum = np.cumsum(self.weights)
#     i, j = 0, 0
#     while i < N:
#         if positions[i] < cumulative_sum[j]:
#             indexes[i] = j
#             i += 1
#         else:
#             j += 1
#     self.particles[:] = self.particles[indexes]
#     self.weights.resize(len(self.particles))
#     self.weights.fill(1.0 / len(self.weights))


# def draw_pf_frame_with_closes_landmark(trueTrajectory, measured_trajectory, trueLandmarks, particles, pos, closest_landmark, r, phi):
#     fig, ax = plt.subplots()
#     ax.plot(trueTrajectory[:, 0], trueTrajectory[:, 1])
#     ax.scatter(trueLandmarks[:, 0], trueLandmarks[:, 1], s=80, facecolors='none', edgecolors='b')
#     line_segments = []
#     for particle in particles:
#         x = particle[0]
#         y = particle[1]
#         heading_line_len = 0.5
#         endx = x + heading_line_len * np.cos(particle[2])
#         endy = y + heading_line_len * np.sin(particle[2])
#         line_segments.append(np.array([[x, y], [endx, endy]]))
#     line_collection = LineCollection(line_segments, color='orange', alpha=0.08)
#     ax.scatter(particles[:, 0], particles[:, 1], s=10, facecolors='none', edgecolors='g', alpha=0.2)
#     ax.add_collection(line_collection)
#     ax.plot(measured_trajectory[:, 0], measured_trajectory[:, 1], color='r')
#
#     # mark closest landmark
#     ax.scatter(pos[0], pos[1], s=20, facecolors='none', edgecolors='orange')
#     ax.scatter(closest_landmark[0], closest_landmark[1], s=60, facecolors='none', edgecolors='orange')
#     x = pos[0]
#     y = pos[1]
#     endx = pos[0] + r * np.cos(pos[2] + phi)
#     endy = pos[1] + r * np.sin(pos[2] + phi)
#     ax.plot([x, endx], [y, endy], color='c', alpha=0.5)
#
#     ax.grid()
#     ax.set_title('Q1 - Ground trues trajectory and landmarks and noisy trajectory')
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlabel("X [m]", fontsize=20)
#     ax.set_ylabel("Y [m]", fontsize=20)
#     ax.legend(['Ground Truth', 'Landmarks'], prop={"size": 20}, loc="best")