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