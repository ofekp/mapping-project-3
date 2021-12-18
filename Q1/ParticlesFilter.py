import numpy as np
import graphs


class ParticlesFilter:
    def __init__(self, worldLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfPaticles=1000):

        # Initialize parameters
        self.numberOfParticles = numberOfPaticles
        self.worldLandmarks = worldLandmarks

        self.sigma_r1 = sigma_r1
        self.sigma_t = sigma_t
        self.sigma_r2 = sigma_r2

        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing

        # Initialize particles - x, y, heading, weight (uniform weight for initialization)
        # self.particles = np.concatenate((np.random.normal(0, np.sqrt(2.0), (self.numberOfParticles, 1)),
        #                                  np.random.normal(0, np.sqrt(2.0), (self.numberOfParticles, 1)),
        #                                  ParticlesFilter.normalize_angles_array(np.random.normal(0.1, np.sqrt(0.1), (self.numberOfParticles, 1)))), axis=1)
        self.particles = np.concatenate((np.random.normal(0, 2.0, (self.numberOfParticles, 1)),
                                         np.random.normal(0, 2.0, (self.numberOfParticles, 1)),
                                         ParticlesFilter.normalize_angles_array(np.random.normal(0.1, 0.1, (self.numberOfParticles, 1)))), axis=1)
        # self.particles = np.concatenate((np.random.normal(0, 0.5, (self.numberOfParticles, 1)),
        #                                  np.random.normal(0, 0.5, (self.numberOfParticles, 1)),
        #                                  ParticlesFilter.normalize_angles_array(np.random.normal(0.1, 0.1, (self.numberOfParticles, 1)))), axis=1)

        # self.particles = np.array([[0, 0, 0.1], [0.5, 0.5, 0.3], [1.5, 1.5, 1.3]])
        self.weights = np.ones(self.numberOfParticles) * (1.0 / self.numberOfParticles)
        # TODO(ofekp): I assumed below that it is 0,0,0.1, should probably take it from the trajectory with the noise
        self.history = np.array((0, 0, 0.1)).reshape(1, 3)
        self.particles_history = np.expand_dims(self.particles.copy(), axis=0)


    def apply(self, Zt, Ut):

        # Motion model based on odometry
        self.motionModel(Ut)

        # graphs.draw_pf_frame(self.true_trajectory, self.history, self.true_landmarks, self.particles)
        # graphs.show_graphs()

        # Observation model
        observations = self.Observation()

        # Observation model
        self.weightParticles(Zt, observations)

        self.history = np.concatenate((self.history, self.bestKParticles(1).reshape(1, 3)), axis=0)

        # Resample particles
        self.resampleParticles()

        self.particles_history = np.concatenate((self.particles_history, np.expand_dims(self.particles.copy(), axis=0)), axis=0)


    # TODO(ofekp): continue - https://github.com/VincentChen95/Robot-Localization-with-Particle-Filter-and-Extend-Kalman-Filter/blob/master/pf.py#L23
    def motionModel(self, odometry):
        # dr1 = odometry['r1']
        # dt  = odometry['t']
        # dr2 = odometry['r2']
        dr1 = odometry['r1'] + np.random.normal(0, self.sigma_r1, (self.numberOfParticles, 1))
        dt  = odometry['t']  + np.random.normal(0, self.sigma_t,  (self.numberOfParticles, 1))
        dr2 = odometry['r2'] + np.random.normal(0, self.sigma_r2, (self.numberOfParticles, 1))
        # dr1 = ParticlesFilter.normalize_angles_array(odometry['r1'] + np.random.normal(0, self.sigma_r1 * np.sqrt(odometry['r1'] ** 2), (self.numberOfParticles, 1)))
        # dt = odometry['t'] + np.random.normal(0, self.sigma_t * np.sqrt(odometry['t'] ** 2), (self.numberOfParticles, 1))
        # dr2 = ParticlesFilter.normalize_angles_array(odometry['r2'] + np.random.normal(0, self.sigma_r2 * np.sqrt(odometry['r2'] ** 2), (self.numberOfParticles, 1)))
        theta = self.particles[:, 2].reshape(-1, 1)
        # TODO(ofekp): does each particle get its own random odometry movement, or do they all get the movement I randomized before
        dMotion = np.concatenate((
            dt * np.cos(theta + dr1),
            dt * np.sin(theta + dr1),
            dr1 + dr2), axis=1)
        self.particles = self.particles + dMotion
        self.particles[:, 2] = ParticlesFilter.normalize_angles_array(self.particles[:, 2])
        # for i, particle in enumerate(self.particles):
        #     theta = particle[2]
        #     # dr1 = ParticlesFilter.normalize_angle(odometry['r1'] + np.random.normal(0, 15.0 * self.sigma_r1 * np.sqrt(odometry['r1'] ** 2)))
        #     # dt = odometry['t'] + np.random.normal(0, 15.0 * self.sigma_t * np.sqrt(odometry['t'] ** 2))
        #     # dr2 = ParticlesFilter.normalize_angle(odometry['r2'] + np.random.normal(0, 15.0 * self.sigma_r2 * np.sqrt(odometry['r2'] ** 2)))
        #     dr1 = ParticlesFilter.normalize_angle(odometry['r1'] + np.random.normal(0, self.sigma_r1))
        #     dt = odometry['t'] + np.random.normal(0, self.sigma_t)
        #     dr2 = ParticlesFilter.normalize_angle(odometry['r2'] + np.random.normal(0, self.sigma_r2))
        #     # dr1 = odometry['r1']
        #     # dt = odometry['t']
        #     # dr2 = odometry['r2']
        #     self.particles[i, 0] += dt * np.cos(theta + dr1)
        #     self.particles[i, 1] += dt * np.sin(theta + dr1)
        #     self.particles[i, 2] += dr1 + dr2
        #     # particle[0] += dt * np.cos(theta + dr1) + np.random.normal(0, self.sigma_t)
        #     # particle[1] += dt * np.sin(theta + dr1) + np.random.normal(0, self.sigma_t)
        #     # particle[2] += dr1 + dr2 + np.random.normal(0, self.sigma_r1 + self.sigma_r2)
        # self.particles[:, 2] = ParticlesFilter.normalize_angles_array(self.particles[:, 2])

    def Observation(self):
        observations = np.zeros((self.particles.shape[0], 2))  # range and bearing for each
        for i, particle in enumerate(self.particles):
            # closest_landmark_id = np.argmin(np.linalg.norm(self.worldLandmarks - particle[0:2], axis=1))
            closest_landmark_id = np.argmin(np.sum((self.worldLandmarks - particle[0:2]) ** 2, axis=1))
            dist_xy = self.worldLandmarks[closest_landmark_id] - particle[0:2]
            r = np.linalg.norm(dist_xy)
            phi = ParticlesFilter.normalize_angle(np.arctan2(dist_xy[1], dist_xy[0]) - particle[2])
            observations[i, 0] = r
            observations[i, 1] = phi
        return observations

    def weightParticles(self, world_measurement, observations):
        # cov = np.diag([self.sigma_range ** 2, self.sigma_bearing ** 2])
        # cov = np.diag([5.0, 0.5])
        diff = world_measurement - observations
        cov = np.cov(diff.T)
        for i, observation in enumerate(observations):
            d = world_measurement - observation
            d[1] = ParticlesFilter.normalize_angle(d[1])
            self.weights[i] = np.exp(-0.5 * np.dot(d.T, np.dot(np.linalg.inv(cov), d))) / np.sqrt(np.linalg.det(2 * np.pi * cov))
            # self.weights[i] = np.sqrt(np.dot(np.dot(d.T, np.linalg.pinv(cov)), d))
        # self.weights = np.max(self.weights) - self.weights
        self.weights += 1.0e-200  # for numerical stability
        self.weights /= sum(self.weights)

    # TODO(ofekp): delete this method
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

    def resampleParticles(self):
        N = len(self.weights)
        r = np.random.uniform(0.0, 1 / N)
        c = self.weights[0]
        i = 0
        idx = 0
        new_particles = np.zeros((self.particles.shape[0], 3))
        for m in range(N):
            u = r + (m / N)
            while u > c:
                i += 1
                c += self.weights[i]
            new_particles[idx, :] = self.particles[i, :]
            idx += 1
        self.particles = new_particles
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    @staticmethod
    def normalize_angle(angle):
        if -np.pi < angle <= np.pi:
            return angle
        if angle > np.pi:
            angle = angle - 2 * np.pi
        if angle <= -np.pi:
            angle = angle + 2 * np.pi
        return ParticlesFilter.normalize_angle(angle)

    @staticmethod
    def normalize_angles_array(angles):
        z = np.zeros_like(angles)
        for i in range(angles.shape[0]):
            z[i] = ParticlesFilter.normalize_angle(angles[i])
        return z

    def bestKParticles(self, K):
        indexes = np.argsort(-self.weights)
        bestK = indexes[:K]
        return self.particles[bestK, :]

