import numpy as np
import matplotlib.pyplot as plt


class ParticlesFilter:
    def __init__(self, worldLandmarks, sigma_r1, sigma_t, sigma_r2, numberOfPaticles=1000):

        # Initialize parameters
        self.numberOfParticles = numberOfPaticles
        self.worldLandmarks = worldLandmarks

        self.sigma_r1 = sigma_r1
        self.sigma_t = sigma_t
        self.sigma_r2 = sigma_r2

        # Initialize particles - x, y, heading, weight (uniform weight for initialization)
        self.particles = np.concatenate((np.random.normal(0, 2.0, (self.numberOfParticles, 1)),
                                    np.random.normal(0, 2.0, (self.numberOfParticles, 1)),
                                    ParticlesFilter.normalize_angles_array(np.random.normal(0.1, 0.1, (self.numberOfParticles, 1)))), axis=1)
        # self.particles = np.array([[0, 0, 0.1], [0.5, 0.5, 0.3], [1.5, 1.5, 1.3]])
        self.weights = np.ones(self.numberOfParticles) * (1.0 / self.numberOfParticles)
        # TODO(ofekp): I assumed below that it is 0,0,0.1, should probably take it from the trajecotry with the noise
        self.history = np.array((0, 0, 0.1)).reshape(1, 3)

    def apply(self, Zt, Ut):

        # Motion model based on odometry
        self.motionModel(Ut)

        # Observation model
        self.Observation()

        # Observation model
        self.weightParticles(Zt)

        self.history = np.concatenate((self.history, self.bestKParticles(1).reshape(1, 3)), axis=0)

        # Resample particles
        self.resampleParticles()

    def motionModel(self, odometry):
        dr1 = ParticlesFilter.normalize_angles_array(odometry['r1'] + np.random.normal(0, self.sigma_r1, (self.numberOfParticles, 1)))
        dt = odometry['t'] + np.random.normal(0, self.sigma_t, (self.numberOfParticles, 1))
        dr2 = ParticlesFilter.normalize_angles_array(odometry['r2'] + np.random.normal(0, self.sigma_r2, (self.numberOfParticles, 1)))
        theta = self.particles[:, 2].reshape(-1, 1)
        # TODO(ofekp): does each particle get its own random odometry movement, or do they all get the movement I randomized before
        dMotion = np.concatenate((
            dt * np.cos(theta + dr1),
            dt * np.sin(theta + dr1),
            np.ones(self.particles.shape[0]).reshape(-1, 1) * ParticlesFilter.normalize_angles_array(dr1 + dr2)), axis=1)
        self.particles = self.particles + dMotion

    def Observation(self):
        # TODO(ofekp): I am calculating this outside, is that ok?
        pass

    def weightParticles(self, worldMeasurment):
        cov = np.diag([1.0, 0.1])
        observations = np.zeros((self.particles.shape[0], 2))  # range and bearing for each
        for i, particle in enumerate(self.particles):
            closest_landmark_id = np.argmin(np.linalg.norm(self.worldLandmarks - particle[0:2], axis=1))
            dist_xy = self.worldLandmarks[closest_landmark_id] - particle[0:2]
            r = np.linalg.norm(dist_xy)
            phi = np.arctan2(dist_xy[1], dist_xy[0])
            observations[i, 0] = r
            observations[i, 1] = phi
        for i, observation in enumerate(observations):
            d = observation - worldMeasurment
            d[1] = ParticlesFilter.normalize_angle(d[1])
            self.weights[i] = (1 / np.sqrt(np.linalg.det(2 * np.pi * cov))) * np.exp(-0.5 * np.dot(np.dot(d.T, np.linalg.pinv(cov)), d))
            # self.weights[i] = np.sqrt(np.dot(np.dot(d.T, np.linalg.pinv(cov)), d))
        # self.weights = np.max(self.weights) - self.weights
        self.weights += 1.e-50  # for numerical stability
        self.weights /= sum(self.weights)

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

    def resampleParticles(self):
        N = len(self.weights)
        indices = []
        r = np.random.uniform() * (1 / N)
        c = self.weights[0]
        i = 0
        for m in range(N):
            u = r + (m - 1) * (1 / N)
            while u > c:
                i += 1
                c += self.weights[i]
            indices.append(i)
        self.particles = self.particles[indices]
        self.weights.resize(len(self.particles))
        self.weights.fill(1.0 / len(self.weights))

    def bestKParticles(self, K):
        indexes = np.argsort(-self.weights)
        bestK = indexes[:K]
        return self.particles[bestK, :]

