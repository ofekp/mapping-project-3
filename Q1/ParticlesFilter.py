import numpy as np


class ParticlesFilter:
    def __init__(self, worldLandmarks, sigma_r1, sigma_t, sigma_r2, sigma_range, sigma_bearing, numberOfPaticles=1000):
        """
        Initialization of the particle filter
        """

        # Initialize parameters
        self.numberOfParticles = numberOfPaticles
        self.worldLandmarks = worldLandmarks

        self.sigma_r1 = sigma_r1
        self.sigma_t = sigma_t
        self.sigma_r2 = sigma_r2

        self.sigma_range = sigma_range
        self.sigma_bearing = sigma_bearing

        # Initialize particles - x, y, heading, weight (uniform weight for initialization)
        self.particles = np.concatenate((np.random.normal(0, 2.0, (self.numberOfParticles, 1)),
                                         np.random.normal(0, 2.0, (self.numberOfParticles, 1)),
                                         ParticlesFilter.normalize_angles_array(np.random.normal(0.1, 0.1, (self.numberOfParticles, 1)))), axis=1)

        self.weights = np.ones(self.numberOfParticles) * (1.0 / self.numberOfParticles)
        self.history = np.array((0, 0, 0.1)).reshape(1, 3)
        self.particles_history = np.expand_dims(self.particles.copy(), axis=0)

    def apply(self, Zt, Ut):
        """
        apply the particle filter on a single step in the sequence
        Parameters:
            Zt - the sensor measurement (range, bearing) as seen from the current position of the car
            Ut - the true odometry control command
        """

        # Motion model based on odometry
        self.motionModel(Ut)

        # Observation model
        observations = self.Observation()

        # Observation model
        self.weightParticles(Zt, observations)

        self.history = np.concatenate((self.history, self.bestKParticles(1).reshape(1, 3)), axis=0)

        # Resample particles
        self.resampleParticles()

        self.particles_history = np.concatenate((self.particles_history, np.expand_dims(self.particles.copy(), axis=0)), axis=0)

    def motionModel(self, odometry):
        """
        Apply the odometry motion model to the particles
        odometry - the true odometry control command
        the particles will be updated with the true odometry control command
        in addition, each particle will separately be added with Gaussian noise to its movement
        """
        dr1 = odometry['r1'] + np.random.normal(0, self.sigma_r1, (self.numberOfParticles, 1))
        dt  = odometry['t']  + np.random.normal(0, self.sigma_t,  (self.numberOfParticles, 1))
        dr2 = odometry['r2'] + np.random.normal(0, self.sigma_r2, (self.numberOfParticles, 1))
        theta = self.particles[:, 2].reshape(-1, 1)
        dMotion = np.concatenate((
            dt * np.cos(theta + dr1),
            dt * np.sin(theta + dr1),
            dr1 + dr2), axis=1)
        self.particles = self.particles + dMotion
        self.particles[:, 2] = ParticlesFilter.normalize_angles_array(self.particles[:, 2])

    def Observation(self):
        """
        Calculates the sensor measurement from the perspective of each of the particles
        returns: an array of size (number of particles x 2)
                 the first coord is the range to the closest landmark and the second coord is the bearing to it in radians
                 the bearing and range are added with gaussian noise with STD of 1.0 and 0.1 respectively
        """
        observations = np.zeros((self.particles.shape[0], 2))  # range and bearing for each
        for i, particle in enumerate(self.particles):
            # closest_landmark_id = np.argmin(np.linalg.norm(self.worldLandmarks - particle[0:2], axis=1))
            closest_landmark_id = np.argmin(np.sum((self.worldLandmarks - particle[0:2]) ** 2, axis=1))
            dist_xy = self.worldLandmarks[closest_landmark_id] - particle[0:2]
            r = np.linalg.norm(dist_xy)
            phi = ParticlesFilter.normalize_angle(np.arctan2(dist_xy[1], dist_xy[0]) - particle[2])
            observations[i, 0] = r
            observations[i, 1] = phi
            r += np.random.normal(0, self.sigma_range)
            phi += np.random.normal(0, self.sigma_bearing)
        return observations

    def weightParticles(self, world_measurement, observations):
        """
        Update the particle weights according to the normal Mahalanobis distance
        Parameters:
            world_measurement - the sensor measurement (range, bearing) as seen from the position of the car
            observations - the sensor measurement (range, bearing) as seen by every particle
        """
        cov = np.diag([self.sigma_range ** 2, self.sigma_bearing ** 2])
        for i, observation in enumerate(observations):
            d = world_measurement - observation
            d[1] = ParticlesFilter.normalize_angle(d[1])
            self.weights[i] = np.exp(-0.5 * np.dot(d.T, np.dot(np.linalg.inv(cov), d))) / np.sqrt(np.linalg.det(2 * np.pi * cov))
        self.weights += 1.0e-200  # for numerical stability
        self.weights /= sum(self.weights)

    def resampleParticles(self):
        """
        law variance resampling
        """
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
        """
        Normalize an angle to the range [-pi, pi]
        """
        while angle < -np.pi:
            angle += 2 * np.pi
        while angle >= np.pi:
            angle -= 2 * np.pi
        return angle
        # if -np.pi < angle <= np.pi:
        #     return angle
        # if angle > np.pi:
        #     angle = angle - 2 * np.pi
        # if angle <= -np.pi:
        #     angle = angle + 2 * np.pi
        # return ParticlesFilter.normalize_angle(angle)

    @staticmethod
    def normalize_angles_array(angles):
        """
        applies normalize_angle on an array of angles
        """
        z = np.zeros_like(angles)
        for i in range(angles.shape[0]):
            z[i] = ParticlesFilter.normalize_angle(angles[i])
        return z

    def bestKParticles(self, K):
        """
        Given the particles and their weights, choose the top K particles according to the weights and return them
        """
        indexes = np.argsort(-self.weights)
        bestK = indexes[:K]
        return self.particles[bestK, :]

