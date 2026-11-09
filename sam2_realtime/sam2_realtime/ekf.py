import numpy as np


class EKF:
    def __init__(self, process_noise_cov, initial_state, initial_covariance, dt):
        self.state = np.array(initial_state, dtype=np.float64)
        self.last_measurement = np.array(initial_state, dtype=np.float64)
        self.covariance = np.array(initial_covariance, dtype=np.float64)
        self.Q = np.array(process_noise_cov, dtype=np.float64)
        self.dt = dt

        # State: [x, y, z, vx, vy, vz]
        self.F = np.eye(6)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        # Observe position only
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=np.float64)

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q

    def update(self, measurement, dynamic_R):
        measurement = np.array(measurement, dtype=np.float64)

        prediction = np.dot(self.H, self.state)
        innovation = measurement - prediction

        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + dynamic_R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))

        self.state = self.state + np.dot(K, innovation)

        I_KH = np.eye(self.state.shape[0]) - np.dot(K, self.H)
        self.covariance = (
            np.dot(np.dot(I_KH, self.covariance), I_KH.T)
            + np.dot(np.dot(K, dynamic_R), K.T)
        )

    def damp_velocity(self, factor: float):
        factor = float(np.clip(factor, 0.0, 1.0))
        self.state[3:6] *= factor

    def zero_velocity(self):
        self.state[3:6] = 0.0

    def clamp_velocity(self, max_speed: float):
        v = self.state[3:6]
        speed = np.linalg.norm(v)

        if speed > max_speed and speed > 1e-6:
            self.state[3:6] = v * (max_speed / speed)

    def get_state(self):
        return self.state[:3]

    def get_velocity(self):
        return self.state[3:6].copy()

    def get_full_state(self):
        return self.state.copy()