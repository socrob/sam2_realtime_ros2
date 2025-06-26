import numpy as np

class EKF:
    def __init__(self, process_noise_cov, initial_state, initial_covariance, dt):
        self.state = np.array(initial_state)
        self.last_measurement = initial_state
        self.covariance = np.array(initial_covariance)
        self.Q = np.array(process_noise_cov)
        self.dt = dt

        # Transition matrix for constant velocity model
        self.F = np.eye(6)
        self.F[0, 3] = self.dt  # Position update for x
        self.F[1, 4] = self.dt  # Position update for y
        self.F[2, 5] = self.dt  # Position update for z

        # Observation matrix (position-only observation model)
        self.H = np.array([[1, 0, 0, 0, 0, 0],  # x
                           [0, 1, 0, 0, 0, 0],  # y
                           [0, 0, 1, 0, 0, 0]]) # z
    

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q
    

    def update(self, measurement, dynamic_R):
        prediction = np.dot(self.H, self.state)
        innovation = measurement - prediction
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + dynamic_R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, innovation)

        I_KH = np.eye(self.state.shape[0]) - np.dot(K, self.H)
        self.covariance = np.dot(np.dot(I_KH, self.covariance), I_KH.T) + np.dot(np.dot(K, dynamic_R), K.T)


    def get_state(self):
        return self.state[:3]