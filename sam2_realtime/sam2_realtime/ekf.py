import numpy as np

class EKF:
    def __init__(self, process_noise_cov, initial_state, initial_covariance, dt):
        # State vector [X, Y, Z, Vx, Vy, Vz]
        # Initial state
        self.state = np.array(initial_state)
        self.last_measurement = initial_state
        # Initial covariance matrix
        self.covariance = np.array(initial_covariance)
        
        # Process noise covariance
        self.Q = np.array(process_noise_cov)
        
        # Transition matrix (A) - assuming constant velocity model
        # For simplicity, identity matrix; adapt for more complex motion models
        self.F = np.eye(6)
        
        # Observation matrix (H) - we're observing positions, so it's just a position measurement
        self.H = np.array([[1,0,0,0,0,0],
                           [0,1,0,0,0,0],
                           [0,0,1,0,0,0]])
        # self.H = np.eye(6)
        
        self.dt = dt

        # self.F[0, 3] = self.dt
        # self.F[1, 4] = self.dt
        # self.F[2, 5] = self.dt
    

    def predict(self):
        # Predict the next state assuming constant velocity model
        # self.state[:3] = self.state[:3] + self.state[3:]*self.dt #np.dot(self.F, self.state)
        self.state = np.dot(self.F, self.state)
        self.covariance = np.dot(np.dot(self.F, self.covariance), self.F.T) + self.Q
    

    def update(self, measurement, dynamic_R):
        """
        Update function for the EKF. This includes computing dynamic R based on the measurement.
        The dynamic R is computed outside the EKF update function and passed as a parameter.

        Parameters:
        - measurement: The current measurement
        - dynamic_R: The dynamically computed measurement noise covariance matrix
        """

        # Predicted state based on current state estimate
        prediction = np.dot(self.H, self.state)
        # new_vel = (measurement-self.last_measurement[:3])/self.dt
        # measurement = np.concatenate([measurement, new_vel])

        # Residual (difference between measurement and predicted state)
        innovation = measurement - prediction

        # Kalman Gain (K)
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + dynamic_R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))

        # Update the state with the new measurement
        self.state = self.state + np.dot(K, innovation)

        # Update the covariance matrix
        comp_1 = np.eye(len(self.state)) - np.dot(K, self.H)
        self.covariance = np.dot(np.dot(comp_1, self.covariance), comp_1.T) + np.dot(np.dot(K, dynamic_R), K.T)

        # self.last_measurement = measurement


    def get_state(self):
        # Return only position (X, Y, Z)
        return self.state[:3]