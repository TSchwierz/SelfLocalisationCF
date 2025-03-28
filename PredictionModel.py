import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

def plot_prediction_path(y, y_pred, mse_mean, ID=0):
    '''
    Plots the predicted path and the actual path and saves the plot under the results folder in the relative directory

    :params y: np.array (ntime, ndim), actual path
    :params y_pred: np.array (ntime, ndim), predicted path
    :params mse_mean: float, the mean-square-error of the prediction
    '''
    size = int(len(y)/5)
    size = np.clip(size, 10, 10000)
    plt.figure(figsize=(12, 7))

    fig, axs = plt.subplots(2,1)
    fig.suptitle(f'Actual vs Predicted Path. Total MSE = {mse_mean}')

    axs[0].plot(y[:size, 0], y[:size, 1], 'b-', label='Actual')
    axs[0].plot(y_pred[:size, 0], y_pred[:size, 1], 'r,', label='Predicted')
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[0].legend()
    axs[0].grid()
    axs[0].set_title(f'First {size} iterations.')
    
    axs[1].plot(y[-size:, 0], y[-size:, 1], 'b-', label="Actual", alpha=0.6) 
    axs[1].plot(y_pred[-size:, 0], y_pred[-size:, 1], 'r,', label="Predicted", alpha=0.6)
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()
    axs[1].grid()
    axs[1].set_title(f'Last {size} iterations.')

    plt.tight_layout()
    plt.savefig(f'Results\\ID{ID}\\result_prediction.png', format='png')
    plt.close()

def makeKFandRLS(feature_dim, pos_dim, process_noise=0.1, measurement_noise=1.):
    
    #A = np.array([[1, 0],
    #          [0, 1]])
    # Process noise covariance matrix (tuning parameter)
    #Q = np.eye(pos_dim) * process_noise

    # Measurement noise covariance matrix (tuning parameter, n x n)
    #R = np.eye(feature_dim) * measurement_noise

    # Observation matrix: mapping from 2D state to n-dimensional neural activity.
    # This needs to be updated using rls on every timestep
    #np.random.seed(42)
    #H = np.random.randn(feature_dim, pos_dim)

    # Initial state: [x, y]
    #x0 = np.array([0, 0])

    # Initial error covariance
    #P0 = np.eye(pos_dim) * measurement_noise

    # Instantiate the Kalman filter
    #kf = KalmanFilter(A, H, Q, R, x0, P0)
    rls = RLSRegressor(feature_dim, pos_dim, lambda_=0.999, delta=1e2)
    return rls

class KalmanFilter:
    def __init__(self, A, H, Q, R, x0, P0):
        """
        Initializes the Kalman Filter.
        
        Parameters:
        A (np.ndarray): State transition matrix.
        H (np.ndarray): Observation (or emission) matrix mapping state to neural activity.
        Q (np.ndarray): Process noise covariance matrix.
        R (np.ndarray): Measurement noise covariance matrix.
        x0 (np.ndarray): Initial state estimate.
        P0 (np.ndarray): Initial error covariance matrix.
        """
        self.A = A      # State transition model
        self.H = H      # Observation model (neural mapping)
        self.Q = Q      # Process noise covariance
        self.R = R      # Measurement noise covariance
        self.x = x0     # Initial state estimate
        self.P = P0     # Initial error covariance

    def predict(self):
        """
        Predict the next state and error covariance.
        """
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        """
        Update the state estimate with a new measurement (neural activity).
        
        Parameters:
        z (np.ndarray): The n-dimensional measurement vector (neural activity).
        
        Returns:
        np.ndarray: The updated state estimate.
        """
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = np.dot((I - np.dot(K, self.H)), self.P)
        return self.x

class RLSRegressor:
    def __init__(self, num_features, num_outputs=2, lambda_=0.99, delta=1e5):
        """
        Initializes the RLS regressor.
        
        Parameters:
        num_features (int): Number of features (e.g., 540 or reduced dimension).
        num_outputs (int): Number of outputs (e.g., 2 for a 2D position).
        lambda_ (float): Forgetting factor (0 < lambda_ <= 1). Closer to 1 means slow forgetting.
        delta (float): Initial scaling factor for the inverse covariance matrix.
        """
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lambda_ = lambda_
        # Weight matrix (mapping from features to outputs), shape: (num_features, num_outputs)
        self.A = np.zeros((num_features, num_outputs))
        # Inverse covariance matrix, initialized as a large multiple of the identity matrix.
        self.P = np.eye(num_features) * delta
        self.eps = 1e-5 # small constant to mitigate blow up in covariance matrix

    def update(self, y, x):
        """
        Update the regression weights using a new data pair.
        
        Parameters:
        y (np.ndarray): Input feature vector (neural activity) of shape (num_features,).
        x (np.ndarray): Target output (e.g., 2D position) of shape (num_outputs,).
        
        Returns:
        np.ndarray: Updated weight matrix.
        """
        # Ensure column vector format for phi and d
        y = y.reshape(-1, 1)  # shape: (num_features, 1)
        x = x.reshape(-1, 1)      # shape: (num_outputs, 1)

        # Compute the denominator (a scalar)
        denom = self.lambda_ + np.dot(y.T, np.dot(self.P, y)) + self.eps

        # Compute the gain vector (shape: num_features x 1)
        K = np.dot(self.P, y) / denom

        # Compute the prediction error (innovation) (shape: num_outputs x 1)
        error = x - np.dot(self.A.T, y)

        # Update the weight matrix; K (num_features x 1) multiplied by error.T (1 x num_outputs)
        self.A = self.A + np.dot(K, error.T)

        # Update the inverse covariance matrix
        self.P = (self.P - np.dot(K, np.dot(y.T, self.P))) / self.lambda_
        return self.A

    def predict(self, y):
        """
        Predict the output given a new feature vector.
        
        Parameters:
        y (np.ndarray): Input feature vector of shape (num_features,).
        
        Returns:
        np.ndarray: Predicted output (e.g., 2D position).
        """
        y = y.reshape(-1, 1)
        x_est = np.dot(self.A.T, y)
        return x_est.flatten()