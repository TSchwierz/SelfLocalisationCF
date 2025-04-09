import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

def plot_prediction_path(y, y_pred, mse_mean, ID=0):
    '''
    Plots the predicted path and the actual path and saves the plot under the results folder in the relative directory

    :params y: np.array (ntime, ndim), actual path
    :params y_pred: np.array (ntime, ndim), predicted path
    :params mse_mean: float, the mean-square-error of the prediction
    '''
    stop = int(len(y)/5)
    stop = np.clip(stop, 10, 10000)

    if np.shape(y)[1]==2: # 2D
        plt.figure(figsize=(12, 7))

        fig, axs = plt.subplots(2,1)
        fig.suptitle(f'Actual vs Predicted Path. Total MSE = {mse_mean}')

        axs[0].plot(y[:stop, 0], y[:stop, 1], 'b-', label='Actual')
        axs[0].plot(y_pred[:stop, 0], y_pred[:stop, 1], 'r,', label='Predicted')
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[0].legend()
        axs[0].grid()
        axs[0].set_title(f'First {stop} iterations.')
    
        axs[1].plot(y[-stop:, 0], y[-stop:, 1], 'b-', label="Actual", alpha=0.6) 
        axs[1].plot(y_pred[-stop:, 0], y_pred[-stop:, 1], 'r,', label="Predicted", alpha=0.6)
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].legend()
        axs[1].grid()
        axs[1].set_title(f'Last {stop} iterations.')

        plt.tight_layout()
        plt.savefig(f'Results\\ID{ID}\\result_prediction.png', format='png')
        plt.close()

    elif np.shape(y)[1]==3:  # 3D
        
        # Unpack the actual and predicted coordinates
        x, y_actual, z = y[:, 0], y[:, 1], y[:, 2]
        x_pred, y_pred_val, z_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

        # Create a new figure and 3D axis
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the actual and predicted paths
        ax.plot(x[:stop], y_actual[:stop], z[:stop], label='Actual path', color='blue')
        ax.plot(x_pred[:stop], y_pred_val[:stop], z_pred[:stop], 'r,', label='Predicted path')

        # Determine constant projection planes.
        # For example, we can set each projection at the minimum value of that coordinate.
        x_min, y_min, z_min = np.min(x), np.min(y_actual), np.min(z)

        # Projection onto the xy-plane (fix z = z_min)
        ax.plot(x[:stop], y_actual[:stop], zs=z_min, zdir='z',
                label='XY projection', color='green', linestyle='--', alpha=0.5)

        # Projection onto the xz-plane (fix y = y_min)
        ax.plot(x[:stop], z[:stop], zs=y_min, zdir='y',
                label='XZ projection', color='purple', linestyle='--', alpha=0.5)

        # Projection onto the yz-plane (fix x = x_min)
        ax.plot(y_actual[:stop], z[:stop], zs=x_min, zdir='x',
                label='YZ projection', color='orange', linestyle='--', alpha=0.5)

        # Label axes, legend, and set title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f'First {stop} iterations. Mean MSE = {mse_mean}')

        plt.tight_layout()
        # Save and show the figure
        plt.savefig(f'Results\\ID{ID}\\prediction.png', format='png')
        plt.show()
        plt.close()


def fit_linear_model(activity_array, pos, return_shuffled=False, alpha=1.0, cv_folds=10, seed=42):
    '''
    Predicts the location of the agent based on the activity level of the network

    :param activity_array: np.array featuring the time history of network activity. shape (ntime, (nmodules,) ngain, nneuron)
    :param pos: list of shape (ntime, ndim), position log of the agent
    :param return_shuffled: bool, default=False
    :param alpha: float, parameter used for the regression model
    :param cv_folds: int, amount of folds to divide the data into

    Returns:
        tuple: (X (np.array), y (np.array), y_pred (np.array), mse_mean (float), r2_mean (float)) OR
        tuple: (X (np.array), y (np.array), y_pred (np.array), mse_mean (float), mse_mean_shuffeled (float), r2_mean (float), r2_mean_shuffeled)
    '''
    np.random.seed(seed)

    X = np.array(activity_array).reshape(np.shape(activity_array)[0],-1)  # shape is (time, modules*gains*N)
    y = np.array(pos)  # shape is (time, 3)

    # Initialize the Ridge regression model with specified alpha
    model = Ridge(alpha=alpha)
    # model = LinearRegression()

    # Perform K-Fold cross-validation
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Cross-validation scores for MSE
    mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1) #n_jobs <- enables parallel processing
    #          parallel processing seems to malfunction with cleaning up temporary files, possible leading to memory leakage

    # Cross-validation scores for R2
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
    # Crossval estimates
    y_pred = cross_val_predict(model, X, y, cv=kf)

    # Compute the average and standard deviation of MSE and R2 scores across folds
    mse_mean = round(np.mean(mse_scores), 5)
    mse_std = round(np.std(mse_scores), 5)
    r2_mean = round(np.mean(r2_scores), 5)
    r2_std = round(np.std(r2_scores), 5)

    if return_shuffled == False:
        return X, y, y_pred, mse_mean, r2_mean

    else:  # Fit linear model with shuffled labels
        y_shuffled = y.copy()
        np.random.shuffle(y_shuffled)  # Shuffle the labels
        mse_shuffled_scores = -cross_val_score(model, X, y_shuffled, cv=kf, scoring='neg_mean_squared_error') # Cross-validation scores for MSE with shuffled labels         
        r2_shuffled_scores = cross_val_score(model, X, y_shuffled, cv=kf, scoring='r2') # Cross-validation scores for R2 with shuffled labels

        # Compute the average and standard deviation of MSE and R2 scores for shuffled data
        mse_shuffled_mean = round(np.mean(mse_shuffled_scores), 5)
        mse_shuffled_std = round(np.std(mse_shuffled_scores), 5)
        r2_shuffled_mean = round(np.mean(r2_shuffled_scores), 5)
        r2_shuffled_std = round(np.std(r2_shuffled_scores), 5)

        return X, y, y_pred, mse_mean, mse_shuffled_mean, r2_mean, r2_shuffled_mean


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