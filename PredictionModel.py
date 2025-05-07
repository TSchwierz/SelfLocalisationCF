import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

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

class RLSRegressor:
    def __init__(self, num_features, num_outputs=2, lambda_=0.99, delta=1e5):
        """
        Initializes the RLS regressor.
        
        Parameters:
        num_features (int): Number of features (e.g., 540 or reduced dimension of activity).
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