import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

def fit_linear_model(activity_array, pos, train_index=None, return_shuffled=False, alpha=1.0, cv_folds=2, seed=42):
    '''
    Predicts the location of the agent based on the activity level of the network
    :param activity_array: np.array featuring the time history of network activity. shape (ntime, (nmodules,) ngain, nneuron)
    :param pos: list of shape (ntime, ndim), position log of the agent
    :param train_index: int, optional. Index indicating the last sample that is part of training data.
                       If None, uses cross-validation on the entire dataset (original behavior).
                       If provided, samples 0:train_index+1 are used for training, train_index+1: for testing.
    :param return_shuffled: bool, default=False
    :param alpha: float, parameter used for the regression model
    :param cv_folds: int, amount of folds to divide the data into (only used when train_index is None)
    Returns:
        When train_index is None (cross-validation mode):
            tuple: (X, y, y_pred, mse_mean, r2_mean) OR
            tuple: (X, y, y_pred, mse_mean, mse_mean_shuffled, r2_mean, r2_mean_shuffled)
        When train_index is provided (train/test split mode):
            tuple: (X_train, X_test, y_train, y_test, y_pred_test, mse_test, r2_test) OR
            tuple: (X_train, X_test, y_train, y_test, y_pred_test, mse_test, mse_test_shuffled, r2_test, r2_test_shuffled)
    '''
    np.random.seed(seed)
    X = np.array(activity_array).reshape(np.shape(activity_array)[0], -1)  # shape is (time, modules*gains*N)
    y = np.array(pos)  # shape is (time, ndim)
    
    # Initialize the Ridge regression model with specified alpha
    model = Ridge(alpha=alpha)
    
    if train_index is None:
        # Original behavior: cross-validation on entire dataset
        # Perform K-Fold cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        
        # Cross-validation scores for MSE and R2
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
        
        # Cross-validation predictions
        y_pred = cross_val_predict(model, X, y, cv=kf)
        
        # Compute the average scores
        mse_mean = round(np.mean(mse_scores), 5)
        r2_mean = round(np.mean(r2_scores), 5)
        
        if not return_shuffled:
            return X, y, y_pred, mse_mean, r2_mean
        else:
            # Fit linear model with shuffled labels
            y_shuffled = y.copy()
            np.random.shuffle(y_shuffled)
            
            mse_shuffled_scores = -cross_val_score(model, X, y_shuffled, cv=kf, scoring='neg_mean_squared_error')
            r2_shuffled_scores = cross_val_score(model, X, y_shuffled, cv=kf, scoring='r2')
            
            mse_shuffled_mean = round(np.mean(mse_shuffled_scores), 5)
            r2_shuffled_mean = round(np.mean(r2_shuffled_scores), 5)
            
            return X, y, y_pred, mse_mean, mse_shuffled_mean, r2_mean, r2_shuffled_mean
    
    else:
        # New behavior: train/test split
        if train_index < 0 or train_index >= len(X):
            raise ValueError(f"train_index must be between 0 and {len(X)-1}")
        
        # Split the data
        X_train, X_test = X[:train_index+1], X[train_index+1:]
        y_train, y_test = y[:train_index+1], y[train_index+1:]
        
        if len(X_test) == 0:
            raise ValueError("No test data available. train_index is too large.")
        
        # Train the model on training data
        model.fit(X_train, y_train)
        
        # Make predictions on test data
        y_pred_test = model.predict(X_test)
        
        # Calculate test metrics
        mse_test = round(mean_squared_error(y_test, y_pred_test), 5)
        r2_test = round(r2_score(y_test, y_pred_test), 5)
        
        if not return_shuffled:
            return X_train, X_test, y_train, y_test, y_pred_test, mse_test, r2_test
        else:
            # Train model with shuffled training labels and test
            y_train_shuffled = y_train.copy()
            np.random.shuffle(y_train_shuffled)
            
            model_shuffled = Ridge(alpha=alpha)
            model_shuffled.fit(X_train, y_train_shuffled)
            y_pred_test_shuffled = model_shuffled.predict(X_test)
            
            mse_test_shuffled = round(mean_squared_error(y_test, y_pred_test_shuffled), 5)
            r2_test_shuffled = round(r2_score(y_test, y_pred_test_shuffled), 5)
            
            return X_train, X_test, y_train, y_test, y_pred_test, mse_test, mse_test_shuffled, r2_test, r2_test_shuffled

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

# ---------------- Optimised --------------------------------
class OptimisedRLS:
    def __init__(self, num_features, num_outputs, lambda_=0.99, delta=1.0, eps=1e-10):
        """
        Initialize the Recursive Least Squares filter.
        
        Parameters:
        num_features (int): Number of input features.
        num_outputs (int): Number of output dimensions.
        lambda_ (float): Forgetting factor (0 < lambda <= 1).
        delta (float): Initial value for P matrix diagonal.
        eps (float): Small constant to avoid division by zero.
        """
        self.lambda_ = lambda_
        self.eps = eps
        
        # Initialize weight matrix (num_features x num_outputs)
        self.A = np.zeros((num_features, num_outputs))
        
        # Initialize inverse correlation matrix (num_features x num_features)
        self.P = np.eye(num_features) * delta
        
        # Pre-allocate memory for commonly used arrays
        self._Py = np.zeros((num_features, 1))
        self._yTPy = 0.0
        self._K = np.zeros((num_features, 1))
        self._error = np.zeros((num_outputs, 1))
        self._KyTP = np.zeros((num_features, num_features))
        
    def update(self, y, x):
        """
        Update the regression weights using a new data pair.
        
        Parameters:
        y (np.ndarray): Input feature vector (neural activity) of shape (num_features,).
        x (np.ndarray): Target output (e.g., 2D position) of shape (num_outputs,).
        
        Returns:
        np.ndarray: Updated weight matrix.
        """
        # Ensure column vector format
        y_col = y.reshape(-1, 1)  # shape: (num_features, 1)
        x_col = x.reshape(-1, 1)  # shape: (num_outputs, 1)
        
        # Reuse pre-allocated arrays
        np.dot(self.P, y_col, out=self._Py)
        self._yTPy = float(np.dot(y_col.T, self._Py)) + self.eps
        denom = self.lambda_ + self._yTPy
        
        # Compute gain vector
        np.divide(self._Py, denom, out=self._K)
        
        # Compute prediction error
        np.dot(self.A.T, y_col, out=self._error)
        np.subtract(x_col, self._error, out=self._error)
        
        # Update weights
        self.A += np.dot(self._K, self._error.T)
        
        # Update inverse correlation matrix
        np.dot(self._K, np.dot(y_col.T, self.P), out=self._KyTP)
        np.subtract(self.P, self._KyTP, out=self.P)
        np.divide(self.P, self.lambda_, out=self.P)
        
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

# ---- RLS with Tracking ------
class RLSRegressorWithTracking:
    def __init__(self, num_features, num_outputs=2, lambda_=0.99, delta=1e5):
        """Enhanced RLS with weight correction tracking"""
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.lambda_ = lambda_
        self.A = np.zeros((num_features, num_outputs))
        self.P = np.eye(num_features) * delta
        self.eps = 1e-5
        
        # Tracking arrays
        self.weight_corrections = []
        self.weight_norms = []
        self.prediction_errors = []
        self.time_step = 0

    def update(self, y, x):
        """Update with tracking of weight corrections"""
        # Store previous weights
        A_prev = self.A.copy()
        
        # Standard RLS update
        y = y.reshape(-1, 1)
        x = x.reshape(-1, 1)
        
        denom = self.lambda_ + np.dot(y.T, np.dot(self.P, y)) + self.eps
        K = np.dot(self.P, y) / denom
        error = x - np.dot(self.A.T, y)
        self.A = self.A + np.dot(K, error.T)
        self.P = (self.P - np.dot(K, np.dot(y.T, self.P))) / self.lambda_
        
        # Track metrics
        weight_correction = np.linalg.norm(self.A - A_prev, 'fro')
        weight_norm = np.linalg.norm(self.A, 'fro')
        prediction_error = np.linalg.norm(error)
        
        self.weight_corrections.append(weight_correction)
        self.weight_norms.append(weight_norm)
        self.prediction_errors.append(prediction_error)
        self.time_step += 1
        
        return self.A

    def predict(self, y):
        """Standard prediction"""
        y = y.reshape(-1, 1)
        x_est = np.dot(self.A.T, y)
        return x_est.flatten()
    
    def get_convergence_metrics(self):
        """Return convergence analysis metrics"""
        corrections = np.array(self.weight_corrections)
        norms = np.array(self.weight_norms)
        errors = np.array(self.prediction_errors)
        
        return {
            'weight_corrections': corrections,
            'weight_norms': norms,
            'prediction_errors': errors,
            'convergence_rate': self._estimate_convergence_rate(corrections),
            'final_correction_mean': np.mean(corrections[-100:]) if len(corrections) > 100 else np.mean(corrections),
            'correction_trend': self._compute_trend(corrections)
        }
    
    def _estimate_convergence_rate(self, corrections, window=100):
        """Estimate convergence rate from recent corrections"""
        if len(corrections) < window:
            return None
        recent_corrections = corrections[-window:]
        # Fit exponential decay: y = a * exp(-b*x)
        x = np.arange(len(recent_corrections))
        try:
            log_corrections = np.log(recent_corrections + 1e-10)
            coeffs = np.polyfit(x, log_corrections, 1)
            return -coeffs[0]  # Negative of slope gives convergence rate
        except:
            return None
    
    def _compute_trend(self, corrections, window=50):
        """Compute trend in recent corrections"""
        if len(corrections) < window:
            return 0
        recent = corrections[-window:]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        return slope