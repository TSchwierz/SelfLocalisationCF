'''
Module for fitting linear models with convergence tracking and comparing Ridge regression with Recursive Least Squares (RLS).
'''
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from numba import jit, cuda

def fit_linear_model(x_train, y_train, x_test, y_test, model=None):
    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("x_train and y_train must have the same length, and x_test and y_test must have the same length.")

    if model is None:
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return y_pred, mse, r2, model
    

def fit_linear_model_old(activity_array, pos, train_index=None, return_shuffled=False, alpha=1.0, cv_folds=8, seed=42):
    '''
    Predicts the location of the agent based on the activity level of the network
    :param activity_array: np.array featuring the time history of network activity. shape (ntime, (nmodules,) ngain, nneuron)
    :param pos: list of shape (ntime, ndim), position log of the agent
    :param train_index: int, optional. Index indicating the last sample that is part of training data.
                       If None, uses cross-validation on the entire dataset.
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
    #t1 = perf_counter()
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
        mse_mean = np.mean(mse_scores)
        r2_mean = np.mean(r2_scores)
        
        if not return_shuffled:
            return X, y, y_pred, mse_mean, r2_mean#, perf_counter()-t1
        else:
            # Fit linear model with shuffled labels
            y_shuffled = y.copy()
            np.random.shuffle(y_shuffled)
            
            mse_shuffled_scores = -cross_val_score(model, X, y_shuffled, cv=kf, scoring='neg_mean_squared_error')
            r2_shuffled_scores = cross_val_score(model, X, y_shuffled, cv=kf, scoring='r2')
            
            mse_shuffled_mean = np.mean(mse_shuffled_scores)
            r2_shuffled_mean = np.mean(r2_shuffled_scores)
            
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
            return [X_train, X_test], [y_train, y_test], y_pred_test, mse_test, r2_test#, perf_counter()-t1
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

# ---------------- Optimised RLS --------------------------------
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

        # Convergence tracking
        #self.convergence_metrics = {
        #    'weight_changes': [],
        ##    'weight_norms': [],
         #   'prediction_errors': [],
         #   'gain_norms': [],
         #   'innovation_norms': [],
        #    'timestamps': []
        #}
        
        self._prev_A = self.A.copy()
        #self._update_count = 0
        
    def update(self, y, x):
        """
        Update the regression weights using a new data pair.
        
        Parameters:
        y (np.ndarray): Input feature vector (neural activity) of shape (num_features,).
        x (np.ndarray): Target output (e.g., 2D position) of shape (num_outputs,).
        
        Returns:
        np.ndarray: Updated weight matrix.
        """
        #t1 = perf_counter() # for execution time measurement
        # Ensure column vector format
        y_col = y.reshape(-1, 1)  # shape: (num_features, 1)
        x_col = x.reshape(-1, 1)  # shape: (num_outputs, 1)

        #self._update_count+=1
        
        # Reuse pre-allocated arrays
        np.dot(self.P, y_col, out=self._Py)
        self._yTPy = float(np.dot(y_col.T, self._Py)) + self.eps
        denom = self.lambda_ + self._yTPy
        
        # Compute gain vector
        np.divide(self._Py, denom, out=self._K)
        
        # Compute prediction error
        np.dot(self.A.T, y_col, out=self._error)
        np.subtract(x_col, self._error, out=self._error)

        ## Track convergence metrics before update
        #if self._update_count > 1:
        #    weight_change = np.linalg.norm(self.A - self._prev_A, 'fro')
        #    pred_error = np.mean(self._error ** 2)
        #    innovation_norm = np.linalg.norm(self._error)

        #    self.convergence_metrics['weight_changes'].append(weight_change)           
        #    self.convergence_metrics['prediction_errors'].append(pred_error)           
        #    self.convergence_metrics['innovation_norms'].append(innovation_norm)
        
        # Store previous weights
        self._prev_A = self.A.copy()
        
        # Update weights
        self.A += np.dot(self._K, self._error.T)

        ## Track post-update metrics
        #weight_norm = np.linalg.norm(self.A, 'fro')
        #gain_norm = np.linalg.norm(self._K)
        #self.convergence_metrics['weight_norms'].append(weight_norm)
        #self.convergence_metrics['gain_norms'].append(gain_norm)
              
        # Update inverse correlation matrix
        np.dot(self._K, np.dot(y_col.T, self.P), out=self._KyTP)
        np.subtract(self.P, self._KyTP, out=self.P)
        np.divide(self.P, self.lambda_, out=self.P)
       
        return self.A#, perf_counter()-t1

    def predict(self, y):
        """
        Predict the output given a new feature vector.
        
        Parameters:
        y (np.ndarray): Input feature vector of shape (num_features,).
        
        Returns:
        np.ndarray: Predicted output (2D/3D position).
        """
        y = y.reshape(-1, 1)
        x_est = np.dot(self.A.T, y)
        return x_est.flatten()