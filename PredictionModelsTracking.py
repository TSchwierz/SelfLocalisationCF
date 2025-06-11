import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from tqdm.auto import tqdm

def fit_linear_model_with_convergence(activity_array, pos, train_index=None, return_shuffled=False, alpha=1.0, cv_folds=2, seed=42):
    '''
    Predicts the location of the agent based on the activity level of the network with convergence tracking
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
            tuple: (X, y, y_pred, mse_mean, r2_mean, convergence_metrics) OR
            tuple: (X, y, y_pred, mse_mean, mse_mean_shuffled, r2_mean, r2_mean_shuffled, convergence_metrics)
        When train_index is provided (train/test split mode):
            tuple: (X_train, X_test, y_train, y_test, y_pred_test, mse_test, r2_test, convergence_metrics) OR
            tuple: (X_train, X_test, y_train, y_test, y_pred_test, mse_test, mse_test_shuffled, r2_test, r2_test_shuffled, convergence_metrics)
    '''
    np.random.seed(seed)
    X = np.array(activity_array).reshape(np.shape(activity_array)[0], -1)  # shape is (time, modules*gains*N)
    y = np.array(pos)  # shape is (time, ndim)
    
    # Initialize the Ridge regression model with specified alpha
    model = Ridge(alpha=alpha)
    
    # Convergence tracking for Ridge regression
    convergence_metrics = {
        'weight_changes': [],
        'weight_norms': [],
        'prediction_errors': [],
        'timestamps': [],
        'final_weights': None
    }
    
    if train_index is None:
        # Original behavior: cross-validation on entire dataset
        # For cross-validation, we'll track convergence by training incrementally
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        
        # Track convergence during incremental training
        prev_weights = None
        window_size = max(10, len(X) // 20)  # Adaptive window size
        
        for i in range(window_size, len(X), window_size):
            # Train on increasing subsets
            X_subset = X[:i]
            y_subset = y[:i]
            
            model.fit(X_subset, y_subset)
            current_weights = model.coef_
            
            if prev_weights is not None:
                # Calculate weight change (Frobenius norm)
                weight_change = np.linalg.norm(current_weights - prev_weights, 'fro')
                convergence_metrics['weight_changes'].append(weight_change)
                
                # Calculate prediction error on recent data
                recent_X = X[max(0, i-window_size):i]
                recent_y = y[max(0, i-window_size):i]
                pred_error = np.mean((model.predict(recent_X) - recent_y) ** 2)
                convergence_metrics['prediction_errors'].append(pred_error)
            
            convergence_metrics['weight_norms'].append(np.linalg.norm(current_weights, 'fro'))
            convergence_metrics['timestamps'].append(i)
            prev_weights = current_weights.copy()
        
        convergence_metrics['final_weights'] = model.coef_.copy() if hasattr(model, 'coef_') else None
        
        # Perform K-Fold cross-validation for final metrics
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=-1)
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
        
        # Cross-validation predictions
        y_pred = cross_val_predict(model, X, y, cv=kf)
        
        # Compute the average scores
        mse_mean = round(np.mean(mse_scores), 5)
        r2_mean = round(np.mean(r2_scores), 5)
        
        if not return_shuffled:
            return X, y, y_pred, mse_mean, r2_mean, convergence_metrics
        else:
            # Fit linear model with shuffled labels
            y_shuffled = y.copy()
            np.random.shuffle(y_shuffled)
            
            mse_shuffled_scores = -cross_val_score(model, X, y_shuffled, cv=kf, scoring='neg_mean_squared_error')
            r2_shuffled_scores = cross_val_score(model, X, y_shuffled, cv=kf, scoring='r2')
            
            mse_shuffled_mean = round(np.mean(mse_shuffled_scores), 5)
            r2_shuffled_mean = round(np.mean(r2_shuffled_scores), 5)
            
            return X, y, y_pred, mse_mean, mse_shuffled_mean, r2_mean, r2_shuffled_mean, convergence_metrics
    
    else:
        # New behavior: train/test split with incremental training for convergence tracking
        if train_index < 0 or train_index >= len(X):
            raise ValueError(f"train_index must be between 0 and {len(X)-1}")
        
        # Split the data
        X_train, X_test = X[:train_index+1], X[train_index+1:]
        y_train, y_test = y[:train_index+1], y[train_index+1:]
        
        if len(X_test) == 0:
            raise ValueError("No test data available. train_index is too large.")
        
        # Track convergence during incremental training on training set
        prev_weights = None
        window_size = max(10, len(X_train) // 20)
        
        for i in range(window_size, len(X_train), window_size):
            X_subset = X_train[:i]
            y_subset = y_train[:i]
            
            model.fit(X_subset, y_subset)
            current_weights = model.coef_
            
            if prev_weights is not None:
                weight_change = np.linalg.norm(current_weights - prev_weights, 'fro')
                convergence_metrics['weight_changes'].append(weight_change)
                
                # Prediction error on recent training data
                recent_X = X_train[max(0, i-window_size):i]
                recent_y = y_train[max(0, i-window_size):i]
                pred_error = np.mean((model.predict(recent_X) - recent_y) ** 2)
                convergence_metrics['prediction_errors'].append(pred_error)
            
            convergence_metrics['weight_norms'].append(np.linalg.norm(current_weights, 'fro'))
            convergence_metrics['timestamps'].append(i)
            prev_weights = current_weights.copy()
        
        # Final training on full training set
        model.fit(X_train, y_train)
        convergence_metrics['final_weights'] = model.coef_.copy()
        
        # Make predictions on test data
        y_pred_test = model.predict(X_test)
        
        # Calculate test metrics
        mse_test = round(mean_squared_error(y_test, y_pred_test), 5)
        r2_test = round(r2_score(y_test, y_pred_test), 5)
        
        if not return_shuffled:
            return X_train, X_test, y_train, y_test, y_pred_test, mse_test, r2_test, convergence_metrics
        else:
            # Train model with shuffled training labels and test
            y_train_shuffled = y_train.copy()
            np.random.shuffle(y_train_shuffled)
            
            model_shuffled = Ridge(alpha=alpha)
            model_shuffled.fit(X_train, y_train_shuffled)
            y_pred_test_shuffled = model_shuffled.predict(X_test)
            
            mse_test_shuffled = round(mean_squared_error(y_test, y_pred_test_shuffled), 5)
            r2_test_shuffled = round(r2_score(y_test, y_pred_test_shuffled), 5)
            
            return X_train, X_test, y_train, y_test, y_pred_test, mse_test, mse_test_shuffled, r2_test, r2_test_shuffled, convergence_metrics


# Legacy function kept for compatibility
def fit_linear_model(activity_array, pos, train_index=None, return_shuffled=False, alpha=1.0, cv_folds=2, seed=42):
    '''
    Original function without convergence tracking - kept for backward compatibility
    '''
    # Call the new function and return only the original outputs
    result = fit_linear_model_with_convergence(activity_array, pos, train_index, return_shuffled, alpha, cv_folds, seed)
    # Return all but the last element (convergence_metrics)
    return result[:-1]

class OptimisedRLS:
    def __init__(self, num_features, num_outputs, lambda_=0.99, delta=1.0, eps=1e-10):
        """
        Initialize the Recursive Least Squares filter with convergence tracking.
        
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
        self.convergence_metrics = {
            'weight_changes': [],
            'weight_norms': [],
            'prediction_errors': [],
            'gain_norms': [],
            'innovation_norms': [],
            'timestamps': [],
            'P_condition_numbers': []  # Track condition number of P matrix
        }
        
        self._prev_A = self.A.copy()
        self._update_count = 0
        
    def update(self, y, x):
        """
        Update the regression weights using a new data pair with convergence tracking.
        
        Parameters:
        y (np.ndarray): Input feature vector (neural activity) of shape (num_features,).
        x (np.ndarray): Target output (e.g., 2D position) of shape (num_outputs,).
        
        Returns:
        np.ndarray: Updated weight matrix.
        """
        self._update_count += 1
        
        # Ensure column vector format
        y_col = y.reshape(-1, 1)  # shape: (num_features, 1)
        x_col = x.reshape(-1, 1)  # shape: (num_outputs, 1)
        
        # Reuse pre-allocated arrays
        np.dot(self.P, y_col, out=self._Py)
        self._yTPy = float(np.dot(y_col.T, self._Py)) + self.eps
        denom = self.lambda_ + self._yTPy
        
        # Compute gain vector
        np.divide(self._Py, denom, out=self._K)
        
        # Compute prediction error (innovation)
        np.dot(self.A.T, y_col, out=self._error)
        np.subtract(x_col, self._error, out=self._error)
        
        # Track convergence metrics before update
        if self._update_count > 1:
            # Weight change (Frobenius norm of difference)
            weight_change = np.linalg.norm(self.A - self._prev_A, 'fro')
            self.convergence_metrics['weight_changes'].append(weight_change)
            
            # Prediction error (MSE)
            pred_error = np.mean(self._error ** 2)
            self.convergence_metrics['prediction_errors'].append(pred_error)
            
            # Innovation norm
            innovation_norm = np.linalg.norm(self._error)
            self.convergence_metrics['innovation_norms'].append(innovation_norm)
            
            # Timestamps
            self.convergence_metrics['timestamps'].append(self._update_count)
        
        # Store previous weights
        self._prev_A = self.A.copy()
        
        # Update weights
        self.A += np.dot(self._K, self._error.T)
        
        # Track post-update metrics
        weight_norm = np.linalg.norm(self.A, 'fro')
        self.convergence_metrics['weight_norms'].append(weight_norm)
        
        gain_norm = np.linalg.norm(self._K)
        self.convergence_metrics['gain_norms'].append(gain_norm)
        
        # Track condition number of P matrix (expensive, so do it sparingly)
        if self._update_count % 100 == 0:  # Every 100 updates
            try:
                cond_num = np.linalg.cond(self.P)
                self.convergence_metrics['P_condition_numbers'].append((self._update_count, cond_num))
            except:
                pass  # Skip if computation fails
        
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
    
    def get_convergence_metrics(self):
        """
        Get the convergence metrics collected during training.
        
        Returns:
        dict: Dictionary containing convergence metrics.
        """
        return self.convergence_metrics.copy()
    
    def reset_convergence_tracking(self):
        """
        Reset convergence tracking metrics.
        """
        for key in self.convergence_metrics:
            self.convergence_metrics[key] = []
        self._update_count = 0
        self._prev_A = self.A.copy()


def compare_convergence(ridge_metrics, rls_metrics, save_path=None):
    """
    Compare convergence metrics between Ridge regression and RLS.
    
    Parameters:
    ridge_metrics (dict): Convergence metrics from Ridge regression
    rls_metrics (dict): Convergence metrics from RLS
    save_path (str, optional): Path to save the comparison plot
    
    Returns:
    matplotlib.figure.Figure: The comparison plot figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Convergence Comparison: Ridge vs RLS', fontsize=16)
    
    # Plot 1: Weight Changes
    ax1 = axes[0, 0]
    if ridge_metrics['weight_changes']:
        ax1.semilogy(ridge_metrics['timestamps'][1:], ridge_metrics['weight_changes'], 
                    'b-', label='Ridge', alpha=0.7)
    if rls_metrics['weight_changes']:
        ax1.semilogy(rls_metrics['timestamps'], rls_metrics['weight_changes'], 
                    'r-', label='RLS', alpha=0.7)
    ax1.set_xlabel('Time/Updates')
    ax1.set_ylabel('Weight Change (Frobenius Norm)')
    ax1.set_title('Weight Matrix Changes Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Weight Norms
    ax2 = axes[0, 1]
    if ridge_metrics['weight_norms']:
        ridge_x = ridge_metrics['timestamps'] if len(ridge_metrics['timestamps']) == len(ridge_metrics['weight_norms']) else range(len(ridge_metrics['weight_norms']))
        ax2.plot(ridge_x, ridge_metrics['weight_norms'], 'b-', label='Ridge', alpha=0.7)
    if rls_metrics['weight_norms']:
        rls_x = list(range(1, len(rls_metrics['weight_norms']) + 1))
        ax2.plot(rls_x, rls_metrics['weight_norms'], 'r-', label='RLS', alpha=0.7)
    ax2.set_xlabel('Time/Updates')
    ax2.set_ylabel('Weight Norm (Frobenius)')
    ax2.set_title('Weight Matrix Norms Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Prediction Errors
    ax3 = axes[1, 0]
    if ridge_metrics['prediction_errors']:
        ax3.semilogy(ridge_metrics['timestamps'][1:], ridge_metrics['prediction_errors'], 
                    'b-', label='Ridge', alpha=0.7)
    if rls_metrics['prediction_errors']:
        ax3.semilogy(rls_metrics['timestamps'], rls_metrics['prediction_errors'], 
                    'r-', label='RLS', alpha=0.7)
    ax3.set_xlabel('Time/Updates')
    ax3.set_ylabel('Prediction Error (MSE)')
    ax3.set_title('Prediction Errors Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: RLS-specific metrics (Innovation and Gain norms)
    ax4 = axes[1, 1]
    if rls_metrics.get('innovation_norms'):
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(range(1, len(rls_metrics['innovation_norms']) + 1), 
                        rls_metrics['innovation_norms'], 'g-', alpha=0.7, label='Innovation Norm')
        line2 = ax4_twin.plot(range(1, len(rls_metrics['gain_norms']) + 1), 
                            rls_metrics['gain_norms'], 'm-', alpha=0.7, label='Gain Norm')
        
        ax4.set_xlabel('Updates')
        ax4.set_ylabel('Innovation Norm', color='g')
        ax4_twin.set_ylabel('Gain Norm', color='m')
        ax4.set_title('RLS Innovation and Gain Norms')
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
    else:
        ax4.text(0.5, 0.5, 'No RLS-specific metrics available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('RLS-Specific Metrics')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_convergence_summary(ridge_metrics, rls_metrics):
    """
    Provide a summary analysis of convergence characteristics.
    
    Parameters:
    ridge_metrics (dict): Convergence metrics from Ridge regression
    rls_metrics (dict): Convergence metrics from RLS
    
    Returns:
    dict: Summary statistics and analysis
    """
    summary = {
        'ridge': {},
        'rls': {},
        'comparison': {}
    }
    
    # Ridge analysis
    if ridge_metrics['weight_changes']:
        summary['ridge']['final_weight_change'] = ridge_metrics['weight_changes'][-1]
        summary['ridge']['mean_weight_change'] = np.mean(ridge_metrics['weight_changes'])
        summary['ridge']['convergence_rate'] = np.std(ridge_metrics['weight_changes']) / np.mean(ridge_metrics['weight_changes'])
    
    if ridge_metrics['weight_norms']:
        summary['ridge']['final_weight_norm'] = ridge_metrics['weight_norms'][-1]
        summary['ridge']['weight_norm_stability'] = np.std(ridge_metrics['weight_norms'][-10:]) if len(ridge_metrics['weight_norms']) >= 10 else 0
    
    # RLS analysis
    if rls_metrics['weight_changes']:
        summary['rls']['final_weight_change'] = rls_metrics['weight_changes'][-1]
        summary['rls']['mean_weight_change'] = np.mean(rls_metrics['weight_changes'])
        summary['rls']['convergence_rate'] = np.std(rls_metrics['weight_changes']) / np.mean(rls_metrics['weight_changes'])
        
        # Detect convergence point (when weight changes drop below threshold)
        threshold = np.mean(rls_metrics['weight_changes']) * 0.1
        converged_idx = next((i for i, change in enumerate(rls_metrics['weight_changes']) if change < threshold), None)
        summary['rls']['convergence_point'] = converged_idx
    
    if rls_metrics['weight_norms']:
        summary['rls']['final_weight_norm'] = rls_metrics['weight_norms'][-1]
        summary['rls']['weight_norm_stability'] = np.std(rls_metrics['weight_norms'][-10:]) if len(rls_metrics['weight_norms']) >= 10 else 0
    
    # Comparison
    if ridge_metrics['weight_changes'] and rls_metrics['weight_changes']:
        summary['comparison']['ridge_converges_faster'] = (
            np.mean(ridge_metrics['weight_changes']) < np.mean(rls_metrics['weight_changes'])
        )
        summary['comparison']['relative_convergence_speed'] = (
            np.mean(ridge_metrics['weight_changes']) / np.mean(rls_metrics['weight_changes'])
        )
    
    return summary