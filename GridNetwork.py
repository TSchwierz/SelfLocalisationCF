from math import isnan, nan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.linear_model import Ridge, LinearRegression
from tqdm import tqdm

class GridNetwork:
    
    def __init__(self, N_x, N_y, Seed=42, beta=0):
        """
        Initialize the network of grid cells

        Args:
            N_x (int): Amount of neurons along the x-axis of a layer
            N_y (int): Amount of neurons along the y-axis of a layer
            Seed (int): Seed used for random number generation. Default=42
            beta (float): The angular orientation of the grid fields. Default=0
        """
        self.N_x =  N_x
        self.N_y = N_y
        self.N = N_x * N_y # N for one layer, not the whole network

        self.distance_matrix = self.initialize_distance_matrix()

        # params of the (dynamic) weight matrix
        self.I = 0.3                                # intensity parameter (overall synaptic strength)
        self.T = 0.05                               # shift parameter (determining excitatory and inhibitory connections)
        self.sigma = 0.24                           # size of the gaussian
        self.tau = 0.8                              # normalization parameter
        self.gains = np.arange(1,3.1,0.5)           # Gain parameters (layers of the network)
        self.beta = beta                               # bias parameter (rotation of the grid)
        self.R = np.array([[np.cos(self.beta), -np.sin(self.beta)],[np.sin(self.beta), np.cos(self.beta)]]) # rotation matrix

        # some extra parameters
        self.n = self.N * len(self.gains)           # actual size of the network
        self.name = 'Grid network'

        # random seed
        self.seed = Seed 
        self.update_network_shape()        # Update network_activity shape based on initial gains     
        
    def update_network_shape(self):
        '''
        Update the number of layers (gains) the network has based on the network size and the shape of gains
        '''
        # np.random.seed(self.seed)
        rng = np.random.default_rng(seed=self.seed) # create a random generator instance (to get locally random seeds)
        self.network_activity = rng.uniform(0, 1 / np.sqrt(self.N), (len(self.gains), self.N))

    def set_gains(self, gains):
        '''
        Set new values for the gains (size of grid field) of the network layers. The size of gains determents the total size of the network. Calls update_network_shape().

        Args:
            gains (tuple): The size of grid fields for each layer of the network.
        '''
        self.gains = gains
        self.update_network_shape()  # Update network_activity shape when gains are changed
        self.n = self.N * len(self.gains) 

    def initialize_distance_matrix(self):
        ''' Vectorized version to initialize the distance matrix between all neurons.
    
        Returns:
            np.array: distance_matrix of shape [N, N, 2]
        '''
        N = self.N
    
        # Create coordinate arrays for the grid
        i = np.arange(1, self.N_x + 1)
        j = np.arange(1, self.N_y + 1)
        x, y = np.meshgrid(i, j)
        x = x.ravel()
        y = y.ravel()
    
        # Compute normalized cell positions
        cx = (x - 0.5) / self.N_x
        cy = (np.sqrt(3) / 2) * (y - 0.5) / self.N_y
        c = np.stack([cx, cy], axis=1)  # Shape: (N, 2)
    
        # Define the possible translations in the twisted toroidal structure
        s_j = np.array([
            [0, 0],
            [-0.5,  np.sqrt(3) / 2],
            [-0.5, -np.sqrt(3) / 2],
            [0.5,  np.sqrt(3) / 2],
            [0.5, -np.sqrt(3) / 2],
            [-1, 0],
            [1, 0]
        ])  # Shape: (7, 2)
    
        # Compute all pairwise differences between neuron coordinates.
        # diff has shape (N, N, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    
        # Add the translations to the differences. Broadcasting produces an array of shape (N, N, 7, 2)
        candidates = diff[:, :, np.newaxis, :] + s_j[np.newaxis, np.newaxis, :, :]
    
        # Compute the Euclidean norms for each candidate translation; result has shape (N, N, 7)
        candidate_norms = np.linalg.norm(candidates, axis=-1)
    
        # For each (i, j), find the index of the translation that gives the minimum norm.
        best_index = np.argmin(candidate_norms, axis=-1)  # Shape: (N, N)
    
        # Use advanced indexing to select the optimal translation for each pair, resulting in shape (N, N, 2)
        best_s = s_j[best_index]
    
        # The distance matrix is the sum of the pairwise differences and the optimal translations.
        distance_matrix = diff + best_s
    
        print('Distance matrix initialized')
        return distance_matrix

    
    #def weight_function(self, I, T, sigma):
     #   ''' 
     #   This is to play around with different values of the parameters of the weight function to understand it better. 
     #   The weight function in the model includes the velocity vector, which here is considered to be 0.
     #   Params:
     #           I = intensity parameter, defines the overall synaptic strenght
     #           sigma = regulates the size of the gaussian
     #           T = shift parameter determining excitatory and inhibitory connections
     #  Outputs: Weight matrix
     #   Note: This functiion is not used to update weights, (there is no input in this one)
     #         This is just to be able to play with the parameters'''
            
     #   W = I * np.exp(-(np.linalg.norm(self.distance_matrix, axis=2)**2)/sigma**2) - T
        
      #  return W
   
    def update_network(self, velocity):
        '''
        Simulate grid-cell activity in real time 

        :param velocity: np.array consisting of 2d velocity
        Returns:
            np.array: (ngains, nneurons) reflecting the updated activity of the network
        '''

        for a, alpha in enumerate(self.gains):  # Iterate over alpha values
            # Update weight matrix based on current velocity and alpha
            W = self.I * np.exp(- (np.linalg.norm(self.distance_matrix + alpha * np.dot(self.R, velocity), axis=2)**2) / self.sigma**2) - self.T
            self.W = W

            # Calculate activity using transfer function
            b_activity = self.network_activity[a, :]
            b_activity = b_activity @ W
            
            # Normalize activity
            net_activity = ((1 - self.tau) * b_activity + self.tau * (b_activity / np.sum(self.network_activity[a, :])))
            net_activity[net_activity < 0] = 0 
            if (np.max(net_activity) != np.min(net_activity)): # normalise only if not all values identical
                net_activity = (net_activity - np.min(net_activity)) / (np.max(net_activity) - np.min(net_activity))        
            
            self.network_activity[a, :] = net_activity # save activity for each gain

        return self.network_activity
           
    #def reset_activity(self):
    #    ''' To reset the activity population in case there are jumps in xpace (the agent is randomly placed in a new location)'''
    #    rng = np.random.default_rng(seed=self.seed)
    #    self.network_activity = rng.uniform(0, 1 / np.sqrt(self.N), (len(self.gains), self.N))

    def plot_frame_figure(self, positions_array, num_bins, network_activity, neuron=42, ID=0):
        """
        Plots a heatmap of network activity at different gain levels and overlays the trajectory. 
        The plot is saved in the results folder within the relative directory.

        :param positions_array: list of shape (ntime, ndim) 
        :param num_bins: int 
        :param network_activity: list of shape (ntime, ngain, nneuron)
        :param neuron: int. Default=42
        """
        x_min, y_min = np.min(positions_array, axis=0)
        x_max, y_max = np.max(positions_array, axis=0)

        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(2, len(self.gains)+1) 

        # Adding subplots to the gridspec
        for a, alpha in enumerate(self.gains):

            heatmap_ax = fig.add_subplot(gs[0, a])
            heatmap_ax.set_aspect('equal')

            # Initialize an empty heatmap
            x_bins = np.linspace(x_min, x_max,num_bins)
            y_bins = np.linspace(y_min, y_max,num_bins)
            heatmap = np.zeros((num_bins, num_bins))

            # Iterate over positions and network_activity (Over time)
            for position, activity in zip(positions_array, network_activity):
                x_index = np.digitize(position[0], x_bins) - 1
                y_index = np.digitize(position[1], y_bins) - 1
                heatmap[x_index, y_index] = max(heatmap[x_index, y_index], np.mean(activity[a, neuron]))
                #                                 Activity is of shape (ngains, neurons) here  ^^ pick any neuron

            im = heatmap_ax.imshow(heatmap.T, origin='lower', extent=[x_min, x_max, y_min, y_max], vmax=1, vmin=0)
            heatmap_ax.set(title=f'Gain = {round(alpha, 2)}', xticks=[], yticks=[])
            # add labels left plot
            if a == 0:
                heatmap_ax.set_xlabel('X axis arena')
                heatmap_ax.set_ylabel('Y axis arena')

        # # add subplot for colorbar (there is sth odd here, max fr is a bit above 1 and not 1)
        cbar_ax = fig.add_subplot(gs[0, -1])  # Spanning the last column
        colorbar = fig.colorbar(im, cax=cbar_ax)
        colorbar.set_label('Normalized firing rate', labelpad=15, rotation=270)
        colorbar.set_ticks([0, 0.5, 1])  # Set ticks at min, mid, and max values
        colorbar.set_ticklabels([f'{0:.2f}', f'{0.5:.2f}', f'{1:.2f}'])  # Set tick labels

        positions_array = np.array(positions_array) # enable numpy slicing      
        trajectory_ax = fig.add_subplot(gs[1, 1:len(self.gains)])  # Adding subplot for the bottom row
        trajectory_ax.plot(positions_array[:, 0], positions_array[:, 1], alpha=0.7, color='purple')
        trajectory_ax.set_title('Effective Arena with travelled path', fontsize=20)
        trajectory_ax.set_aspect('equal')


        fig.tight_layout(h_pad=3.0) # Adjust layout # change spacing between plots
        plt.savefig(f'Results\\ID{ID}\\result_activity_figure.png', format='png') # save in relative folder Results in Source/Repos/SelfLocalisationCF
        plt.close()

    def plot_activity_neurons(self, positions_array, num_bins, neuron_range, network_activity, ID=0):
        """
        Plots a heatmap of network activity at different gain levels for each neuron specified in the neuron range.
        Saves the plot under the results folder in the relative directory

        Parameters:
        - positions_array (list): List of (x, y) positions over time.
        - num_bins (int): Number of bins for the heatmap.
        - network_activity (np.array): Activity of the network, shape (ntime, ngain, nneuron).
        """

        x_min, y_min = np.min(positions_array, axis=0)
        x_max, y_max = np.max(positions_array, axis=0)

        x_bins = np.linspace(x_min, x_max, num_bins)
        y_bins = np.linspace(y_min, y_max, num_bins)

        print(f'Generating activity plot for {len(neuron_range)} neurons')
        for neuron in tqdm(neuron_range):
            fig = plt.figure(figsize=(13, 8))
            gs = fig.add_gridspec(1, len(self.gains) + 1, height_ratios=[1], width_ratios=[1] * len(self.gains) + [0.07])
        
            for a, alpha in enumerate(self.gains):
                heatmap_ax = fig.add_subplot(gs[0, a])
                heatmap_ax.set_aspect('equal')

                # Compute bin indices for all positions at once
                x_indices = np.digitize(positions_array[:, 0], x_bins) - 1
                y_indices = np.digitize(positions_array[:, 1], y_bins) - 1

                valid_mask = (x_indices >= 0) & (x_indices < num_bins) & (y_indices >= 0) & (y_indices < num_bins)
                x_indices = x_indices[valid_mask]
                y_indices = y_indices[valid_mask]
    
                # Use the individual activity values rather than their mean over time
                activity_values = network_activity[valid_mask, a, neuron]

                # Create heatmap and update using np.maximum.at to accumulate max values per bin
                heatmap = np.zeros((num_bins, num_bins))
                np.maximum.at(heatmap, (x_indices, y_indices), activity_values)

                im = heatmap_ax.imshow(heatmap.T, origin='lower', extent=[x_min, x_max, y_min, y_max], vmax=1, vmin=0)
                heatmap_ax.set(title=f'Gain = {round(alpha, 2)}', xticks=[], yticks=[])

                if a == 0:
                    heatmap_ax.set_xlabel('X axis arena')
                    heatmap_ax.set_ylabel('Y axis arena')


            cbar_ax = fig.add_subplot(gs[0, -1])
            colorbar = fig.colorbar(im, cax=cbar_ax)
            colorbar.set_label('Normalized firing rate', labelpad=15, rotation=270)
            colorbar.set_ticks([0, 0.5, 1])
            colorbar.set_ticklabels([f'{0:.2f}', f'{0.5:.2f}', f'{1:.2f}'])

            fig.tight_layout(h_pad=3.0)
            plt.savefig(f'Results\\ID{ID}\\result_activity_neuron{neuron}.png', format='png')
            plt.close()

    def fit_linear_model(self, activity_array, pos, return_shuffled=False, alpha=1.0, cv_folds=10):
        '''
        Predicts the location of the agent based on the activity level of the network

        :param activity_array: np.array featuring the time history of network activity. shape (ntime, ngain, nneuron)
        :param pos: list of shape (ntime, ndim), position log of the agent
        :param return_shuffled: bool, default=False
        :param alpha: float, parameter used for the regression model
        :param cv_folds: int, amount of folds to divide the data into

        Returns:
            tuple: (X (np.array), y (np.array), y_pred (np.array), mse_mean (float), r2_mean (float)) OR
            tuple: (X (np.array), y (np.array), y_pred (np.array), mse_mean (float), mse_mean_shuffeled (float), r2_mean (float), r2_mean_shuffeled)
        '''
        np.random.seed(self.seed)

        X = np.array(activity_array).reshape(np.shape(activity_array)[0],-1)  # shape is (time, gains*N)
        y = np.array(pos)  # shape is (time, 2)

        # Initialize the Ridge regression model with specified alpha
        model = Ridge(alpha=alpha)
        # model = LinearRegression()

        # Perform K-Fold cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)

        # Cross-validation scores for MSE
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error') #, n_jobs=-1) <- this enables parallel processing
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

    def plot_prediction_path(self, y, y_pred, mse_mean, r2_mean, ID=0):
        '''
        Plots the predicted path and the actual path and saves the plot under the results folder in the relative directory

        :params y: np.array (ntime, ndim), actual path
        :params y_pred: np.array (ntime, ndim), predicted path
        :params mse_mean: float, the mean-square-error of the prediction
        :params r2_mean: float, the r2-error of the prediction
        '''
        size = int(len(y)/2)
        size = np.clip(size, 10, 10000)
        plt.figure(figsize=(8, 6))
        plt.plot(y[-size:, 0], y[-size:, 1], 'b.-', label="Actual Path", alpha=0.6) 
        plt.plot(y_pred[-size:, 0], y_pred[-size:, 1], 'r.-', label="Predicted Path", alpha=0.6)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.title(f'Actual vs Predicted Positions. MSE={mse_mean}, Rˆ2={r2_mean}')
        plt.savefig(f'Results\\ID{ID}\\result_prediction.png', format='png')
        plt.close()


        