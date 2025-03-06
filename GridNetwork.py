from math import isnan, nan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.linear_model import Ridge, LinearRegression

class GridNetwork:
    
    def __init__(self, N_x, N_y, Seed=43, beta=0):
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
        '''Function to update the num of layers (gains) the network has'''
        # np.random.seed(self.seed)
        rng = np.random.default_rng(seed=self.seed) # create a random generator instance (to get locally random seeds)
        self.network_activity = rng.uniform(0, 1 / np.sqrt(self.N), (len(self.gains), self.N))

    def set_gains(self, gains):
        '''Function to change what gains are being used. It can vary in lenghts (layers of the network)'''
        self.gains = gains
        self.update_network_shape()  # Update network_activity shape when gains are changed

    def initialize_distance_matrix(self):
        ''' Function to initialize the distance matrix between all neurons. 
            Returns a matrix of N by N by 2. The last dimension corresponds to the 2d coordinates.
        '''
        
        N =  self.N
        
        i = np.arange(1, self.N_x+1)
        j = np.arange(1, self.N_y+1)

        x, y = np.meshgrid(i,j) # create x y coordinates (in 2d grids)
        x = np.ravel(x) # np.ravel and .flatten() is, in practice, the same 
        y = np.ravel(y)

        # compute c (position of the cells on the sheet, defined by coordinates c_x and c_y)
        cx = (x - 0.5)/self.N_x
        cy = (np.sqrt(3)/2) * (y - 0.5)/self.N_y
        c = np.array([[cx[i], cy[i]] for i in range(N)]) # list with 2d coordinates (len(c)=N)

        # Initialize ll possible rotations in the twisted toroidal structure
        s_j = [[0, 0], [-0.5, np.sqrt(3)/2], [-0.5,-np.sqrt(3)/2], [0.5, np.sqrt(3)/2],  [0.5, -np.sqrt(3)/2], [-1, 0], [1, 0]] 
        
        distance_matrix = np.zeros((N,N,2))

        # loop through all combinations of neurons
        for i in range(N):
            for j in range(N):
                min_norm = float('inf')  # Initialize with positive infinity to ensure the first norm is smaller
                for s in s_j: # Iterate over each s_j and compute the norm
                    # Compute the the min norm of the vector c[i] - c[j] + s
                    norm = np.linalg.norm(c[i] - c[j] + s)
                    # Update the minimum norm and its index if the current norm is smaller
                    if norm < min_norm:
                        min_norm = norm # update min norm
                        min_s = s
                        
                dist = c[i] - c[j] + min_s
                distance_matrix[i,j] = dist

        print('Distance matrix initialized') 
        return distance_matrix    

    def weight_function(self, I, T, sigma):
        ''' 
        This is to play around with different values of the parameters of the weight function to understand it better. 
        The weight function in the model includes the velocity vector, which here is considered to be 0.
        Params:
                I = intensity parameter, defines the overall synaptic strenght
                sigma = regulates the size of the gaussian
                T = shift parameter determining excitatory and inhibitory connections
        Outputs: Weight matrix
        Note: This functiion is not used to update weights, (there is no input in this one)
              This is just to be able to play with the parameters'''
            
        W = I * np.exp(-(np.linalg.norm(self.distance_matrix, axis=2)**2)/sigma**2) - T
        
        return W
   
    def update_network(self, velocity_vec, get_next_state=False):
        '''This function is to simulate grid-cell activity in real time'''
        
        if get_next_state: # init variable if get)next_state is True
            next_state_activity = np.zeros((len(self.gains), self.N))# this is for the RL part

        for a, alpha in enumerate(self.gains):  # Iterate over alpha values
            # Update weight matrix based on current velocity and alpha
            W = self.I * np.exp(- (np.linalg.norm(self.distance_matrix + alpha * np.dot(self.R, velocity_vec), axis=2)**2) / self.sigma**2) - self.T
            self.W = W

            # Calculate activity using transfer function
            b_activity = self.network_activity[a, :]
            b_activity = b_activity @ W
            
            # Normalize activity
            net_activity = ((1 - self.tau) * b_activity + self.tau * (b_activity / np.sum(self.network_activity[a, :])))
            net_activity[net_activity < 0] = 0 
            if (np.max(net_activity) != np.min(net_activity)): # normalise only if not all values identical
                net_activity = (net_activity - np.min(net_activity)) / (np.max(net_activity) - np.min(net_activity))
          
            if np.isnan(net_activity).any():
                print(f'velocity:{velocity_vec}')
                print(f'b_activity min/max/value:{np.min(b_activity)}/{np.max(b_activity)}/{b_activity}')
                print(f'network activity:{net_activity}')
                print(f'w/max/min={W}/{np.min(W)}/{np.max(W)}')
                print(f"Row sum for gain {alpha}: {np.sum(W, axis=1)}")
                print(f'np.max(net_a)={np.max(net_activity)}')
                raise ValueError('check this out, activity is exploading')
            
            # save activity for each gain
            if not get_next_state: # this is when network is run normally
                self.network_activity[a, :] = net_activity
            else: # this is when only evaluating next state ('kick') for RL. 
                next_state_activity[a,:] = net_activity  

        if get_next_state:
            # print('test with net activity:', np.mean(next_state_activity-self.network_activity) )
            return next_state_activity
           
    def reset_activity(self):
        ''' To reset the activity population in case there are jumps in xpace (the agent is randomly placed in a new location)'''
        rng = np.random.default_rng(seed=self.seed)
        self.network_activity = rng.uniform(0, 1 / np.sqrt(self.N), (len(self.gains), self.N))

    def plot_frame_figure(self, positions_array, num_bins, **kwargs):
        """
        Plots a heatmap of network activity at different gain levels and overlays the trajectory.

        Parameters:
        - positions_fig (array-like): List of (x, y) positions over time.
        - num_bins (int): Number of bins for the heatmap.
        - **kwargs: Optional parameters like 'arena_size' and 'network_activity'.

        """
        arena_size = kwargs.get('arena_size', 1)
        network_activity = kwargs.get('network_activity', self.network_activity)

        x_min, y_min = np.min(positions_array, axis=0)
        x_max, y_max = np.max(positions_array, axis=0)

        fig = plt.figure(figsize=(13, 8))
        gs = fig.add_gridspec(2, 6, height_ratios=[1, 2.5], width_ratios=[1, 1, 1, 1, 1, 0.07]) 

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
                heatmap[x_index, y_index] = max(heatmap[x_index, y_index], np.mean(activity[a, 42]))
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
        trajectory_ax = fig.add_subplot(gs[1, 1:4])  # Adding subplot for the bottom row # Spanning 3 columns in the middle
        trajectory_ax.plot(positions_array[:, 0], positions_array[:, 1], alpha=0.7, color='purple')
        trajectory_ax.set_title('Effective Arena with travelled path', fontsize=20)
        trajectory_ax.set_aspect('equal')


        fig.tight_layout(h_pad=3.0) # Adjust layout # change spacing between plots
        plt.savefig('Results\\result_activity_figure.png', format='png') # save in relative folder Results in Source/Repos/SelfLocalisationCF


    # Function to fit a linear model from grid network activity with cross-validation
    def fit_linear_model(self, activity_array, pos, return_shuffled=False, alpha=1.0, cv_folds=10):
        '''
        This function predicts the location of the object based on the activity level of the network
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
        mse_scores = -cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
        # Cross-validation scores for R2
        r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
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

    def plot_prediction_path(self, y, y_pred, mse_mean, r2_mean):
        # Plot actual vs predicted trajectories (first 1000 timesteps)
        size = int(len(y)/20)
        plt.figure(figsize=(8, 6))
        plt.plot(y[-size:, 0], y[-size:, 1], 'bo-', label="Actual Path", alpha=0.6) 
        plt.plot(y_pred[-size:, 0], y_pred[-size:, 1], 'ro-', label="Predicted Path", alpha=0.6)
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend()
        plt.title(f'Actual vs Predicted Positions. MSE={mse_mean}, Rˆ2={r2_mean}')
        plt.savefig('Results\\result_prediction.png', format='png')


        