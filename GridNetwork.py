from math import isnan, nan
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.linear_model import Ridge, LinearRegression
from tqdm import tqdm

class MixedModularCoder:
    def __init__(self, M=2, N=3, gains=[0.1, 0.2, 0.3, 0.4, 0.5]):
        # Random projection assignment
        #self.A = np.random.normal(size=(M, 2, N))
        #norm = np.linalg.norm(A, axis=(1, 2), keepdims=True)
        #self.A = A / norm

        self.A = np.array([
            [[1, 0, 0], [0,1,0]],
            [[0,1,0], [0,0,1]]
        ])
        # booplean mask for each m stating the dimension it projects
        self.projected_dim = np.any(self.A != 0, axis=1)

        nx = 10
        ny = 9
        self.M = M
        self.nrGains = len(gains)
        self.mod_size = nx*ny*len(gains)
        self.ac_size = self.mod_size * M
        self.Module = []
        for m in range(M):
            self.Module.append(GridNetwork(nx, ny, gains=gains))
        print('Initialised Mixed Modular Coder')

    def set_integrator(self, pos):
        self.pos_integrator = pos.copy()

    def update(self, velocity):
        activity = np.zeros((self.M, self.nrGains, self.mod_size//self.nrGains))
        vel2D = self.project_velocity(velocity)
        for i, m in enumerate(self.Module):
            activity[i] = m.update_network(vel2D[i])

        self.pos_integrator += velocity
        return activity, self.pos_integrator

    def project_velocity(self, vel3D):
        return np.einsum('mnx, x->mn', self.A, vel3D)
    def project_positions(self, pos3D):
        return np.einsum('mnx,tx->mtn', self.A, pos3D)

class GridNetwork:
    
    def __init__(self, N_x, N_y, gains=None, Seed=42, beta=0):
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
        if gains is None:
            self.gains = np.arange(1,3.1,0.5)           # Gain parameters (layers of the network)
        else:
            self.gains = gains
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
         # Initialize with small positive values to avoid zero-sum issues
        self.network_activity = rng.uniform(0.01, 0.1, (len(self.gains), self.N))
        # Normalize so it sums to 1 across neurons for each gain
        for g in range(len(self.gains)):
            sum_activity = np.sum(self.network_activity[g, :])
            if sum_activity > 0:  # Safety check
                self.network_activity[g, :] /= sum_activity

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
            
            # FIX 1: Handle potential division by zero in normalization
            sum_activity = np.sum(self.network_activity[a, :])
            if sum_activity <= 1e-10:  # If sum is too close to zero
                # Reinitialize this layer with small positive values
                self.network_activity[a, :] = np.random.uniform(0.01, 0.1, self.N)
                sum_activity = np.sum(self.network_activity[a, :])
                if sum_activity > 0:  # Safety check
                    self.network_activity[a, :] /= sum_activity
                continue  # Skip this iteration, try again next time
            
            # FIX 2: Safe normalization with epsilon
            epsilon = 1e-10
            net_activity = ((1 - self.tau) * b_activity + 
                           self.tau * (b_activity / (sum_activity + epsilon)))
            
            # FIX 3: Replace any NaN values that might still occur
            net_activity = np.nan_to_num(net_activity, nan=0.0, posinf=1.0, neginf=0.0)

            net_activity[net_activity < 0] = 0 # turn all negative values to zero

            # FIX 4: Safe normalization of output
            activity_range = np.max(net_activity) - np.min(net_activity)
            if activity_range > 1e-10:  # Only normalize if there's a meaningful range
                net_activity = (net_activity - np.min(net_activity)) / activity_range
            else:
                # If no range, initialize with small random values
                net_activity = np.random.uniform(0.01, 0.1, self.N)
                net_activity = net_activity / np.sum(net_activity)
            
            # FIX 5: Final check to ensure no NaNs
            if np.any(np.isnan(net_activity)):
                net_activity = np.random.uniform(0.01, 0.1, self.N)
                net_activity = net_activity / np.sum(net_activity)

            self.network_activity[a, :] = net_activity # save activity for each gain

        return self.network_activity           


        