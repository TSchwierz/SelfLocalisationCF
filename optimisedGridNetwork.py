import numpy as np
from numba import njit, prange, cuda
import math

class MixedModularCoder:
    def __init__(self, M=3, N=2, gains=[0.1, 0.2, 0.3, 0.4, 0.5], two_dim=False):
        M = 1 if two_dim else 3
        
        # Fixed projection assignment matrix
        if two_dim:
            self.A = np.array([[[1,0], [0,1]]])
        else:
            self.A = np.array([
                [[1, 0, 0], [0,1,0]],
                [[0,1,0], [0,0,1]],
                [[1,0,0], [0,0,1]]
            ])
        # Boolean mask for each m stating the dimensions it projects to
        self.projected_dim = np.any(self.A != 0, axis=1)

        nx = 10
        ny = 9
        self.N = N
        self.M = M
        self.nrGains = len(gains)
        self.mod_size = nx*ny*self.nrGains
        self.ac_size = self.mod_size * M
        self.Module = []
        
        # Pre-compute the network once
        for m in range(M):
            self.Module.append(GridNetwork(nx, ny, gains=gains))
        
        #print('Initialised Mixed Modular Coder')

    def set_integrator(self, pos):
        self.pos_integrator = pos.copy()

    def update(self, velocity):
        # Pre-allocate activity array
        activity = np.zeros((self.M, self.nrGains, self.mod_size//self.nrGains))
        
        if np.ndim(velocity) == 2:
            vel2D = velocity       
        else:
            vel2D = self.project_velocity(velocity)
        
        # Update each module
        for i, m in enumerate(self.Module):
            activity[i] = m.update_network(vel2D[i])

        self.pos_integrator += velocity
        return activity, self.pos_integrator

    def project_velocity(self, vel3D):
        result = np.zeros((self.M, 2))
        for m in range(self.M):
            for n in range(self.N):
                for x in range(vel3D.shape[0]):
                    result[m, n] += self.A[m, n, x] * vel3D[x]
        return result
        
    def project_positions(self, pos3D):
        return np.dot(self.A, pos3D.T).transpose(0, 2, 1)


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
        self.N_x = N_x
        self.N_y = N_y
        self.N = N_x * N_y  # N for one layer, not the whole network

        # Parameters of the (dynamic) weight matrix
        self.I = 0.3                    # intensity parameter
        self.T = 0.05                   # shift parameter
        self.sigma = 0.24               # size of the gaussian
        self.tau = 0.8                  # normalization parameter
        self.beta = beta                # bias parameter (rotation of the grid)

        # Set gains
        if gains is None:
            self.gains = np.arange(1, 3.1, 0.5)
        else:
            self.gains = np.array(gains, dtype=np.float64)
            
        # Pre-compute rotation matrix
        self.R = np.array([
            [np.cos(self.beta), -np.sin(self.beta)],
            [np.sin(self.beta), np.cos(self.beta)]
        ])
        
        # Pre-compute distance matrix
        self.distance_matrix = self.initialize_distance_matrix()
        
        # Initialize network activity
        self.seed = Seed
        self.update_network_shape()
        
        # Pre-compute squared sigma for efficiency
        self.sigma_squared = self.sigma ** 2
        
        # Pre-compute W matrices for each gain and possible velocity
        self.precomputed_W = None
        
    def update_network_shape(self):
        """Update the network activity shape based on gains"""
        rng = np.random.default_rng(seed=self.seed)
        # Initialize with small positive values
        self.network_activity = rng.uniform(0.01, 0.1, (len(self.gains), self.N))
        
        # Normalize activities
        sum_activity = np.sum(self.network_activity, axis=1, keepdims=True)
        self.network_activity /= sum_activity
        
    def initialize_distance_matrix(self):
        """Vectorized version to initialize the distance matrix between all neurons."""
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
    
        # Define the possible translations for the twisted toroidal structure
        s_j = np.array([
            [0, 0],
            [-0.5,  np.sqrt(3) / 2],
            [-0.5, -np.sqrt(3) / 2],
            [0.5,  np.sqrt(3) / 2],
            [0.5, -np.sqrt(3) / 2],
            [-1, 0],
            [1, 0]
        ])
    
        # Compute all pairwise differences
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    
        # Add translations to differences
        candidates = diff[:, :, np.newaxis, :] + s_j[np.newaxis, np.newaxis, :, :]
    
        # Compute Euclidean norms
        candidate_norms = np.linalg.norm(candidates, axis=-1)
    
        # Find index of minimum norm translation
        best_index = np.argmin(candidate_norms, axis=-1)
    
        # Select optimal translation for each pair
        best_s = s_j[best_index]
    
        # Final distance matrix
        distance_matrix = diff + best_s
        
        #print('Distance matrix initialized')
        return distance_matrix

    def update_network(self, velocity):
        """
        Simulate grid-cell activity in real time
        
        Args:
            velocity: np.array consisting of 2d velocity
            
        Returns:
            np.array: (ngains, nneurons) reflecting the updated activity of the network
        """        
        # Convert velocity to standard numpy array if it's not already
        velocity = np.asarray(velocity, dtype=np.float64)
        
        # Compute rotated velocity once
        rotated_velocity = np.dot(self.R, velocity)
        
        # Create a copy of network_activity to avoid modifying the original during JIT execution
        # This helps prevent potential race conditions in parallel execution
        network_activity_copy = self.network_activity.copy()
        
        # Update all layers at once using JIT-compiled function
        result = _update_network_jit_fixed(
            network_activity_copy,
            self.distance_matrix,
            self.gains,
            rotated_velocity,
            self.I,
            self.T,
            self.sigma_squared,
            self.tau
        )
        
        # Update the network activity
        self.network_activity = result
        
        return result


@njit(parallel=True)
def _update_network_jit_fixed(network_activity, distance_matrix, gains, rotated_velocity, I, T, sigma_squared, tau):
    """JIT-compiled function with proper range normalization"""
    n_gains = len(gains)
    n_neurons = network_activity.shape[1]
    result = np.zeros_like(network_activity)
    
    # Process each gain in parallel
    for a in prange(n_gains):
        alpha = gains[a]
        
        # Compute weight matrix
        W = np.zeros((n_neurons, n_neurons))
        for i in range(n_neurons):
            for j in range(n_neurons):
                dx = distance_matrix[i, j, 0] + alpha * rotated_velocity[0]
                dy = distance_matrix[i, j, 1] + alpha * rotated_velocity[1]
                dist_squared = dx*dx + dy*dy
                W[i, j] = I * math.exp(-dist_squared / sigma_squared) - T
        
        # Apply weights to current activity
        b_activity = np.zeros(n_neurons)
        for i in range(n_neurons):
            sum_weighted = 0.0
            for j in range(n_neurons):
                sum_weighted += network_activity[a, j] * W[i, j]
            b_activity[i] = sum_weighted
        
        # Apply tau normalization
        sum_activity = 0.0
        for i in range(n_neurons):
            sum_activity += network_activity[a, i]
        
        epsilon = 1e-10
        temp_activity = np.zeros(n_neurons)
        
        for i in range(n_neurons):
            net_activity = (1 - tau) * b_activity[i] + tau * (b_activity[i] / (sum_activity + epsilon))
            if net_activity < 0:
                net_activity = 0
            temp_activity[i] = net_activity
        
        # RANGE NORMALIZATION: Scale to use full 0-1 range
        min_val = temp_activity[0]
        max_val = temp_activity[0]
        
        # Find min and max
        for i in range(1, n_neurons):
            if temp_activity[i] < min_val:
                min_val = temp_activity[i]
            if temp_activity[i] > max_val:
                max_val = temp_activity[i]
        
        # Scale to 0-1 range
        activity_range = max_val - min_val
        if activity_range > epsilon:
            for i in range(n_neurons):
                result[a, i] = (temp_activity[i] - min_val) / activity_range
                network_activity[a, i] = result[a, i]
        else:
            # If no range, keep current values
            for i in range(n_neurons):
                result[a, i] = temp_activity[i]
                network_activity[a, i] = temp_activity[i]
    
    return result