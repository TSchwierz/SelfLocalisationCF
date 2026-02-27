"""
Drone Simulation Script
-----------------------
This script simulates a drone's movement within a defined arena. The drone's motion is determined by:
- A smooth random angular walk that produces small, continuous turns.
- An optional drift component that nudges the drone toward the arena center.
- A boundary correction mechanism that gently clips the movement to remain within the arena.

The simulation logs the drone's position and grid network activity, then visualizes the movement and 
predicts the drone's path using a linear model.
"""

import os
import numpy as np
from sklearn.linear_model import Ridge
from controller import Robot
from controller import Supervisor
from DroneController import DroneController
from optimisedGridNetwork import MixedModularCoder
from datetime import datetime
import pickle
from PredictionModel import OptimisedRLS
from time import perf_counter
from sklearn.metrics import mean_squared_error, r2_score
from joblib import Parallel, delayed
from multiprocessing import shared_memory
import cupy as cp

# ---------------- Simulation Parameters ----------------
FLYING_ATTITUDE = 0                 # Base altitude (z-value) for flying
INITIAL_PAUSE = 12                  # Time (in seconds) for the drone to lift off and stabilize
COMMAND_INTERVAL = 1                # Interval (in seconds) between new movement commands
COMMAND_TOLERANCE = 0.032           # Tolerance (in seconds) for command timing
MOVEMENT_MAGNITUDE = 0.5           # Magnitude of the movement vector in the plane
ARENA_BOUNDARIES = np.array([[-1, 1],  # x boundaries
                             [-1, 1],  # y boundaries
                             [-1, 1]]) # z boundaries


# ---------------- Helper Functions ----------------
def visited_volume_percentages(trajectory, bounds, voxel_size=0.05, t=-1):
    """
    Compute the covered‐volume percentages for a 3D trajectory in a box.
    
    Parameters
    ----------
    trajectory : ndarray, shape (T, 3)
        Sequence of (x,y,z) points.
    bounds : tuple of floats
        (xmin, xmax, ymin, ymax, zmin, zmax).
    voxel_size : float
        Edge length of each cubic voxel.
    t : int
        Time‐index (0-based) up to which to report the partial coverage.
    
    Returns
    -------
    pct_up_to_t : float
        Percentage of box‐volume visited at least once in timesteps [0..t].
    pct_total : float
        Percentage of box‐volume visited at least once in the entire trajectory.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    # number of voxels along each axis
    nx = int(np.floor((xmax - xmin) / voxel_size)) + 1
    ny = int(np.floor((ymax - ymin) / voxel_size)) + 1
    nz = int(np.floor((zmax - zmin) / voxel_size)) + 1
    total_voxels = nx * ny * nz

    # map coords → integer voxel indices and clamp in [0, n-1]
    # shape (T,3) → (T,) linear indices
    scaled = (trajectory - np.array([xmin, ymin, zmin])) / voxel_size
    ijk = np.floor(scaled).astype(int)
    # clamp out-of-bounds points
    ijk[:,0] = np.clip(ijk[:,0], 0, nx-1)
    ijk[:,1] = np.clip(ijk[:,1], 0, ny-1)
    ijk[:,2] = np.clip(ijk[:,2], 0, nz-1)
    lin_idx = ijk[:,0] * (ny*nz) + ijk[:,1] * nz + ijk[:,2]

    # unique counts via np.unique
    unique_all = np.unique(lin_idx)
    unique_t   = np.unique(lin_idx[:t+1])

    pct_up_to_t = unique_t.size   / total_voxels * 100.0
    pct_total   = unique_all.size / total_voxels * 100.0

    return pct_up_to_t, pct_total

def generate_gain_lists(Lsize, Lspacing, start=0.1):
    '''
    Generates a list of gains according to the desired sizes and spacings

    :param Lsize: list of ints containing the amount of gains 
    :param Lspacing: list of floats containing the constant increase between the gains

    Return:
    - an inhomogenous list of shape (len(Lsize)*len(Lspacing), Lsize) containing lists of gains
    '''
    gain_list = []

    for n in Lsize:
        for s in Lspacing:
            gains = [round(start + i * s, 1) for i in range(n)]
            gain_list.append(gains)
    return gain_list

def save_object(obj, fname='data.pickle'):
    '''
    saves a python object to a file using the built-in pickle library
    
    :param obj: The object to be saved
    :param fname: Name of the file in which to save the obj. Default is data.pickle
    '''
    try:
        with open(fname, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

def adjust_for_boundaries(boundaries, position, movement_vector):
    """
    Adjust the movement vector if the new position would exceed the arena boundaries.

    Rather than reversing the entire movement, this function clips the movement such that the drone
    ends exactly at the boundary if it would otherwise overshoot.

    :param boundaries: Array of shape (3, 2) with lower and upper bounds for x, y, and z.
    :param position: Current 3D position of the drone.
    :param movement_vector: Proposed 3D movement vector [dx, dy, dz].
    :return: Tuple (adjusted 2D movement vector, adjusted z-component).
    """
    new_position = position + movement_vector
    adjusted_vector = movement_vector.copy()
    lower_bounds, upper_bounds = boundaries[:, 0], boundaries[:, 1]
    
    for i in range(3):
        if new_position[i] < lower_bounds[i]:
            # Move only as much as needed to hit the lower bound
            adjusted_vector[i] = lower_bounds[i] - position[i]
        elif new_position[i] > upper_bounds[i]:
            # Move only as much as needed to hit the upper bound
            adjusted_vector[i] = upper_bounds[i] - position[i]
    return adjusted_vector

def update_direction(current_direction, magnitude, dt, angular_std=0.25):
    """
    Update the drone's movement direction in 3D using a small random rotation.
    
    This function performs a random walk on the unit sphere. At each update, the current
    direction is perturbed by a small random rotation about a random axis perpendicular to
    the current direction. The rotation angle is drawn from a normal distribution (scaled by √dt
    for consistency with time resolution), resulting in smooth and continuous changes in direction.
    
    :param current_direction: Current 3D movement vector.
    :param magnitude: Desired magnitude of the new movement vector.
    :param dt: Time step (in seconds).
    :param angular_std: Standard deviation of the angular change (as a fraction of pi).
    :return: Updated 3D movement vector with the specified magnitude.
    """
    norm_curr = np.linalg.norm(current_direction)
    if norm_curr < 1e-6:
        # If current_direction is near zero, choose a random direction uniformly on the sphere.
        random_vec = np.random.randn(3)
        #random_vec = np.array([1.0, 1.0, 1.0]) # for more controlled trajectories
        unit_current = random_vec / np.linalg.norm(random_vec)
    else:
        unit_current = current_direction / norm_curr

    # Draw a small random rotation angle (scaled by √dt)
    d_angle = np.random.normal(0, angular_std * np.pi * np.sqrt(dt))
    
    # If the rotation angle is effectively zero, return the current direction (scaled).
    if np.abs(d_angle) < 1e-8:
        return unit_current * magnitude

    # Generate a random vector and project it to get a vector perpendicular to unit_current.
    random_vector = np.random.randn(3)
    perp = random_vector - np.dot(random_vector, unit_current) * unit_current
    perp_norm = np.linalg.norm(perp)
    if perp_norm < 1e-6:
        # Fallback: choose an arbitrary perpendicular vector.
        if np.abs(unit_current[0]) < 0.9:
            perp = np.cross(unit_current, np.array([1, 0, 0]))
        else:
            perp = np.cross(unit_current, np.array([0, 1, 0]))
        perp_norm = np.linalg.norm(perp)
    axis = perp / perp_norm

    # Use Rodrigues rotation formula to compute the rotated vector. simplifies to:
    #   v_rot = v*cos(theta) + (a x v)*sin(theta)
    new_direction = np.cos(d_angle) * unit_current + np.sin(d_angle) * np.cross(axis, unit_current)
    #print(f'{new_direction}, size={np.linalg.norm(new_direction)}')
    return new_direction * magnitude

def simulate_webots(gains, robot_, simulated_minutes=1, noise_scales=(0 , 0), angular_std=0.33, two_dim=False):
    # Initialize simulation components
    robot = robot_
    timestep_ms = int(robot.getBasicTimeStep())
    dt = timestep_ms / 1000.0  # Convert timestep to seconds   
    controller = DroneController(robot, FLYING_ATTITUDE)
    mmc = MixedModularCoder(gains=gains, two_dim=two_dim)
    neural_noise, velocity_noise = noise_scales 
    
    # Initialize state variables
    previous_direction = np.array([0, 0, 0])  # Initial xyz movement direction
    altitude = FLYING_ATTITUDE           # Constant base altitude
    target_altitude = altitude           # Target altitude (may be updated)
    elapsed_time = 0
    network_states = []
    activity_log = []
    position_log = []   
    velocity_log = []
    acceleration_log = []
    integrated_pos_log = []
    execution_time = []
    
    MAX_SIMULATION_TIME = (60 * simulated_minutes) + INITIAL_PAUSE # 1min in seconds * amount of hours
    UPDATE_INTERVAL = MAX_SIMULATION_TIME/10 #define amount of console updates by changing denominator
    
    print(f'Starting Webots Simulation, gains:{gains}')
    # Main loop: run until simulation termination signal or time limit reached
    while robot.step(timestep_ms) != -1 and elapsed_time < MAX_SIMULATION_TIME:
        elapsed_time += dt  # Update elapsed time in seconds

        # print progress
        if (elapsed_time%UPDATE_INTERVAL<= dt):
            print(f'{datetime.now().time()} - simulated time: {int(elapsed_time/60)} of {MAX_SIMULATION_TIME/60} minutes {elapsed_time/MAX_SIMULATION_TIME:.1%}')
        
        # Initial start-up phase (Drone needs to get to stable hover altitude)
        if (elapsed_time < INITIAL_PAUSE):
            position_real, velocity_gps = controller.update(np.array([0,0])) # desired initial position
            velocity, az_corrected = controller.get_velocity()

        # After Initial start-up
        else:
            if(controller.initial_pid):
                controller.change_gains_pid(kp=0.5, kd=1.0, ki=0.0)
                mmc.set_integrator(controller.get_location(two_dim=two_dim)) # get real position once to set integrator
                controller.reset_velocity() # sets the velocity integrated by imu sensors to gps derived velocity
                # Default movement: no change unless a new command is issued at the interval #2d for proper function
                movement_direction = np.array([0, 0])           
        
            # Issue a new movement command at defined intervals (after the initial pause)
            if (elapsed_time % COMMAND_INTERVAL) <= COMMAND_TOLERANCE:
                movement_direction = update_direction(previous_direction, MOVEMENT_MAGNITUDE, dt, angular_std=angular_std) # Update direction using a small-angle random walk     
                movement_direction = adjust_for_boundaries(ARENA_BOUNDARIES, position_real, movement_direction) # Adjust the movement to respect arena boundaries
                previous_direction = movement_direction  # Use the latest command as the basis for the next direction update
                #print(movement_direction)
        
            # Controller + Network Update         
            position_real, velocity_gps = controller.update( (movement_direction[:2] if two_dim else movement_direction) ) 
            velocity, az_corrected = controller.get_velocity(two_dim)
            vel_ = (velocity + np.random.normal(0, velocity_noise, size=velocity.shape)) * dt # add velocity noise for prediction model input
            activity, pos_internal = mmc.update(vel_, noise=neural_noise)

            # saving values
            integrated_pos_log.append(pos_internal.copy())
            activity_log.append(activity)
            position_log.append(position_real)
            velocity_log.append(velocity)

        # fail save, in case drone flys too far out of arena radius
        if (np.linalg.norm(position_real) > 3*(2.5**2)):
            break
    
    # ---------------- End of Simulation ----------------
    print(f'Simulation finished at {elapsed_time/60:.0f} minutes')
    train_perc, total_perc = visited_volume_percentages(integrated_pos_log, ARENA_BOUNDARIES.flatten())
    # bUILD dictionary of the results
    data = {
            'sim time' : simulated_minutes,
            'dt ms' : timestep_ms,
            'gains' : gains,
            'modular projections' : mmc.projected_dim,
            'module operators' : mmc.A,
            'activity' : activity_log,
            'velocity' : velocity_log,
            'position' : position_log,
            'position internal' : integrated_pos_log,
            'volume visited' : total_perc
        }
    return data

def create_k_folds(X, Y, k):
    """
    Create K folds of two multi-dimensional arrays X and Y, preserving temporal connection.

    Parameters:
    - X: numpy.ndarray, shape (t, ...)
    - Y: numpy.ndarray, shape (t, ...)
    - k: int, number of folds

    Returns:
    - folds: list of tuples, each tuple contains (X_fold, Y_fold) of shape (fold_size, ...)
    """
    t = X.shape[0]
    if t != Y.shape[0]:
        raise ValueError("X and Y must have the same first dimension (t).")

    fold_size = t // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else t  # Handle last fold

        X_fold = X[start:end]
        Y_fold = Y[start:end]

        folds.append((X_fold, Y_fold))

    return folds

def create_shared_memory_array(array, name_prefix):
    """
    Create a shared memory block for a numpy array.
    
    Returns:
    - shm: SharedMemory object
    - shape: Original array shape
    - dtype: Original array dtype
    """
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes, name=f"{name_prefix}_{os.getpid()}")
    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[:] = array[:]
    return shm, array.shape, array.dtype


def get_shared_memory_array(shm_name, shape, dtype):
    """Reconstruct numpy array from shared memory."""
    shm = shared_memory.SharedMemory(name=shm_name)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return array, shm


def process_overfit_window_optimized(i, overfit_test_folds, rr_weights_shm_name, rr_weights_shape,
                                        rr_intercept_shm_name, rr_intercept_shape,
                                      rls_weights_shm_name, rls_weights_shape, weights_dtype,
                                      train_pos_overfit, k_folds):
    """
    Optimized version with vectorized predictions and shared memory.
    """
    test_act_window, test_pos_window = overfit_test_folds[i]
    
    # Reconstruct models from shared memory
    rr_weights, rr_shm = get_shared_memory_array(rr_weights_shm_name, rr_weights_shape, weights_dtype)
    rr_intercept, rr_intercept_shm = get_shared_memory_array(rr_intercept_shm_name, rr_intercept_shape, weights_dtype)
    rls_weights, rls_shm = get_shared_memory_array(rls_weights_shm_name, rls_weights_shape, weights_dtype)
    
    # === Ridge Regression - Vectorized Prediction ===
    # Manual prediction: X @ weights (sklearn Ridge stores coef_ as (n_outputs, n_features))
    y_pred_rr_i = test_act_window @ rr_weights.T + rr_intercept  # Shape: (num_timesteps, 3)
    mse_rr_i = mean_squared_error(test_pos_window, y_pred_rr_i)
    r2_rr_i = r2_score(test_pos_window, y_pred_rr_i)
    
    # === RLS - Vectorized Prediction ===
    pred_pos = test_act_window @ rls_weights  # Shape: (num_timesteps, 3)
    
    # Vectorized MSE calculation
    squared_errors = (test_pos_window - pred_pos)**2
    mse_per_step = np.mean(squared_errors, axis=1)
    avg_mse = np.mean(mse_per_step)
    
    # Vectorized R² calculation
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((test_pos_window - np.mean(train_pos_overfit, axis=0))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Cleanup shared memory references
    rr_shm.close()
    rr_intercept_shm.close()
    rls_shm.close()
    
    return (y_pred_rr_i, mse_rr_i, r2_rr_i, pred_pos, avg_mse, r2)


def process_fold_heldout_optimized(i, folds, activity_shape, alpha = 1.0, lambda_= 0.999):
    """
    Optimized held-out fold processing with vectorized RLS predictions.
    """
    test_act, test_pos = folds[i]
    train_act = np.vstack([folds[j][0] for j in range(len(folds)) if j != i])
    train_pos = np.vstack([folds[j][1] for j in range(len(folds)) if j != i])
    
    # === Train and test Ridge Regression ===
    rr_model = Ridge(alpha=alpha)
    rr_model.fit(train_act, train_pos)
    y_pred_rr_i = rr_model.predict(test_act)
    mse_rr_i = mean_squared_error(test_pos, y_pred_rr_i)
    r2_rr_i = r2_score(test_pos, y_pred_rr_i)
    
    # === Train RLS model (sequential - unavoidable) ===
    rls_model = OptimisedRLS(activity_shape, num_outputs=3, lambda_=lambda_)
    for t, (pos, act) in enumerate(zip(train_pos, train_act)):
        rls_model.update(act, pos)
    
    # === Test RLS model - VECTORIZED ===
    pred_pos = test_act @ rls_model.A  # Vectorized prediction
    
    # Vectorized MSE and R² calculation
    squared_errors = (test_pos - pred_pos)**2
    mse_per_step = np.mean(squared_errors, axis=1)
    avg_mse = np.mean(mse_per_step)
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((test_pos - np.mean(train_pos, axis=0))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return (y_pred_rr_i, mse_rr_i, r2_rr_i, pred_pos, avg_mse, r2)


def run_decoders_optimized(data, n_folds=5, n_jobs=-1, alpha_=1, lambda_=0.999):
    """
    Fully optimized decoder evaluation with:
    - Vectorized predictions for both RR and RLS
    - Shared memory for model weights
    - joblib parallelization
    
    Parameters:
    - n_jobs: Number of parallel jobs (-1 uses all cores, -2 uses all but one)
    """
    print('Running Optimized Decoders')
    position = np.array(data['position'])
    activity = np.array(data['activity']).reshape(len(position), -1)
    k_folds = n_folds
    
    # Create k-fold splits
    folds = create_k_folds(activity, position, k_folds)
    
    # ====================================================================
    # OVERFITTING (IN-SAMPLE) PROTOCOL - OPTIMIZED
    # ====================================================================
    print('\n' + '='*70)
    print('OVERFITTING (IN-SAMPLE) PROTOCOL')
    print('='*70)
    
    # Use 80% of data for training
    train_folds_overfit = int(0.8 * k_folds)
    train_act_overfit = np.vstack([folds[j][0] for j in range(train_folds_overfit)])
    train_pos_overfit = np.vstack([folds[j][1] for j in range(train_folds_overfit)])
    
    print(f'Training on {train_folds_overfit}/{k_folds} folds ({len(train_act_overfit)} timesteps)')
    
    # === Train Ridge Regression ===
    print('Training Ridge Regression model...')
    t_start = perf_counter()
    rr_model_overfit = Ridge(alpha=alpha_)
    rr_model_overfit.fit(train_act_overfit, train_pos_overfit)
    print(f'  ✓ Completed in {perf_counter() - t_start:.2f}s')
    
    # === Train RLS model ===
    print('Training RLS model...')
    t_start = perf_counter()
    rls_model_overfit = OptimisedRLS(activity.shape[1], num_outputs=3, lambda_=lambda_)
    for t, (pos, act) in enumerate(zip(train_pos_overfit, train_act_overfit)):
        rls_model_overfit.update(act, pos)
        if (t + 1) % 5000 == 0:
            print(f'  Progress: {t+1}/{len(train_pos_overfit)} timesteps', end='\r')
    print(f'\n  ✓ Completed in {perf_counter() - t_start:.2f}s')
    
    # === Create shared memory for model weights ===
    print('Creating shared memory for model weights...')
    rr_weights_shm, rr_shape, rr_dtype = create_shared_memory_array(
        rr_model_overfit.coef_, "rr_weights"
    )
    rr_intercept_shm, rr_intercept_shape, rr_intercept_dtype = create_shared_memory_array(
    rr_model_overfit.intercept_, "rr_intercept"
    )
    rls_weights_shm, rls_shape, rls_dtype = create_shared_memory_array(
        rls_model_overfit.A, "rls_weights"
    )
    
    # Create test folds from training data
    overfit_test_folds = create_k_folds(train_act_overfit, train_pos_overfit, k_folds)
    
    # === Parallel testing on windows ===
    print(f'\nTesting on {k_folds} windows (parallelized with joblib, n_jobs={n_jobs})...')
    t_start = perf_counter()
    
    overfit_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_overfit_window_optimized)(
            i, overfit_test_folds, 
            rr_weights_shm.name, rr_shape,
            rr_intercept_shm.name, rr_intercept_shape,
            rls_weights_shm.name, rls_shape,
            rr_dtype, train_pos_overfit, k_folds
        ) for i in range(k_folds)
    )
    
    print(f'  ✓ Completed in {perf_counter() - t_start:.2f}s')
    
    # Cleanup shared memory
    rr_weights_shm.close()
    rr_weights_shm.unlink()
    rr_intercept_shm.close()
    rr_intercept_shm.unlink()
    rls_weights_shm.close()
    rls_weights_shm.unlink()
    
    # Unpack results
    y_pred_rr_overfit = []
    mse_rr_overfit = []
    r2_rr_overfit = []
    y_pred_rls_overfit = []
    mse_rls_overfit = []
    r2_rls_overfit = []
    
    for y_rr, m_rr, r_rr, y_rls, m_rls, r_rls in overfit_results:
        y_pred_rr_overfit.append(y_rr)
        mse_rr_overfit.append(m_rr)
        r2_rr_overfit.append(r_rr)
        y_pred_rls_overfit.append(y_rls)
        mse_rls_overfit.append(m_rls)
        r2_rls_overfit.append(r_rls)
    
    y_pred_rr_overfit = np.concatenate(y_pred_rr_overfit)
    y_pred_rls_overfit = np.concatenate(y_pred_rls_overfit)
    
    print(f'\nOverfit Results:')
    print(f'  Ridge Regression - MSE: {np.mean(mse_rr_overfit):.6f}, R²: {np.mean(r2_rr_overfit):.4f}')
    print(f'  RLS              - MSE: {np.mean(mse_rls_overfit):.6f}, R²: {np.mean(r2_rls_overfit):.4f}')
    
    # ====================================================================
    # CROSS-VALIDATION (HELD-OUT) PROTOCOL - OPTIMIZED
    # ====================================================================
    print('\n' + '='*70)
    print('CROSS-VALIDATION (HELD-OUT) PROTOCOL')
    print('='*70)
    print(f'Running {k_folds}-fold cross-validation (parallelized with joblib, n_jobs={n_jobs})...')
    
    t_start = perf_counter()
    heldout_results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(process_fold_heldout_optimized)(
            i, folds, activity.shape[1], alpha_, lambda_
        ) for i in range(k_folds)
    )
    print(f'  ✓ Completed in {perf_counter() - t_start:.2f}s')
    
    # Unpack results
    y_pred_rr = []
    mse_rr = []
    r2_rr = []
    y_pred_rls = []
    mse_rls = []
    r2_rls = []
    
    for y_rr, m_rr, r_rr, y_rls, m_rls, r_rls in heldout_results:
        y_pred_rr.append(y_rr)
        mse_rr.append(m_rr)
        r2_rr.append(r_rr)
        y_pred_rls.append(y_rls)
        mse_rls.append(m_rls)
        r2_rls.append(r_rls)
    
    y_pred_rr = np.concatenate(y_pred_rr)
    y_pred_rls = np.concatenate(y_pred_rls)
    
    print(f'\nHeld-Out Results:')
    print(f'  Ridge Regression - MSE: {np.mean(mse_rr):.6f}, R²: {np.mean(r2_rr):.4f}')
    print(f'  RLS              - MSE: {np.mean(mse_rls):.6f}, R²: {np.mean(r2_rls):.4f}')
    
    # ====================================================================
    # COMBINE RESULTS
    # ====================================================================
    print('\n' + '='*70)
    print('SUMMARY')
    print('='*70)
    print(f'Performance Gap (Overfit - Held-Out):')
    print(f'  Ridge Regression - ΔR²: {np.mean(r2_rr_overfit) - np.mean(r2_rr):.4f}')
    print(f'  RLS              - ΔR²: {np.mean(r2_rls_overfit) - np.mean(r2_rls):.4f}')
    
    results = {
        'y_pred_rr_overfit': y_pred_rr_overfit,
        'mse_rr_overfit': mse_rr_overfit,
        'r2_rr_overfit': r2_rr_overfit,
        
        'y_pred_rls_overfit': y_pred_rls_overfit,
        'mse_rls_overfit': mse_rls_overfit,
        'r2_rls_overfit': r2_rls_overfit,
        
        'y_pred_rr_heldout': y_pred_rr,
        'mse_rr_heldout': mse_rr,
        'r2_rr_heldout': r2_rr,
        
        'y_pred_rls_heldout': y_pred_rls,
        'mse_rls_heldout': mse_rls,
        'r2_rls_heldout': r2_rls,
    }
    
    return results

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    robot = Robot()
    supervisor = Supervisor()
    robot_node = supervisor.getFromDef("Crazyflie")
    trans_field = robot_node.getField("translation")
    INITIAL = [0, 0, 0]

    trial_per_setting_ = 1

    # Generate Gain Lists for Benchmark
    nr = [3, 4, 5]
    spacing = [0.1, 0.2, 0.3, 0.4]
    gain_list = generate_gain_lists(nr, spacing, start=0.2)
    #gain_list = [[0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]

    # Name for the results folder (used for id)
    name = 'TestLowLambda'

    ###################### Test Setting
    setting_name = 'Testing Model Parameter'
    model_confs = [[1.0, 0.99], [1.0, 0.999], [1.0, 0.9999]] #alpha, lambda
    setting = model_confs #list of parameter to test
    gains = [[0.2, 0.3, 0.4, 0.5]] * len(setting) #Example gain setting
    times = 1.0 * np.ones(len(setting)) # in minutes
    noise_act = 0.0 * np.ones(len(setting)) # in fraction of max firing rate
    noise_vel = 0.0 * np.ones(len(setting)) # in fraction of max firing rate
    noise_ = np.stack([noise_act, noise_vel], axis=1)

    ##################### Velocity Noise Setting
    #setting_name = 'velocity noise variation'
    #noise_vel = [0, 0.005, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    #setting = noise #list of parameter to test
    #gains = [[0.2, 0.3, 0.4, 0.5]] * len(setting) #Example gain setting
    #times = 10.0 * np.ones(len(setting)) # in minutes
    #noise = 0.0 * np.ones(len(setting)) # in fraction of max firing rate
    #model_confs = [[1.0, 0.999], [0.0, 0.999], [1.0, 1.0], [0.0, 1.0]]
    #noise_act = 0.0 * np.ones(len(setting)) # in fraction of max firing rate
    #noise_ = np.stack([noise_act, noise_vel], axis=1)
    
    #################### Benchmark Gain Settings
    #setting_name = 'gain variation'
    #print(gain_list[4:8])
    #gains = gain_list[4:8]
    #setting = gains
    #times = 10.0 * np.ones(len(setting)) # in minutes
    #noise_act = 0 * np.ones(len(setting)) # in fraction of max firing rate
    #noise_vel = 0.0 * np.ones(len(setting)) # in fraction of max firing rate
    #noise_ = np.stack([noise_act, noise_vel], axis=1)

    ##################### Time Variation Settings
    #setting_name = 'time variation'
    #times = [5, 10, 15, 20, 25, 30] # in minutes
    #setting = times
    #gains = [[0.2, 0.3, 0.4, 0.5]]*len(setting)
    #noise_act = 0.05 * np.ones(len(setting)) # in fraction of max firing rate
    #noise_vel = 0.0 * np.ones(len(setting)) # in fraction of max firing rate
    #noise_ = np.stack([noise_act, noise_vel], axis=1)

    ###################### Activity Noise Variation Settings
    #setting_name = 'noise variation'
    #noise_act = [0, 0.005, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    #setting = noise
    #times = 10.0 * np.ones(len(setting)) # in minutes
    #gains = [[0.2, 0.3, 0.4, 0.5]]*len(setting)
    #noise_vel = 0.0 * np.ones(len(setting)) # in fraction of max firing rate
    #noise_ = np.stack([noise_act, noise_vel], axis=1)

     ###################### Both Noise Variation Settings
    #setting_name = 'both noise variation'
    #noise_act = [0, 0.1]#[0, 0.005, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    #noise_vel = [0, 0.1]#[0, 0.005, 0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    #a_, v_ = np.meshgrid(noise_act, noise_vel)
    #noise_ = np.stack([a_.ravel(), v_.ravel()], axis=1)
    #setting = noise_
    #times = 10.0 * np.ones(len(setting)) # in minutes
    #gains = [[0.2, 0.3, 0.4, 0.5]]*len(setting)

    for i, var in enumerate(setting):
        dim2 = False
        trial_per_setting = trial_per_setting_ # set to var if you want to vary trials per setting

        print(f'Running {trial_per_setting} trials for setting {i+1}/{len(setting)}')
        id_ = f'{name} Setting {i}of{len(setting)}'
        data_all = []

        results_dir = f"Results\\ID {id_}"
        os.makedirs(results_dir, exist_ok=True)

        for trial in range(trial_per_setting):
            t1 = perf_counter()
            print(f'\n--- Trial {trial+1} of {trial_per_setting} for setting {i+1} of {len(setting)} ---')
            # Generate webots data
            trans_field.setSFVec3f(INITIAL)
            robot_node.resetPhysics()
            print(f'Simulating with noise [activity={noise_[i][0]}, velocity={noise_[i][1]}]')
            data = simulate_webots(gains=gains[i], robot_=robot, simulated_minutes=times[i],
                                  noise_scales=(noise_[i][0], noise_[i][0]), angular_std=0.33, two_dim=dim2)
            
            # Run decoders
            alpha_, lambda_ = var if setting_name == 'Testing Model Parameter' else (1.0, 0.999) # default values if not provided
            print(f'Decoding with [alpha={alpha_}, lambda={lambda_}]')
            decoder_results = run_decoders_optimized(data, alpha_ = alpha_, lambda_ = lambda_)
            data.update(decoder_results) # add decoder results to data of trial
            decoder_results.update({'volume': data['volume visited']}) # add visited volume to the values to avg over trials

            time_taken = perf_counter() - t1
            decoder_results['time taken'] = time_taken

            # Save and append data
            data_all.append(decoder_results)
            save_object(data, f'{results_dir}\\data_trial{trial}.pickle')

        summary = {}
        for key in data_all[0].keys():
            # stack along new axis and average
            vals = [m[key] for m in data_all]
            flat_vals = np.array(vals).flatten()

            print(f"Key: {key}")
            print(f"Type: {type(vals)}")
            print(f"Shape: {np.array(vals).shape}")
            #print(f"Sample values: {vals[:3]}")  # First 3 values
            print(f"Min/Max: {np.min(vals)}, {np.max(vals)}")
            print("---")
            summary[f'mse {key}'] = np.mean(flat_vals)
            summary[f'median {key}'] = np.median(flat_vals)
            summary[f'std {key}'] = np.std(flat_vals)
            summary[f'iqr {key}'] = np.percentile(flat_vals, [25, 75])

        summary['setting var'] = setting
        summary['setting mode'] = setting_name
        summary['noise'] = noise_[i]
        summary['model parameters'] = (alpha_, lambda_)
        save_object(summary, f'{results_dir}\\summary.pickle')

        print(f"\n→ All trials done. Summary saved to {results_dir}\\Summary.pickle")
    print("Finished all settings. Completed execution.")