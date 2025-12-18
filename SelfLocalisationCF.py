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
from controller import Robot
from controller import Supervisor
from DroneController import DroneController
from optimisedGridNetwork import MixedModularCoder
from datetime import datetime
import pickle
from PredictionModel import fit_linear_model
from PredictionModel import OptimisedRLS
import copy
from time import perf_counter

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
def visited_area_percentages(trajectory, bounds, voxel_size=0.05, t=-1):
    """
    Compute the covered-area percentages for a 2D trajectory in a rectangle.
    
    Parameters
    ----------
    trajectory : ndarray, shape (T, 2)
        Sequence of (x,y) points.
    bounds : tuple of floats
        (xmin, xmax, ymin, ymax).
    voxel_size : float
        Edge length of each square voxel (pixel).
    t : int
        Time-index (0-based) up to which to report the partial coverage.
    
    Returns
    -------
    pct_up_to_t : float
        Percentage of rectangle area visited at least once in timesteps [0..t].
    pct_total : float
        Percentage of rectangle area visited at least once in the entire trajectory.
    """
    xmin, xmax, ymin, ymax = bounds
    
    # number of voxels along each axis
    nx = int(np.floor((xmax - xmin) / voxel_size)) + 1
    ny = int(np.floor((ymax - ymin) / voxel_size)) + 1
    total_voxels = nx * ny
    
    # map coords → integer voxel indices and clamp in [0, n-1]
    # shape (T,2) → (T,) linear indices
    scaled = (trajectory - np.array([xmin, ymin])) / voxel_size
    ij = np.floor(scaled).astype(int)
    
    # clamp out-of-bounds points
    ij[:,0] = np.clip(ij[:,0], 0, nx-1)
    ij[:,1] = np.clip(ij[:,1], 0, ny-1)
    
    # convert 2D indices to linear indices
    lin_idx = ij[:,0] * ny + ij[:,1]
    
    # unique counts via np.unique
    unique_all = np.unique(lin_idx)
    unique_t   = np.unique(lin_idx[:t+1])
    
    pct_up_to_t = unique_t.size   / total_voxels * 100.0
    pct_total   = unique_all.size / total_voxels * 100.0
    
    return pct_up_to_t, pct_total

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

# ---------------- Main Simulation Loop ----------------
def main(ID, gains, robot_, simulated_minutes=1, training_fraction=0.8, noise_scales=(0 , 0), angular_std=0.33, two_dim=False, results_dir=None, trial=0):
    t1 = perf_counter()
    if (results_dir is None):
        results_dir = f"Results\\ID {ID}"
    # Initialize simulation components
    robot = robot_
    timestep_ms = int(robot.getBasicTimeStep())
    dt = timestep_ms / 1000.0  # Convert timestep to seconds   
    controller = DroneController(robot, FLYING_ATTITUDE)
    mmc = MixedModularCoder(gains=gains, two_dim=two_dim)
    rls = OptimisedRLS(mmc.ac_size, num_outputs=(2 if two_dim else 3)  , lambda_=0.999, delta=1e2) # for 3D

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

    # For online prediction:
    integrated_pos_log = []
    last_trained_step = 0
    predicted_pos_log = []
    prediction_mse_log = []
    pred_pos_short_log = []
    pred_mse_short_log = []
    execution_time = []
    
    training_done = False
    MAX_SIMULATION_TIME = (60 * simulated_minutes) + INITIAL_PAUSE # 1min in seconds * amount of hours
    UPDATE_INTERVAL = MAX_SIMULATION_TIME/10 #define amount of console updates by changing denominator
    TRAINING_TIME = (MAX_SIMULATION_TIME*training_fraction) + INITIAL_PAUSE # in seconds
    print(f'Starting Simulation\nID:{ID}, gains:{gains}')
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
            activity, pos_internal = mmc.update(velocity*dt, noise=neural_noise)

            # After training time is reached
            if (np.abs(TRAINING_TIME-elapsed_time) < COMMAND_TOLERANCE) and not training_done: # Command tolerance is one timestep
                last_trained_step = int(elapsed_time//dt) #index of last training data
                print(f'Training ends at {elapsed_time}s, planned at {TRAINING_TIME}s, Index={last_trained_step}')
                #rls_short = copy.deepcopy(rls)
                pred_pos_short_log = copy.deepcopy(predicted_pos_log)
                pred_mse_short_log = copy.deepcopy(prediction_mse_log)
                training_done = True                     

            # Long online prediction                        
            prediction_pos = rls.predict(activity)           
            predicted_pos_log.append(prediction_pos)
            prediction_mse_log.append(np.sum((prediction_pos-pos_internal)**2)/len(pos_internal))  # was compared to pos_real before
 
            if (not training_done):
                _ = rls.update(activity, pos_internal)  # returns (A, time) 
                #prediction_short = rls_short.predict(activity)
                #pred_pos_short_log.append(prediction_short)
                #pred_mse_short_log.append(np.sum((prediction_short-pos_internal)**2)/len(pos_internal))

            # saving values
            integrated_pos_log.append(pos_internal.copy())
            network_states.append(activity)
            activity_log.append(activity)
            #execution_time.append(time)
        position_log.append(position_real)
        acceleration_log.append(az_corrected)
        velocity_log.append(velocity)

        # fail save, in case drone flys too far out of arena radius
        if (np.linalg.norm(position_real) > 3*(2.5**2)):
            break
    
    # ---------------- End of Simulation ----------------
    print(f'Simulation finished at {elapsed_time/60:.0f} minutes')
    print('Calculating offline prediction...')
    # Predict the position using a linear model and plot the results
    pos_state = np.array(integrated_pos_log) # take 2D or 3D position and make array of it
    pos_state_short = pos_state[:last_trained_step+1] # for the short training take only the data until training end
    activity_short = np.array(network_states)[:last_trained_step+1]
    X, Y, y_pred, mse_mean, r2_mean = fit_linear_model(network_states, pos_state, train_index=last_trained_step, return_shuffled=False)
    X, Y, y_pred_short, mse_mean_short, r2_mean_short = fit_linear_model(activity_short, pos_state_short, return_shuffled=False)
    print(' - Predicted location data')
    if (trial == 0):
        print('Saving data...')
    # Save the results of the network
        data = {
            'name' : ID,
            'sim time' : simulated_minutes,
            'dt ms' : timestep_ms,
            'boundaries' : ARENA_BOUNDARIES,
            'gains' : gains,
            'modular projections' : mmc.projected_dim,
            'module operators' : mmc.A,
            'noise' : noise_scales,
            'activity' : network_states,
            'activity no noise' : activity_log,
            'velocity' : velocity_log,
            'acceleration' : acceleration_log,
            'position' : position_log,
            'position internal' : integrated_pos_log,
            'training time index' : last_trained_step,
            'online prediction' : predicted_pos_log,
            'online mse' : prediction_mse_log,
            'online short prediction' : pred_pos_short_log,
            'online short mse' : pred_mse_short_log,
            'offline prediction' : np.array(y_pred),
            'mse' : mse_mean,
            #'mse_shuffeled' : mse_shuffeled,
            'r2_mean' : r2_mean,
            #'r2_shuffeled' : r2_shuffeled,
            'offline short prediction' : np.array(y_pred_short),
            'mse short' : mse_mean_short,
            #'mse shuffeled short' : mse_shuffeled_short,
            'r2 mean short' : r2_mean_short,
            #'r2 shuffeled short' : r2_shuffeled_short,
            #'rls convergence matrix' : rls.convergence_metrics.copy()
            }
        filename = f'Data {ID}.pickle'
        save_object(data, f'{results_dir}\\{filename}')
        print(f' - Saved Data as {filename}')
    
    
    if (two_dim):
        train_perc, total_perc = visited_area_percentages(integrated_pos_log, ARENA_BOUNDARIES.flatten()[:4], t=last_trained_step)
    else:
        train_perc, total_perc = visited_volume_percentages(integrated_pos_log, ARENA_BOUNDARIES.flatten(), t=last_trained_step)

    print(f'Finished execution of ID {ID}')
    metrics = {
        'online mse' : prediction_mse_log,
        'online short mse' : pred_mse_short_log,
        'offline mse' : mse_mean,
        'offline mse short' : mse_mean_short,
        #'mse shuffeled' : mse_shuffeled,
        'r2_mean' : r2_mean,
        #'r2_shuffeled' : r2_shuffeled,
        #'mse shuffeled short' : mse_shuffeled_short,
        'r2 mean short' : r2_mean_short,
        #'r2 shuffeled short' : r2_shuffeled_short,
        'coverage' : total_perc,
        'coverage train' : train_perc,
        #'time rls' : np.mean(execution_time),
        #'time rr' : time
        'time total' : perf_counter() - t1
    }
    return metrics

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

if __name__ == '__main__':
    robot = Robot()
    supervisor = Supervisor()
    robot_node = supervisor.getFromDef("Crazyflie")
    trans_field = robot_node.getField("translation")
    INITIAL = [0, 0, 0]

    trial_per_setting = 20

    nr = [3, 4, 5]
    spacing = [0.1, 0.2, 0.3, 0.4]
    gain_list = generate_gain_lists(nr, spacing, start=0.2)
    #gains = [[0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6]]

    setting = gain_list
    times = 10.0 * np.ones(len(setting)) #
     #[0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00] #  Noise levels for each setting
    noise = 0.05 * np.ones(len(setting)) 

    for i, var in enumerate(setting):
        dim2 = False

        print(f'Running {trial_per_setting} trials for setting {i+1}/{len(setting)}')#'')
        #id_ = f'Benchmark 3D setting{i+1}of{len(setting)}'#'{len(setting)}'
        id_ = f'Paper Gain Setting {i}of{len(setting)}'
        data_all = []

        results_dir = f"Results\\ID {id_}"
        os.makedirs(results_dir, exist_ok=True)

        for trial in range(trial_per_setting):
            trans_field.setSFVec3f(INITIAL)
            robot_node.resetPhysics()
            data = main(ID=f'trial{trial}', gains=var, robot_=robot, simulated_minutes=times[i%3],
               training_fraction=0.8, noise_scales=(noise[i], 0.00), angular_std=0.33, two_dim=dim2,
              results_dir=results_dir, trial=trial)
            data_all.append(data)
            # noise scales = (Neural, Velocity)

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
            summary[f'avg {key}'] = np.mean(flat_vals)
            summary[f'std {key}'] = np.std(flat_vals)
            summary[f'per {key}'] = np.percentile(flat_vals, [25, 75])

        summary['setting var'] = setting
        summary['setting mode'] = 'gains|dimension'
        save_object(summary, f'{results_dir}\\summary.pickle')

        print(f"\n→ All trials done. Summary saved to {results_dir}\\Summary.pickle")
    print("Finished all settings. Completed execution.")
