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

from asyncio.windows_events import NULL
from multiprocessing.heap import Arena
import os
from PIL.Image import ID
import numpy as np
import matplotlib.pyplot as plt
from controller import Robot
#from controller import Supervisor
from DroneController import DroneController
from GridNetwork import GridNetwork, MixedModularCoder
import PredictionModel
from datetime import datetime
import pickle
from PredictionModel import fit_linear_model, plot_prediction_path

# ---------------- Simulation Parameters ----------------
FLYING_ATTITUDE = 0              # Base altitude (z-value) for flying
INITIAL_PAUSE = 6                # Time (in seconds) for the drone to lift off and stabilize
COMMAND_INTERVAL = 1           # Interval (in seconds) between new movement commands
COMMAND_TOLERANCE = 0.032        # Tolerance (in seconds) for command timing
MOVEMENT_MAGNITUDE = 0.25         # Magnitude of the movement vector in the xy-plane
NEURAL_NOISE_VAR = 0.01 * 5#%    # The value of the standart variation for noise on network activity (normalised to 0.0-1.0)
ARENA_BOUNDARIES = np.array([[-2.5, 2.5],  # x boundaries
                             [-2.5, 2.5],  # y boundaries
                             [-2.5, 2.5]]) # z boundaries


# ---------------- Helper Functions ----------------
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

def load_object(filename):
    '''
    load an object from a file using the built-in pickle library

    :param filename: Name of the file from which to load data
    '''
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as ex:
        print("Error during unpickling object (Possibly unsupported):", ex)

def plot_modular_activity(mmc, pos_log, ac_log, ID):
    pos2D = mmc.project_positions(pos_log)
    activity = np.array(ac_log)
    for i, m in enumerate(mmc.Module):
        m.plot_frame_figure(pos2D[i], 50, activity[:,i], ID=ID, subID=f'mod{i}')

def plot_3d_trajectory(pos, ID='null'):
    x, y, z = pos[:,0], pos[:,1], pos[:,2]

    start, stop = 0, len(x)
    time = np.linspace(start, stop, stop) / stop
    plt.plot(time, x[start:stop], 'b:', label='x')
    plt.plot(time, y[start:stop], 'r:', label='y')
    plt.plot(time, z[start:stop], 'g:', label='z')
    plt.axhline(ARENA_BOUNDARIES[0,0], c='c', label = 'x-y limit')
    plt.axhline(ARENA_BOUNDARIES[0,1], c='c')
    plt.axhline(ARENA_BOUNDARIES[2,0], c='g', label = 'alt limit')
    plt.axhline(ARENA_BOUNDARIES[2,1], c='g')
    plt.ylim(-3, 3.5)
    plt.grid()
    plt.legend()

    plt.savefig(f'Results\\ID{ID}\\3d_trajectory_timeplot.png', format='png')
    plt.close()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x[:stop], y[:stop], z[:stop], label='path')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()

    plt.savefig(f'Results\\ID{ID}\\3d_trajectory_spaceplot.png', format='png')
    plt.close()

def plot_fitting_results(n, spacing, score):
    '''
    Plot the scoring of gain configurations as a heatmap based on the number of gains and the spacing between them
    
    :param n: list with the amount of gains
    :param spacing: list with the spacings between gains
    :param score: a flattened list of size (n, spacing) that contains the score of each configuration 
    '''
    heatmap = np.array(score).reshape((len(n), len(spacing)))
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    ax.set_yticks(range(len(n)), labels=n,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(range(len(spacing)), labels=spacing)

    for i in range(len(n)):
        for j in range(len(spacing)):
            text = ax.text(j, i, f'ID{i*j+j}\n{heatmap[i, j]}',
                           ha="center", va="center", color="w")

    ax.set_title("MSE Scoring over number (y) and spacing (x) of gains")
    fig.colorbar(im)
    fig.tight_layout()
    plt.savefig(f'Results\\best_gain_results.png', format='png') # save in relative folder Results in Source/Repos/SelfLocalisationCF
    plt.close()

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

def update_direction(current_direction, magnitude, dt, angular_std=0.1):
    """
    Update the drone's movement direction using a small random angular increment.
    
    This function implements a simple random walk in the angular domain. At each update, the current
    angle is perturbed by a small random value (scaled by sqrt(dt) for consistency with time resolution),
    producing smooth and continuous changes in the movement direction.

    :param current_direction: Current 2D movement vector.
    :param magnitude: Desired magnitude of the new movement vector.
    :param dt: Time step (in seconds).
    :param angular_std: Standard deviation of the angular change (in percentage of pi).
    :return: Updated 2D movement vector with the specified magnitude.
    """
    # If the current direction is nearly zero, choose a random initial angle
    if np.linalg.norm(current_direction) < 1e-6:
        current_angle = np.random.uniform(-np.pi, np.pi)
    else:
        current_angle = np.arctan2(current_direction[1], current_direction[0])
    
    # Add a small random angular change; using sqrt(dt) for time scaling
    d_angle = np.random.normal(0, angular_std*np.pi) #* np.sqrt(dt))
    new_angle = current_angle + d_angle
    # Ensure the angle remains in the interval (-pi, pi)
    new_angle = (new_angle + np.pi) % (2 * np.pi) - np.pi
    return np.array([np.cos(new_angle), np.sin(new_angle)]) * magnitude

def update_direction3D(current_direction, magnitude, dt, angular_std=0.25):
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
def main(ID, gains, robot_, simulated_minutes=1, predict_during_simulation=False):
    os.makedirs(f"Results\\ID {ID}", exist_ok=True)
    # Initialize simulation components
    robot = robot_
    timestep_ms = int(robot.getBasicTimeStep())
    dt = timestep_ms / 1000.0  # Convert timestep to seconds
    controller = DroneController(robot, FLYING_ATTITUDE)
    
    #grid_network = GridNetwork(10, 9) # make a network with Nx=10 x Ny=9 neurons 
    #grid_network.set_gains(gains)
    #grid_network = load_object('data.pickle')
    mmc = MixedModularCoder(gains=gains)
    rls = PredictionModel.RLSRegressor(mmc.ac_size, num_outputs=3, lambda_=0.999, delta=1e2)
    
    # Initialize state variables
    previous_direction = np.array([0, 0, 0])  # Initial xyz movement direction
    altitude = FLYING_ATTITUDE           # Constant base altitude
    target_altitude = altitude           # Target altitude (may be updated)
    elapsed_time = 0
    network_states = []
    position_log = []
    position_real = np.array(robot_.getDevice("gps").getValues())
    # For online prediction:
    predicted_pos_log = []
    prediction_mse_log = []
    integrated_pos_log = []
    
    MAX_SIMULATION_TIME = 60 * simulated_minutes # 1min in seconds * amount of hours
    UPDATE_INTERVAL = MAX_SIMULATION_TIME/10 #define amount of updates by changing denominator
    print(f'Starting Simulation\nID:{ID}, gains:{gains}')
    # Main loop: run until simulation termination signal or time limit reached
    while robot.step(timestep_ms) != -1 and elapsed_time < MAX_SIMULATION_TIME:
        elapsed_time += dt  # Update elapsed time in seconds

        # print progress
        if (elapsed_time%UPDATE_INTERVAL<= dt):
            print(f'{datetime.now().time()} - simulated time: {int(elapsed_time/60)} of {MAX_SIMULATION_TIME/60} minutes {elapsed_time/MAX_SIMULATION_TIME:.1%}')
        
        # Initial start-up phase (Drone needs to get to stable hover altitude)
        if (elapsed_time < INITIAL_PAUSE):
            position_real, velocity = controller.update([0,0]) # desired initial position

        # After Initial start-up
        else:
            if(controller.initial_pid):
                controller.change_gains_pid(kp=2.0, kd=3.0, ki=0.0)

            # Default movement: no change unless a new command is issued at the interval #2d for proper function
            movement_direction = np.array([0, 0])
        
            # Issue a new movement command at defined intervals (after the initial pause)
            if (elapsed_time % COMMAND_INTERVAL) <= COMMAND_TOLERANCE:
                movement_direction = update_direction3D(previous_direction, MOVEMENT_MAGNITUDE, dt, angular_std=0.5) # Update direction using a small-angle random walk     
                movement_direction = adjust_for_boundaries(ARENA_BOUNDARIES, position_real, movement_direction) # Adjust the movement to respect arena boundaries
                previous_direction = movement_direction  # Use the latest command as the basis for the next direction update
                #print(movement_direction)
        
            # Controller + Network Update
            position_real, velocity = controller.update(movement_direction)
            #grid_network.update_network(velocity[:2]*dt)
            #activity = grid_network.network_activity.copy()
            activity, pos_internal = mmc.update(velocity*dt)


            # network online prediction
            if (predict_during_simulation):
                noisy_activity = np.clip(activity + np.random.normal(0, NEURAL_NOISE_VAR, np.shape(activity)), 0.0, 1.0)
                prediction_pos = rls.predict(noisy_activity)
                predicted_pos_log.append(prediction_pos)
                prediction_mse_log.append((prediction_pos-position_real)**2)
                integrated_pos_log.append(pos_internal)

                # learn using noisy position (internal integrator + global gps)
                noisy_position = (0.05*position_real + 0.95*pos_internal)            
                rls.update(noisy_activity, noisy_position) # update using noise

            # saving values
            position_log.append(position_real)
            network_states.append(activity)
        
        # fail save, adjust for actual arena radius size
        if (np.linalg.norm(position_real) > 3*(2.5**2)):
            break
    
    # ---------------- End of Simulation ----------------
    print(f'Simulation finished at {elapsed_time/60:.0f} minutes')
    print(f' - Position log: shape {np.shape(position_log)}, min {np.min(position_log, axis=0)}, max {np.max(position_log, axis=0)}')
    print(f' - Network state: shape {np.shape(network_states)}, min {np.min(network_states)}, max {np.max(network_states)}')
    
    # Compute the effective arena size (maximum radial distance reached)
    arena_size = np.sqrt(np.max(np.sum(np.array(position_log)**2, axis=1)))
    print(f' - effective Arena size: {arena_size}')

    # Visualize the network activity and complete path coverage
    #print('Generating Images...')
    # 3D path
    #plot_3d_trajectory(np.array(position_log), ID)
    #plot_modular_activity(mmc, position_log, network_states, ID)
 
    #grid_network.plot_frame_figure(positions_array=position_log, network_activity=network_states, num_bins=60, ID=ID)
    # Generate activity plots for each neuron
    #grid_network.plot_activity_neurons(np.array(position_log), num_bins=60, neuron_range=range(grid_network.N), network_activity=np.array(network_states), ID=ID)
    
    print('Calculating prediction...')
    if (predict_during_simulation):
        mse_mean = np.mean(prediction_mse_log)
        #PredictionModel.plot_prediction_path(np.array(position_log), np.array(predicted_pos_log), mse_mean, ID=ID)
        mse_shuffeled = 'NaN'
        r2_mean = 'NaN'
        r2_shuffeled = 'NaN'
    else:
         # Predict the position using a linear model and plot the results
        X, y, y_pred, mse_mean, mse_shuffeled, r2_mean, r2_shuffeled = fit_linear_model(network_states, position_log, return_shuffled=True)
        #plot_prediction_path(y, y_pred, mse_mean, ID=ID)
    print('Predicted location data')

    # Save the results of the network
    data = {
        'online' : predict_during_simulation,
        'name' : ID,
        'sim time' : simulated_minutes,
        'dt ms' : timestep_ms,
        'radius' : arena_size,
        'boundaries' : ARENA_BOUNDARIES,
        'position' : position_log,
        'position internal' : integrated_pos_log,
        'position prediction' : predicted_pos_log,
        'activity' : network_states,
        'predicted mse' : prediction_mse_log,
        'mse' : mse_mean,
        'mse_shuffeled' : mse_shuffeled,
        'r2_mean' : r2_mean,
        'r2_shuffeled' : r2_shuffeled,
        'gains' : gains,
        'modular projections' : mmc.projected_dim,
        'module operators' : mmc.A
        }
    filename = f'Data {ID}.pickle'
    save_object(data, f'Results\\ID {ID}\\{filename}')
    print(f'Saved Data as {filename}')
    
    print(f'Finished execution of ID {ID}')

if __name__ == '__main__':
    robot = Robot()
    #supervisor = Supervisor()
    #robot_node = supervisor.getFromDef("Crazyflie")
    #trans_field = robot_node.getField("translation")

    #INITIAL = [0, 0, 1]
    gains = [0.2, 0.4, 0.6, 0.8, 1.0]
    id_ = 'MMC Online'

    #trans_field.setSFVec3f(INITIAL)
    #robot_node.resetPhysics()
    main(ID=id_, gains=gains, robot_=robot, simulated_minutes=6.0, predict_during_simulation=True)

    #plot_fitting_results(nr, spacing, mse_means)
