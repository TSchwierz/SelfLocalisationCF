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
import matplotlib.pyplot as plt
from controller import Robot
from controller import Supervisor
from DroneController import DroneController
from GridNetwork import GridNetwork
from datetime import datetime
import pickle

# ---------------- Simulation Parameters ----------------
FLYING_ATTITUDE = 1              # Base altitude (z-value) for flying
INITIAL_PAUSE = 6                # Time (in seconds) for the drone to lift off and stabilize
COMMAND_INTERVAL = 1             # Interval (in seconds) between new movement commands
COMMAND_TOLERANCE = 0.032        # Tolerance (in seconds) for command timing
MOVEMENT_MAGNITUDE = 1.0         # Magnitude of the movement vector in the xy-plane
DRIFT_COEFFICIENT = 0.03         # Lowered drift coefficient to reduce abrupt corrections
ARENA_BOUNDARIES = np.array([[-2.8, 2.8],  # x boundaries
                             [-2.8, 2.8],  # y boundaries
                             [0.5, 3.5]])      # z boundaries

# ---------------- Helper Functions ----------------
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

def compute_drift_vector(position, drift_coefficient=DRIFT_COEFFICIENT, arena_radius=5):
    """
    Compute a drift vector that nudges the drone toward the arena center.

    :param position: Current 3D position [x, y, z] of the drone.
    :param drift_coefficient: Scalar controlling the strength of the drift.
    :param arena_radius: Effective arena radius used for scaling.
    :return: A 2D drift vector for the xy-plane.
    """
    current_xy = np.array(position[:2])
    distance = np.linalg.norm(current_xy)
    if distance > 0:
        drift_direction = -current_xy / distance
    else:
        drift_direction = np.zeros(2)
    # Scale drift proportional to how far out the drone is (relative to arena_radius)
    return drift_direction * drift_coefficient * (distance / arena_radius)

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
    return adjusted_vector[:2], adjusted_vector[2]

def get_new_altitude(current_altitude, sigma=0.25):
    """
    Generate a new altitude based on a normal distribution centered around current_altitude.

    :param current_altitude: [Float] The current height at which the drone is flying.
    :param sigma: [Float, DEFAULT=0.25] The standart variance on the new height
    
    :return: A new altitude value.
    """
    return np.random.normal(current_altitude, sigma)

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

# ---------------- Main Simulation Loop ----------------
def main(ID, gains, robot_, simulated_hours=1):
    os.makedirs(f"Results\\ID{ID}", exist_ok=True)
    # Initialize simulation components
    robot = robot_
    timestep_ms = int(robot.getBasicTimeStep())
    dt = timestep_ms / 1000.0  # Convert timestep to seconds
    controller = DroneController(robot, FLYING_ATTITUDE)
    grid_network = GridNetwork(9, 10) # make a network with Nx=12 x Ny=12 neurons 
    grid_network.set_gains(gains)
    #grid_network = load_object('data.pickle')
    
    # Initialize state variables
    previous_direction = np.array([0, 0])  # Initial xy movement direction
    yaw = 0                              # Initial yaw (no rotation)
    altitude = FLYING_ATTITUDE           # Constant base altitude
    target_altitude = altitude           # Target altitude (may be updated)
    elapsed_time = 0
    network_states = []
    position_log = []
    current_position = np.array([0, 0])
    
    MAX_SIMULATION_TIME = 3600 * simulated_hours # 1h in seconds * amount of hours
    UPDATE_INTERVAL = MAX_SIMULATION_TIME/10 #define amount of updates by changing denominator
    print(f'Starting Simulation\nID:{ID}, gains:{gains}')
    # Main loop: run until simulation termination signal or time limit reached
    while robot.step(timestep_ms) != -1 and elapsed_time < MAX_SIMULATION_TIME:
        elapsed_time += dt  # Update elapsed time in seconds
        
        if (elapsed_time%UPDATE_INTERVAL<= dt):
            print(f'{datetime.now().time()} - simulated time: {int(elapsed_time/60)} of {MAX_SIMULATION_TIME/60} minutes {elapsed_time/MAX_SIMULATION_TIME:.1%}')

        # Default movement: no change unless a new command is issued at the interval
        movement_direction = np.array([0, 0])
        yaw = 0
        
        # Issue a new movement command at defined intervals (after the initial pause)
        if elapsed_time >= INITIAL_PAUSE and (elapsed_time % COMMAND_INTERVAL) <= COMMAND_TOLERANCE:
            # Generate random movement commands
            target_altitude = altitude # keeping it constant for now #get_new_altitude(altitude) 
            smooth_direction = update_direction(previous_direction, MOVEMENT_MAGNITUDE, dt, angular_std=0.33) # Update xy-direction using a small-angle random walk        
            drift = compute_drift_vector(np.concatenate((current_position, [altitude]))) # Add drift toward the arena center
            movement_direction = smooth_direction + drift
            
            # Form the complete 3D movement vector
            current_3d_position = np.concatenate((current_position, [altitude]))
            movement_3d = np.concatenate((movement_direction, [target_altitude]))
            
            # Adjust the movement to respect arena boundaries
            movement_direction[:2], target_altitude = adjust_for_boundaries(ARENA_BOUNDARIES, current_3d_position, movement_3d)

            #movement_direction = (velocity + movement_direction)/2 # use the average between new direction and current velocity for smoothness
            previous_direction = movement_direction  # Use the latest command as the basis for the next direction update
            #yaw += 0.2
        
        # Update the drone's state with the new movement command
        current_position, velocity, altitude = controller.update(movement_direction, yaw, target_altitude)       
        grid_network.update_network(velocity*dt)
        
        position_log.append(current_position)
        network_states.append(grid_network.network_activity.copy())
    
    # ---------------- End of Simulation ----------------
    print(f'Simulation finished at {elapsed_time/60:.0f} minutes')
    print(f' - Position log: shape {np.shape(position_log)}, min {np.min(position_log, axis=0)}, max {np.max(position_log, axis=0)}')
    print(f' - Network state: shape {np.shape(network_states)}, min {np.min(network_states)}, max {np.max(network_states)}')
    
    # Compute the effective arena size (maximum radial distance reached)
    arena_size = np.sqrt(np.max(np.sum(np.array(position_log)**2, axis=1)))
    print(f' - effective Arena size: {arena_size}')
    
    # Visualize the network activity and complete path coverage
    print('Generating Images...')
    grid_network.plot_frame_figure(positions_array=position_log, network_activity=network_states, num_bins=60, ID=ID)

    # Generate activity plots for each neuron
    #grid_network.plot_activity_neurons(np.array(position_log), num_bins=60, neuron_range=range(grid_network.N), network_activity=np.array(network_states), ID=ID)
    print('Saved activity plot\nCalculating prediction...')
    
    # Predict the position using a linear model and plot the results
    X, y, y_pred, mse_mean, mse_shuffeled, r2_mean, r2_shuffeled = grid_network.fit_linear_model(network_states, position_log, return_shuffled=True)
    grid_network.plot_prediction_path(y, y_pred, mse_mean, r2_mean, ID=ID)
    print('Saved prediction plot')

    # Save the results of the network
    save_object(grid_network, f'Results\\ID{ID}\\network{ID}.pickle')
    with open(f'Results\\SummaryResults.txt', 'a') as file:
        file.write(f'ID:{ID}, gains:{gains}\nmse_mean:{mse_mean}\tmse_shuffeled:{mse_shuffeled}\nr2_mean:{r2_mean}\tr2_shuffeled:{r2_shuffeled}\n --- \n')
        file.close()
    print(f'Saved Network.')
    print(f'Finished execution of ID{ID}')
    return mse_mean

if __name__ == '__main__':
    robot = Robot()
    supervisor = Supervisor()

    robot_node = supervisor.getFromDef("Crazyflie")
    trans_field = robot_node.getField("translation")
    INITIAL = [0, 0, 1]

    # define a set of gains to be tested for best performance
    gain_list = [[0.1, 0.4, 0.7, 1.0],
                 [0.1, 0.4, 0.7, 1.0, 1.3],
                 [0.2, 0.7, 1.2, 1.7, 2.2],
                 [0.1, 0.2, 0.3, 0.4, 0.5],
                 [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                 [0.3, 0.5, 0.8, 1.2, 1.8, 2.1],
                 [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
                 [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]]

    nr = [3, 4, 5, 6]
    spacing = [0.1, 0.2, 0.3, 0.4]
    gain_list = generate_gain_lists(nr, spacing, start=0.2)

    mse_means = []

    for i, gains in enumerate(gain_list):
        trans_field.setSFVec3f(INITIAL)
        robot_node.resetPhysics()
        mse_means.append(main(ID=i, gains=gains, robot_=robot, simulated_hours=0.15))

    plot_fitting_results(nr, spacing, mse_means)
