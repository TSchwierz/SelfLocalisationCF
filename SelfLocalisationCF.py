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

#from asyncio.windows_events import NULL
#from multiprocessing.heap import Arena
import os
#from turtle import position
#from PIL.Image import ID
import numpy as np
from controller import Robot
#from controller import Supervisor
from DroneController import DroneController
from GridNetwork import MixedModularCoder
import PredictionModel
from datetime import datetime
import pickle
from PredictionModel import fit_linear_model

# ---------------- Simulation Parameters ----------------
FLYING_ATTITUDE = 0              # Base altitude (z-value) for flying
INITIAL_PAUSE = 6                # Time (in seconds) for the drone to lift off and stabilize
COMMAND_INTERVAL = 1           # Interval (in seconds) between new movement commands
COMMAND_TOLERANCE = 0.032        # Tolerance (in seconds) for command timing
MOVEMENT_MAGNITUDE = 0.25         # Magnitude of the movement vector in the plane
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
def main(ID, gains, robot_, simulated_minutes=1, predict_during_simulation=False, noise_scales=(0 , 0)):
    os.makedirs(f"Results\\ID {ID}", exist_ok=True)
    # Initialize simulation components
    robot = robot_
    timestep_ms = int(robot.getBasicTimeStep())
    dt = timestep_ms / 1000.0  # Convert timestep to seconds   
    controller = DroneController(robot, FLYING_ATTITUDE)
    mmc = MixedModularCoder(gains=gains)
    rls = PredictionModel.RLSRegressor(mmc.ac_size, num_outputs=3, lambda_=0.999, delta=1e2)

    neural_noise, velocity_noise = noise_scales 
    
    # Initialize state variables
    previous_direction = np.array([0, 0, 0])  # Initial xyz movement direction
    altitude = FLYING_ATTITUDE           # Constant base altitude
    target_altitude = altitude           # Target altitude (may be updated)
    elapsed_time = 0
    network_states = []
    position_log = []   
    velocity_log = []

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
                controller.change_gains_pid(kp=0.5, kd=1.0, ki=0.0)
                mmc.set_integrator(controller.get_location()) # get real position once to set integrator
                # Default movement: no change unless a new command is issued at the interval #2d for proper function
                movement_direction = np.array([0, 0])
            
        
            # Issue a new movement command at defined intervals (after the initial pause)
            if (elapsed_time % COMMAND_INTERVAL) <= COMMAND_TOLERANCE:
                movement_direction = update_direction(previous_direction, MOVEMENT_MAGNITUDE, dt, angular_std=0.01) # Update direction using a small-angle random walk     
                movement_direction = adjust_for_boundaries(ARENA_BOUNDARIES, position_real, movement_direction) # Adjust the movement to respect arena boundaries
                previous_direction = movement_direction  # Use the latest command as the basis for the next direction update
                #print(movement_direction)
        
            # Controller + Network Update
            position_real, velocity = controller.update(movement_direction)   
            noisy_velocity = velocity + np.random.normal(scale=velocity_noise, size=(3,))
            activity, pos_internal = mmc.update(velocity*dt)

            # Noise addition
            noise = np.random.normal(0, neural_noise, np.shape(activity))
            noisy_activity = np.clip(activity + noise, 0.0, 1.0)

            # learn using noisy position (internal integrator + global gps)
            # noise_ratio = 1.
            noisy_position = pos_internal #(noise_ratio*pos_internal + (1.-noise_ratio)*position_real)    

            # network online prediction
            if (predict_during_simulation):               
                prediction_pos = rls.predict(noisy_activity)
                predicted_pos_log.append(prediction_pos)
                prediction_mse_log.append((prediction_pos-position_real)**2)       
 
                rls.update(noisy_activity, noisy_position) # update using noise

            # saving values
            velocity_log.append(velocity)
            position_log.append(position_real)
            integrated_pos_log.append(pos_internal.copy())
            network_states.append(activity)
        
        # fail save, adjust for actual arena radius size
        if (np.linalg.norm(position_real) > 3*(2.5**2)):
            break
    
    # ---------------- End of Simulation ----------------
    print(f'Simulation finished at {elapsed_time/60:.0f} minutes')
    print(f' - Position log: shape {np.shape(position_log)}, min {np.min(position_log, axis=0)}, max {np.max(position_log, axis=0)}')
    print(f' - Network state: shape {np.shape(network_states)}, min {np.min(network_states)}, max {np.max(network_states)}')
    print('Calculating prediction...')
    if (predict_during_simulation):
        mse_mean = np.mean(prediction_mse_log)
        mse_shuffeled = 'NaN'
        r2_mean = 'NaN'
        r2_shuffeled = 'NaN'
    else:
         # Predict the position using a linear model and plot the results
        X, y, y_pred, mse_mean, mse_shuffeled, r2_mean, r2_shuffeled = fit_linear_model(network_states, position_log, return_shuffled=True)
        predicted_pos_log = np.array(y_pred)
    print(' - Predicted location data\nSaving data...')

    # Save the results of the network
    data = {
        'online' : predict_during_simulation,
        'name' : ID,
        'sim time' : simulated_minutes,
        'dt ms' : timestep_ms,
        'boundaries' : ARENA_BOUNDARIES,
        'noise' : noise_scales,
        'velocity' : velocity_log,
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
    print(f' - Saved Data as {filename}')
    
    print(f'Finished execution of ID {ID}')

if __name__ == '__main__':
    robot = Robot()
    #supervisor = Supervisor()
    #robot_node = supervisor.getFromDef("Crazyflie")
    #trans_field = robot_node.getField("translation")

    #INITIAL = [0, 0, 1]
    gains = [0.2, 0.4, 0.6, 0.8, 1.0]
    id_ = 'MovementCheck3D'

    #trans_field.setSFVec3f(INITIAL)
    #robot_node.resetPhysics()
    main(ID=id_, gains=gains, robot_=robot, simulated_minutes=5.0, predict_during_simulation=False)
