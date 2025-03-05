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

import numpy as np
from controller import Robot
from DroneController import DroneController
from GridNetwork import GridNetwork

# ---------------- Simulation Parameters ----------------
FLYING_ATTITUDE = 1              # Base altitude (z-value) for flying
INITIAL_PAUSE = 6                # Time (in seconds) for the drone to lift off and stabilize
COMMAND_INTERVAL = 1             # Interval (in seconds) between new movement commands
COMMAND_TOLERANCE = 0.032        # Tolerance (in seconds) for command timing
MOVEMENT_MAGNITUDE = 1.0         # Magnitude of the movement vector in the xy-plane
DRIFT_COEFFICIENT = 0.03         # Lowered drift coefficient to reduce abrupt corrections
ARENA_BOUNDARIES = np.array([[-4.8, 4.8],  # x boundaries
                             [-4.8, 4.8],  # y boundaries
                             [0, 4]])      # z boundaries

# ---------------- Helper Functions ----------------
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

def get_new_altitude():
    """
    Generate a new altitude based on a normal distribution centered around FLYING_ATTITUDE.
    
    :return: A new altitude value.
    """
    return np.random.normal(FLYING_ATTITUDE, 0.25)

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
def main():
    # Initialize simulation components
    robot = Robot()
    timestep_ms = int(robot.getBasicTimeStep())
    dt = timestep_ms / 1000.0  # Convert timestep to seconds
    controller = DroneController(robot, FLYING_ATTITUDE)
    grid_network = GridNetwork(12, 12)
    
    # Initialize state variables
    previous_direction = np.array([0, 0])  # Initial xy movement direction
    yaw = 0                              # Initial yaw (no rotation)
    altitude = FLYING_ATTITUDE           # Constant base altitude
    target_altitude = altitude           # Target altitude (may be updated)
    elapsed_time = 0
    network_states = []
    position_log = []
    current_position = np.array([0, 0])
    
    MAX_SIMULATION_TIME = 3600 * 0.1 # 1h in seconds * amount of hours
    UPDATE_INTERVAL = MAX_SIMULATION_TIME/10 #define amount of updates by changing denominator
    print('Starting Simulation')
    # Main loop: run until simulation termination signal or time limit reached
    while robot.step(timestep_ms) != -1 and elapsed_time < MAX_SIMULATION_TIME:
        elapsed_time += dt  # Update elapsed time in seconds
        
        if (elapsed_time%UPDATE_INTERVAL<= dt):
            print(f'elapsed time {elapsed_time/60:.0} of {MAX_SIMULATION_TIME/60} minutes {100*elapsed_time/MAX_SIMULATION_TIME}%')

        # Default movement: no change unless a new command is issued at the interval
        movement_direction = np.array([0, 0])
        yaw = 0
        
        # Issue a new movement command at defined intervals (after the initial pause)
        if elapsed_time >= INITIAL_PAUSE and (elapsed_time % COMMAND_INTERVAL) <= COMMAND_TOLERANCE:
            target_altitude = altitude  # Here altitude is kept constant; can be randomized if desired
            
            # Update xy-direction using a small-angle random walk
            smooth_direction = update_direction(previous_direction, MOVEMENT_MAGNITUDE, dt, angular_std=0.33)
            # Add drift toward the arena center
            drift = compute_drift_vector(np.concatenate((current_position, [altitude])))
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
        grid_network.update_network(velocity, get_next_state=False)
        
        position_log.append(current_position)
        network_states.append(grid_network.network_activity.copy())
    
    # ---------------- End of Simulation ----------------
    print(f'Simulation finished at {elapsed_time/60:.0f} minutes')
    print(f' - Position log: shape {np.shape(position_log)}, min {np.min(position_log, axis=0)}, max {np.max(position_log, axis=0)}')
    print(f' - Network state: shape {np.shape(network_states)}, min {np.min(network_states)}, max {np.max(network_states)}')
    
    # Compute the effective arena size (maximum radial distance reached)
    arena_size = np.sqrt(np.max(np.sum(np.array(position_log)**2, axis=1)))
    print(f' - effective Arena size: {arena_size}')
    
    # Visualize the network activity and prediction of the drone's path
    print('Generating Images...')
    grid_network.plot_frame_figure(positions_array=position_log, network_activity=network_states, num_bins=60, arena_size=arena_size)
    print('Saved activity plot\nCalculating prediction...')
    
    # Predict the position using a linear model and plot the results
    X, y, y_pred, mse_mean, r2_mean = grid_network.fit_linear_model(network_states, position_log)
    grid_network.plot_prediction_path(y, y_pred, mse_mean, r2_mean)
    print('Saved prediction plot')

if __name__ == '__main__':
    main()
