import numpy as np
from controller import Robot
from CFController import controller
from GridNetwork import GridNetwork as gn

# Initialise needed instances
FLYING_ATTITUDE = 1
robot = Robot()
timestep = int(robot.getBasicTimeStep())
control = controller(robot, FLYING_ATTITUDE)
network = gn(9,10)

def generate_biased_vector(previous_vector: np.ndarray, size: float, bias: float) -> np.ndarray:
    """
    Generates a new vector in the xy-plane with a given magnitude (size).
    The direction is Gaussian distributed around the angle of the previous vector.
    The bias determines the tendency to keep the new direction aligned with the previous one.
    If the bias is lower than 0, the new angle will be drawn from a uniform distribution.

    :param previous_vector: np.ndarray, the previous vector in the xy-plane (2D array-like)
    :param size: float, the magnitude of the new vector
    :param bias: float, a value between -inf and 1 determining the tendency to align with the previous vector
    :return: np.ndarray, the new generated vector in xy-space
    """
    if np.linalg.norm(previous_vector) == 0:
        base_angle = np.random.uniform(0, 2 * np.pi)
    else:
        base_angle = np.arctan2(previous_vector[1], previous_vector[0])

    if bias < 0.0:
        new_angle = np.random.uniform(-np.pi, np.pi)
    else:
        sigma = (1 - bias) * np.pi  # Standard deviation for the Gaussian spread
        new_angle = np.random.normal(base_angle, sigma) #% (2 * np.pi) # Sample a new angle from a Gaussian distribution centered around base_angle
    
    new_vector = np.array([np.cos(new_angle), np.sin(new_angle)]) * size # Convert polar to Cartesian coordinates
    return new_vector

# Simulation Constants
initial_pause = 6 #(in s) amount of time at the start for the drone to lift off and stabilise
modi_d = 2 #(in s) the interval with which a new direction commands should be given
modi_y = 1.5 #(in s) interval after which a new yaw direction is given
modi_pr = 0.032 #(in s) setting this to 32ms (equal to the robot timestep) ensures only one new command per interval
size = 0.1 # magnitude of movement vector
bias = -1 # randomness of movement 

# Initialising state variables
prev_direction = np.array([0, 0]) # initial direction of (xy)-movement
direction = prev_direction 
yaw = 0 # initial yaw
elapsed_time = 0
network_state = []
position_log = []

print('Starting Simulation')
# Main loop:
while robot.step(timestep) != -1 and elapsed_time < 360*4: # max time in hours #:
    elapsed_time += (timestep/1000) # ms to s
    direction=[0,0] # set direction to no movement
    yaw = 0 # no yaw adjustment

    # new direction 
    if (elapsed_time>=initial_pause and elapsed_time%modi_d<=modi_pr): # after start-off and in 2s interval
        direction = generate_biased_vector(prev_direction, size, bias) # get a random new direction to move to
        print(f'time={elapsed_time:.4}, direction={direction}, new direction angle {np.arctan2(direction[1], direction[0])/(0.5*np.pi):.2}')

    # new yaw
    if (elapsed_time>=initial_pause and elapsed_time%modi_y<=(modi_pr) and elapsed_time%modi_d>(modi_pr)): # after start-off and in modi_d (s) interval but not during translation
       yaw = np.random.uniform(-np.pi,np.pi) # turn around by this value
       print(f'time={elapsed_time:.4}, new yaw={yaw}')

    position, velocity = control.update(direction, yaw) # pass new desired state to control and update
    prev_direction = velocity 
    network.update_network(velocity, get_next_state=False) # update grid network with velocity
    position_log.append(position)
    network_state.append(network.network_activity.copy())

print(f'Simulation finished at time={elapsed_time/60:.0}minutes\nGenerating image')
print(f'Position log (shape, min, max):{np.shape(position_log)}, {np.min(position_log)}, {np.max(position_log)}')
print(f'Network (shape, min, max):{np.shape(network_state)}, {np.min(network_state)}, {np.max(network_state)}')

# Plot the network activity
network.plot_frame_figure(positions_fig=position_log, network_activity=network_state, num_bins=60)
print('Saved activity plot\ncalculating prediction')

# Predict position
X, y, y_pred, mse_mean, r2_mean = network.fit_linear_model(network_state, position_log)
network.plot_prediction_path(y, y_pred, mse_mean, r2_mean)
print('Saved prediction plot')