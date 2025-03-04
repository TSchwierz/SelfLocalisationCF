import numpy as np
from controller import Robot
from DroneController import DroneController as dc
from GridNetwork import GridNetwork as gn

# Initialise needed instances
FLYING_ATTITUDE = 1
robot = Robot()
timestep = int(robot.getBasicTimeStep())
control = dc(robot, FLYING_ATTITUDE)
network = gn(9,10)
'''
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
    bias = np.clip(bias, -1, 1) # ensures the bias is in the correct range

    if np.linalg.norm(previous_vector) == 0: # if previous_vector is zero (no direction)
        base_angle = np.random.uniform(-np.pi, np.pi)
    else:
        base_angle = np.arctan2(previous_vector[1], previous_vector[0])

    if bias < 0.0:
        new_angle = np.random.uniform(-np.pi, np.pi)
    else:
        sigma = (1 - bias) * np.pi  # Standard deviation for the Gaussian spread
        new_angle = np.random.normal(base_angle, sigma) #% (2 * np.pi) # Sample a new angle from a Gaussian distribution centered around base_angle
    
    new_vector = np.array([np.cos(new_angle), np.sin(new_angle)]) * size # Convert polar to Cartesian coordinates
    return new_vector

def generate_new_yaw(bias=0):
    """
    Generate a random angle in the range (-pi, pi) based on a bias parameter.
    
    - If bias is 1, the angle is highly likely to be near 0.
    - If bias is 0, the angle follows an approximately uniform distribution.
    - If bias is -1, the angle is highly likely to be near -pi/2 or pi/2.
    
    :param bias: A float between -1 and 1 controlling the distribution bias.
    :return: A float representing an angle in the range (-pi, pi).
    """
    bias = np.clip(bias, -1, 1)  # Ensure bias is within valid range
    
    # Mixture of distributions based on bias
    if bias == 0:
        return np.random.uniform(-np.pi, np.pi)  # Uniform distribution
    
    # Create a probability distribution that smoothly varies with bias
    if bias > 0:
        scale = 1 - bias  # Control spread around 0
        angle = np.random.normal(0, scale * np.pi)  # Gaussian around 0
    else:
        scale = 1 + bias  # Control spread around +-pi/2
        peak_choice = np.random.choice([-np.pi/2, np.pi/2])  # Pick a peak
        angle = np.random.normal(peak_choice, scale * np.pi / 2)
    
    # Ensure angle remains in (-pi, pi)
    return (angle + np.pi) % (2 * np.pi) - np.pi
'''
def drift_vector(position, drift_coefficient=0.01, arena_radius=5):
     # Compute the drift toward the origin
     current_xy = np.array(position[:2])
     distance = np.linalg.norm(current_xy)
     if distance != 0:
        # The drift direction is opposite to the current position vector
        drift_direction = -current_xy #/ distance
     else:
        drift_direction = np.array([0, 0])
    
     return drift_direction * drift_coefficient * (distance/arena_radius)

def check_for_boundaries(boundaries, position, move_towards):
    new_pos = position + move_towards  # Compute new position
    lower_bounds, upper_bounds = boundaries[:, 0], boundaries[:, 1]  # Extract lower & upper bounds
    
    # Check if new_pos is outside boundaries
    out_of_bounds = (new_pos < lower_bounds) | (new_pos > upper_bounds)
    move_towards[out_of_bounds] *= -0.5  # reverse (and scale down) if out-of-bounds
    if (np.sum(out_of_bounds)>0):
        print(f'Boundary Proximity! mask = {out_of_bounds}, new vector = {move_towards}')

    return move_towards[:2], move_towards[2]  # Return updated direction & height

def generate_new_height():
    return np.random.normal(FLYING_ATTITUDE, 0.25)

def new_direction(v_old, size, dt):
    angle_old = np.arctan2(v_old[1], v_old[0])
    # Update using an Ornstein-Uhlenbeck process
    theta=1.0
    mu=0
    sigma=0.1   
    angle_new = angle_old + theta * (mu - angle_old) * dt + sigma * np.sqrt(dt) * np.random.normal()
    return np.array([np.cos(angle_new), np.sin(angle_new)]) * size


# Simulation Constants
initial_pause = 6 #(in s) amount of time at the start for the drone to lift off and stabilise
#modi_d = 2 #(in s) the interval with which a new direction commands should be given
#modi_y = 1.5 #(in s) interval after which a new yaw direction is given
modi_pr = 0.032 #(in s) setting this to 32ms (equal to the robot timestep) ensures only one new command per interval
modi = 1 #(s) time between two movement commands
size = 1.0 # magnitude of movement vector
#bias_d = 0.45 # (-1, 1) randomness of movement direction
#bias_y = 0 # (-1, 1) randomness of rotation angle
drift_coef = 0.05  # adjust new directions to point more towards the origin
choices = ['translation', 'rotation', 'altitude']
p_choices = [1.0, 0.0, 0.0]
boundaries = np.array([[-4.8, 4.8],[-4.8, 4.8],[0, 4]]) # [[-x, +x][-y, +y][z=0, z=max]]

if sum(p_choices) != 1.0:
    raise ValueError("Probabilities should add to 1!")

# Initialising state variables
prev_direction = np.array([0, 0]) # initial direction of (xy)-movement
direction = prev_direction 
yaw = 0 # initial yaw
altitude = FLYING_ATTITUDE
height = altitude
elapsed_time = 0
network_state = []
position_log = []
position = np.array([0, 0])

print('Starting Simulation')
# --------- Main loop: ------------
while robot.step(timestep) != -1 and elapsed_time < 360*10: # max time in hours #:
    elapsed_time += (timestep/1000) # ms to s
    direction=[0,0] # set direction to no movement
    yaw = 0 # no yaw adjustment
    
    if (elapsed_time >= initial_pause and elapsed_time%modi<=modi_pr):
        height = altitude;
        action = np.random.choice(choices, p=p_choices)
        match action:
            case 'translation':
                aux_normal = new_direction(prev_direction, size, timestep) #generate_biased_vector(prev_direction, size, bias_d)
                aux_drift = drift_vector(position, drift_coef)
                #direction = generate_biased_vector(prev_direction, size, bias_d) + drift_vector(position, drift_coef)
                direction = aux_normal + aux_drift
                print(f'Translation (normal+drift): {aux_normal}+{aux_drift}')
            case 'rotation':
                #yaw = generate_new_yaw(bias=bias_y)
                print('Deprecated yaw adjustment')
            case 'altitude':
                height = generate_new_height()

        # construct 3d position from [x y] and z
        pos_3d = np.concatenate((position, np.array([altitude])))
        move_to = np.concatenate((direction, np.array([height]))) 
        direction, height = check_for_boundaries(boundaries, pos_3d, move_to) # turn movement if it would collide with wall

    ''' Old deprecated method
    # new direction 
    if (elapsed_time>=initial_pause and elapsed_time%modi_d<=modi_pr): # after start-off and in 2s interval
        direction = generate_biased_vector(prev_direction, size, bias) # get a random new direction to move to
        print(f'time={elapsed_time:.4}, direction={direction}, new direction angle {np.arctan2(direction[1], direction[0])/(0.5*np.pi):.2}')

    # new yaw
    if (elapsed_time>=initial_pause and elapsed_time%modi_y<=(modi_pr) and elapsed_time%modi_d>(modi_pr)): # after start-off and in modi_d (s) interval but not during translation
       yaw = np.random.uniform(-np.pi,np.pi) # turn around by this value
       print(f'time={elapsed_time:.4}, new yaw={yaw}')
    '''
    position, velocity, altitude = control.update(direction, yaw, height) # pass new desired state to control and update
    prev_direction = velocity 
    network.update_network(velocity, get_next_state=False) # update grid network with velocity
    position_log.append(position)
    network_state.append(network.network_activity.copy())
# ---------------- end loop ---------------
print(f'Simulation finished at time={elapsed_time/60:.0}minutes\nData:')
print(f' - Position log (shape, [min x, min y], [max x, max y]):{np.shape(position_log)}, {np.min(position_log, axis=0)}, {np.max(position_log, axis=0)}')
print(f' - Network (shape, min, max):{np.shape(network_state)}, {np.min(network_state)}, {np.max(network_state)}')

arena_size = np.sqrt(np.max(np.sum(np.array(position_log)**2, axis=1))) # size = sqrt( sum ( max_over_time(x**2+y**2) ) )
#arena_size = np.sqrt(np.sum((np.max(position, axis=0) - np.min(position, axis=0))**2))
print(f' - Arena size (min, max, value) = {np.min(position, axis=0)}, {np.max(position, axis=0)}, {arena_size}')

# Plot the network activity
print('Generating Images')
network.plot_frame_figure(positions_fig=position_log, network_activity=network_state, num_bins=60, arena_size=arena_size)
print(' - Saved activity plot\ncalculating prediction')

# Predict position
X, y, y_pred, mse_mean, r2_mean = network.fit_linear_model(network_state, position_log)
network.plot_prediction_path(y, y_pred, mse_mean, r2_mean)
print(' - Saved prediction plot')