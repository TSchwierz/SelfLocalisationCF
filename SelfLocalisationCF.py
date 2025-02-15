import numpy as np
from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller
from CFController import controller
from GridNetwork import GridNetwork as gn

FLYING_ATTITUDE = 1
robot = Robot()
timestep = int(robot.getBasicTimeStep())
PID_crazyflie = pid_velocity_fixed_height_controller() # Crazyflie velocity PID controller
control = controller(robot, PID_crazyflie, FLYING_ATTITUDE)
network = gn(9,10)
elapsed_time = 0
position_log = []

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

initial_pause = 6 #(in s) amount of time at the start for the drone to lift off and stabilise
modi = 2 #(in s) the interval with which new directionak commands should be given
modi_pr = 0.032 #(in s) setting this to 32ms or equal to the robot timestep ensures only one new command per interval
size = 0.1 # magnitude of movement vector
bias = -1 # randomness of movement 

prev_direction = np.array([0, 0])
direction = prev_direction
yaw = 0
network_state = 0

# Main loop:
while robot.step(timestep) != -1 and elapsed_time < 60*60: #1h #:
    elapsed_time += (timestep/1000) # ms to s
    direction=[0,0]
    yaw = 0
    if (elapsed_time>=initial_pause and elapsed_time%modi<=modi_pr):
        direction = generate_biased_vector(prev_direction, size, bias)
        print(f'time={elapsed_time}direction={direction} new direction angle {np.arctan2(direction[1], direction[0])/(0.5*np.pi):.2}')
    if (elapsed_time>=initial_pause and elapsed_time%3<=(modi_pr) and elapsed_time%2>(modi_pr)):
       yaw = np.random.uniform(0,np.pi)
       print(f'new yaw={yaw}')
    position, velocity = control.update(direction, yaw)
    prev_direction = velocity 
    network_state = network.update_network(velocity, get_next_state=True)
    position_log.append(position)
    #print(f'time={elapsed_time}')#, planned direction={direction}, actual direction={velocity}')

print(f'Simulation finished at time={elapsed_time}\nGenerating image')
network.plot_frame_figure(positions_fig=position_log, network_activity=network_state, num_bins=60)
