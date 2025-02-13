import numpy as np
from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller
from CFController import controller


FLYING_ATTITUDE = 1
robot = Robot()
timestep = int(robot.getBasicTimeStep())
PID_crazyflie = pid_velocity_fixed_height_controller() # Crazyflie velocity PID controller
control = controller(robot, PID_crazyflie, FLYING_ATTITUDE)
elapsed_time = 0

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
        new_angle = np.random.uniform(0,2*np.pi)

    sigma = (1 - bias) * np.pi  # Standard deviation for the Gaussian spread
    new_angle = np.random.normal(base_angle, sigma) % (2 * np.pi) # Sample a new angle from a Gaussian distribution centered around base_angle
    new_vector = np.array([np.cos(new_angle), np.sin(new_angle)]) * size # Convert polar to Cartesian coordinates
    return new_vector

initial_pause = 6 #s
modi = 2 #s
modi_pr = 0.032 #s
size = 0.05 # magnitude of movement vector
bias = 0.15 # randomness of movement 

prev_direction = np.array([0, 0])
direction = prev_direction
# Main loop:
while robot.step(timestep) != -1:
    elapsed_time += (timestep/1000) # ms to s
    if (elapsed_time>=initial_pause and elapsed_time%modi<=modi_pr):
        direction = generate_biased_vector(prev_direction, size, bias)
        print(f'time={elapsed_time} new direction')
    cmds, velocity = control.update(direction)
    prev_direction = velocity #control.get_velocity()
    #print(f'time={elapsed_time}')#, planned direction={direction}, actual direction={velocity}')