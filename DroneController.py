import numpy as np
from controller import Robot
from numpy import cos, sin

class DroneController:
    """
    Controller for a flying robot (quadcopter) using a fixed-height PID velocity controller.
    
    Attributes:
        robot (Robot): The robot instance.
        pid_controller (PIDVelocityController): Instance of the PID controller.
        motors (list): List of motor devices.
        imu: Inertial measurement unit sensor.
        gps: GPS sensor.
        gyro: Gyroscope sensor.
        past_pos (np.array): Previous (x, y) position from the GPS.
        past_time (float): Timestamp of the last update.
        hover_altitude (float): Desired altitude for hovering.
        altitude_command (float): Current commanded altitude (set to hover_altitude).
    """
    def __init__(self, robot, hover_altitude=1.0):
        """
        Initialize the DroneController.
        
        Args:
            robot (Robot): The robot instance.
            hover_altitude (float): Desired altitude for hovering.
        """
        self.robot = robot
        self.pid_controller = PIDVelocityController()
        timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize motors
        self.motors = []
        for i in range(4):
            motor = self.robot.getDevice(f"m{i+1}_motor")
            motor.setPosition(float('inf'))
            motor.setVelocity(-1)
            self.motors.append(motor)
        
        # Initialize sensors
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(timestep)
        
        # Initialize state variables
        self.past_pos = np.array([0, 0, 0])
        self.past_time = 0.0
        self.hover_altitude = hover_altitude
        self.initial_pid = True

        print('DroneController initialized')

    def change_gains_pid(self, kp = 1.0, kd = 1.4, ki = 0.0):
        self.pid_controller.gains["kp_z"] = kp
        self.pid_controller.gains["kd_z"] = kd
        self.pid_controller.gains["ki_z"] = ki
        self.initial_pid = False

    def update(self, direction): # Override for 3d movement
        """
        Update the controller based on desired movement and sensor readings.
        
        Args:
            direction (np.array): Desired 3d direction vector (global frame) for movement.
        
        Returns:
            tuple: (current position (np.array), global velocity (np.array))
        """
        # Get sensor data
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        gps_values = self.gps.getValues()
        pos_global = np.array(gps_values)
        altitude = gps_values[2]

        # check for 2d or 3d input
        if len(direction) == 3 and direction[2]!=0:            
            horizontal_direction = direction[:2]
            self.hover_altitude = direction[2] + altitude
            #print(f'alt change to = {self.hover_altitude}')
        else:
            horizontal_direction = direction

        desired_altitude = self.hover_altitude

        # Get velocities
        current_time = self.robot.getTime()
        dt = current_time - self.past_time
        rotation_matrix = np.array([[cos(yaw), sin(yaw)],
                                    [-sin(yaw), cos(yaw)]])

        global_velocity = (pos_global - self.past_pos) / dt  
        vx_body, vy_body = rotation_matrix @ global_velocity[:2] # body-centred current velocity
        
        desired_velocity_body = rotation_matrix @ horizontal_direction
        desired_forward, desired_side = desired_velocity_body  # body centred desired velocity        

        # Compute desired yaw based on horizontal direction
        desired_yaw = yaw
        if np.linalg.norm(horizontal_direction) > 0:
            desired_yaw = np.arctan2(horizontal_direction[1], horizontal_direction[0])       

        # Compute motor speeds using the PID controller
        motor_speeds = self.pid_controller.compute(
            dt,
            desired_forward, desired_side, desired_yaw, desired_altitude,
            roll, pitch, yaw, yaw_rate,
            altitude, vx_body, vy_body
        )
        
        # Reverse motor outputs for motors 1 and 3 for stable lift
        motor_speeds[0] *= -1
        motor_speeds[2] *= -1
        
        # Reverse action if Drone is flipped
        flipped = (abs(roll) > 90) or (abs(pitch) > 90) # check if drone is flipped upside down
        if (flipped):
            motor_speeds = [-50, 50, 50, -50] # thurst one side up and other down to turn horizontally 
            print('Drone is flipped, trying to stabilise')

        # Set motor speeds
        for motor, speed in zip(self.motors, motor_speeds):
            motor.setVelocity(speed)
        
        # Update state
        self.past_time = current_time
        self.past_pos = pos_global
        
        return pos_global, global_velocity


class PIDVelocityController:
    """
    PID controller for velocity and altitude control of a quadcopter.
    
    This controller computes the necessary motor commands based on the error in horizontal 
    velocity, altitude, and attitude (roll, pitch, and yaw).
    """
    def __init__(self):
        # Initialize error accumulators and previous error values
        self.prev_vx_error = 0.0
        self.prev_vy_error = 0.0
        self.prev_alt_error = 0.0
        self.prev_pitch_error = 0.0
        self.prev_roll_error = 0.0
        self.altitude_integrator = 0.0
        self.prev_yaw_error = 0.0
        self.yaw_integrator = 0.0
        self.hover_speed = 48 #48

        self.prev_altitude = None
        self.prev_vz_error = 0.0
        self.altitude_integrator_vz = 0.0

        # PID gains
        self.gains = {
            "kp_att_y": 1.0,
            "kd_att_y": 0.5,
            "kp_att_rp": 0.5,
            "kd_att_rp": 0.1,
            "kp_vel_xy": 2.0,
            "kd_vel_xy": 0.5,
            "kp_z": 1.0,
            "ki_z": 0.01,
            "kd_z": 1.4,
            "k_alt": 1.0,    # converts altitude error to desired vertical velocity
            "kp_vz": 1.0,
            "kd_vz": 0.5,
            "ki_vz": 0.01,
            "Kp_yaw" : 2.0,
            "Ki_yaw" : 0.0,
            "Kd_yaw" : 0.5,
        }
        self.clip_bound = 15

        print('PIDVelocityController initialized')

    def compute(self, dt, desired_vx, desired_vy, desired_yaw, desired_altitude,
                actual_roll, actual_pitch, actual_yaw, actual_yaw_rate, actual_altitude,
                actual_vx, actual_vy):
        """
        Compute motor commands using PID control.
        
        Args:
            dt (float): Time step.
            desired_vx (float): Desired forward velocity (body frame).
            desired_vy (float): Desired lateral velocity (body frame).
            desired_yaw_rate (float): Desired yaw rate.
            desired_altitude (float): Desired altitude.
            actual_roll (float): Measured roll angle.
            actual_pitch (float): Measured pitch angle.
            actual_yaw_rate (float): Measured yaw rate.
            actual_altitude (float): Measured altitude.
            actual_vx (float): Measured forward velocity (body frame).
            actual_vy (float): Measured lateral velocity (body frame).
        
        Returns:
            list: Motor speeds [m1, m2, m3, m4].
        """
        
        
        # Velocity PID for forward (x) and lateral (y)
        vx_error = desired_vx - actual_vx
        vx_deriv = (vx_error - self.prev_vx_error) / dt
        vy_error = desired_vy - actual_vy
        vy_deriv = (vy_error - self.prev_vy_error) / dt
        yaw_error = desired_yaw - actual_yaw
        yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi # Wrap to [-pi, pi]
        yaw_deriv = (yaw_error - self.prev_yaw_error) / dt
        self.yaw_integrator += yaw_error * dt
        
        desired_pitch = self.gains["kp_vel_xy"] * np.clip(vx_error, -self.clip_bound, self.clip_bound) + self.gains["kd_vel_xy"] * vx_deriv
        desired_roll = -self.gains["kp_vel_xy"] * np.clip(vy_error, -self.clip_bound, self.clip_bound) - self.gains["kd_vel_xy"] * vy_deriv
        yaw_rate_command = (self.gains["Kp_yaw"] * yaw_error
                        + self.gains["Ki_yaw"] * self.yaw_integrator
                        + self.gains["Kd_yaw"] * yaw_deriv)
    
        self.prev_yaw_error = yaw_error    
        self.prev_vx_error = vx_error
        self.prev_vy_error = vy_error
        
        # Altitude PID control
        alt_error = desired_altitude - actual_altitude
        alt_deriv = (alt_error - self.prev_alt_error) / dt
        self.altitude_integrator += alt_error * dt
        alt_command = (self.gains["kp_z"] * alt_error +
                       self.gains["kd_z"] * alt_deriv +
                       self.gains["ki_z"] * self.altitude_integrator) 
        alt_command = self.hover_speed + (alt_command * 10)  # 48 is the bare minimum velocity for lift off | times a scaled ratio
        self.prev_alt_error = alt_error

        # ----- Velocity-Based Altitude Control -----
        # Compute vertical (z) velocity
        if self.prev_altitude is None:
            # On first call, assume no vertical velocity.
            vertical_velocity = 0.0
        else:
            vertical_velocity = (actual_altitude - self.prev_altitude) / dt
        self.prev_altitude = actual_altitude

        # Compute altitude error and derive a desired vertical velocity
        alt_error = desired_altitude - actual_altitude
        desired_vz = self.gains["k_alt"] * alt_error

        # Compute the velocity error in the vertical axis
        vz_error = desired_vz - vertical_velocity
        vz_deriv = (vz_error - self.prev_vz_error) / dt
        self.altitude_integrator_vz += vz_error * dt
        self.prev_vz_error = vz_error

        # Compute the altitude command adjustment based on vertical velocity error
        vz_command = (self.gains["kp_vz"] * vz_error +
                      self.gains["kd_vz"] * vz_deriv +
                      self.gains["ki_vz"] * self.altitude_integrator_vz)
        
        # Combine with the known hover throttle.
        # 55.35 is the motor speed for stable hovering; we add a scaled correction.
        alt_command = 55.35 + 2*(vz_command)
        #print(f'alt com={alt_command}, vz com = {vz_command}')
        #print(f'alt={actual_altitude}, alt_d={desired_altitude}, alt_e={alt_error}, alt_c={alt_command}')
        
        # Attitude PID for pitch, roll, and yaw rate
        pitch_error = desired_pitch - actual_pitch
        pitch_deriv = (pitch_error - self.prev_pitch_error) / dt
        roll_error = desired_roll - actual_roll
        roll_deriv = (roll_error - self.prev_roll_error) / dt
        yaw_rate_error = yaw_rate_command - actual_yaw_rate
        
        roll_command = self.gains["kp_att_rp"] * np.clip(roll_error, -self.clip_bound, self.clip_bound) + self.gains["kd_att_rp"] * roll_deriv
        pitch_command = -self.gains["kp_att_rp"] * np.clip(pitch_error, -self.clip_bound, self.clip_bound) - self.gains["kd_att_rp"] * pitch_deriv
        yaw_command = self.gains["kp_att_y"] * np.clip(yaw_rate_error, -self.clip_bound, self.clip_bound)
        
        self.prev_pitch_error = pitch_error
        self.prev_roll_error = roll_error
        
        # Motor mixing to compute individual motor speeds
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command
        
        # Limit motor commands to acceptable range
        m1 = np.clip(m1, 0, 600)
        m2 = np.clip(m2, 0, 600)
        m3 = np.clip(m3, 0, 600)
        m4 = np.clip(m4, 0, 600)
        
        return [m1, m2, m3, m4]