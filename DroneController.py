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
        self.past_pos = np.array([0, 0])
        self.past_time = 0.0
        self.hover_altitude = hover_altitude
        self.altitude_command = hover_altitude

        print('DroneController initialized')

    def update(self, direction, yaw_rate_command, altitude_adjustment):
        """
        Update the controller based on desired movement and sensor readings.
        
        Args:
            direction (np.array): Desired direction vector (global frame) for movement.
            yaw_rate_command (float): Desired yaw rate.
            altitude_adjustment (float): Change to apply to the desired altitude (currently not used).
        
        Returns:
            tuple: (current position (np.array), global velocity (np.array), altitude (float))
        """
        current_time = self.robot.getTime()
        dt = current_time - self.past_time
        
        # Get sensor data
        roll, pitch, yaw = self.imu.getRollPitchYaw()
        yaw_rate = self.gyro.getValues()[2]
        gps_values = self.gps.getValues()
        pos_global = np.array(gps_values[:2])
        altitude = gps_values[2]

        flipped = (abs(roll) > 90) or (abs(pitch) > 90) # check if drone is flipped upside down
        
        # Compute global velocity based on GPS difference
        global_velocity = (pos_global - self.past_pos) / dt
        
        # Convert global velocity to body-fixed frame
        rotation_matrix = np.array([[cos(yaw), sin(yaw)], 
                                    [-sin(yaw), cos(yaw)]])
        body_velocity = rotation_matrix @ global_velocity
        vx_body, vy_body = body_velocity
        
        # Compute desired velocity in body frame from the global direction
        desired_velocity_body = rotation_matrix @ direction
        desired_forward, desired_side = desired_velocity_body
        
        # Command altitude remains at hover_altitude (altitude_adjustment is not applied)
        self.altitude_command = self.hover_altitude
        
        # Compute motor speeds using the PID controller
        motor_speeds = self.pid_controller.compute(
            dt,
            desired_forward, desired_side, yaw_rate_command, self.altitude_command,
            roll, pitch, yaw_rate,
            altitude, vx_body, vy_body
        )
        
        # Reverse motor outputs for motors 1 and 3 for stable lift
        motor_speeds[0] *= -1
        motor_speeds[2] *= -1
        
        # Reverse action if Drone is flipped
        if (flipped):
            motor_speeds = [-50, 50, 50, -50] # thurst one side up and other down to turn horizontally 
            print('Drone is flipped, trying to stabilise')

        # Set motor speeds
        for motor, speed in zip(self.motors, motor_speeds):
            motor.setVelocity(speed)
        
        # Update state
        self.past_time = current_time
        self.past_pos = pos_global
        
        return pos_global, global_velocity, altitude


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

        print('PIDVelocityController initialized')

    def compute(self, dt, desired_vx, desired_vy, desired_yaw_rate, desired_altitude,
                actual_roll, actual_pitch, actual_yaw_rate, actual_altitude,
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
        # PID gains
        gains = {
            "kp_att_y": 1.0,
            "kd_att_y": 0.5,
            "kp_att_rp": 0.5,
            "kd_att_rp": 0.1,
            "kp_vel_xy": 2.0,
            "kd_vel_xy": 0.5,
            "kp_z": 10.0,
            "ki_z": 5.0,
            "kd_z": 5.0
        }
        
        # Velocity PID for forward (x) and lateral (y)
        vx_error = desired_vx - actual_vx
        vx_deriv = (vx_error - self.prev_vx_error) / dt
        vy_error = desired_vy - actual_vy
        vy_deriv = (vy_error - self.prev_vy_error) / dt
        
        desired_pitch = gains["kp_vel_xy"] * np.clip(vx_error, -1, 1) + gains["kd_vel_xy"] * vx_deriv
        desired_roll = -gains["kp_vel_xy"] * np.clip(vy_error, -1, 1) - gains["kd_vel_xy"] * vy_deriv
        
        self.prev_vx_error = vx_error
        self.prev_vy_error = vy_error
        
        # Altitude PID control
        alt_error = desired_altitude - actual_altitude
        alt_deriv = (alt_error - self.prev_alt_error) / dt
        self.altitude_integrator += alt_error * dt
        alt_command = (gains["kp_z"] * alt_error +
                       gains["kd_z"] * alt_deriv +
                       gains["ki_z"] * np.clip(self.altitude_integrator, -2, 2) + 48) # 48 is the bare minimum velocity for lift off 
        self.prev_alt_error = alt_error
        
        # Attitude PID for pitch, roll, and yaw rate
        pitch_error = desired_pitch - actual_pitch
        pitch_deriv = (pitch_error - self.prev_pitch_error) / dt
        roll_error = desired_roll - actual_roll
        roll_deriv = (roll_error - self.prev_roll_error) / dt
        yaw_rate_error = desired_yaw_rate - actual_yaw_rate
        
        roll_command = gains["kp_att_rp"] * np.clip(roll_error, -1, 1) + gains["kd_att_rp"] * roll_deriv
        pitch_command = -gains["kp_att_rp"] * np.clip(pitch_error, -1, 1) - gains["kd_att_rp"] * pitch_deriv
        yaw_command = gains["kp_att_y"] * np.clip(yaw_rate_error, -1, 1)
        
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