import numpy as np
from controller import Robot
from math import cos, sin

class controller():
    def __init__(self, robot_, FLYING_ATTITUDE = 1):
        self.robot = robot_
        #self.pid_model = pid_model_
        self.pid_model = pid_velocity_fixed_height_controller() 
        timestep = int(self.robot.getBasicTimeStep())
        # Initialize motors
        self.motors = []
        for i in range(4):
            self.motors.append(self.robot.getDevice(f"m{i+1}_motor"))
            self.motors[i].setPosition(float('inf'))
            self.motors[i].setVelocity(-1)

        # Initialize Sensors
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(timestep)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(timestep)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(timestep)

        # Initialize variables
        self.past_x = 0
        self.past_y = 0
        self.past_time = 0
        self.height_desired = FLYING_ATTITUDE

        print('Controller initialised')

    def update(self, direction, yaw_change):
        dt = self.robot.getTime() - self.past_time
        actual_state = {}

        # Get sensor data
        roll = self.imu.getRollPitchYaw()[0]
        pitch = self.imu.getRollPitchYaw()[1]
        yaw = self.imu.getRollPitchYaw()[2]
        yaw_rate = self.gyro.getValues()[2]
        x = self.gps.getValues()[0]
        v_x = (x - self.past_x)/dt
        y = self.gps.getValues()[1]
        v_y = (y - self.past_y)/dt
        altitude = self.gps.getValues()[2]

        # Get body fixed velocities
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x * cos_yaw + v_y * sin_yaw
        v_y = - v_x * sin_yaw + v_y * cos_yaw

        # Initialize values
        desired_state = [0, 0, 0, 0]
        forward_desired, sideways_desired = direction
        #np.clip([forward_desired, sideways_desired], -0.5, 0.5)
        
        #print(f'yaw={yaw}, rate={yaw_rate}')
        yaw_rate_desired = yaw_change # if change is zero, stay at current yaw

        height_diff_desired = 0      
        self.height_desired += height_diff_desired * dt

        # PID velocity controller with fixed height
        motor_power = self.pid_model.pid(dt, forward_desired, sideways_desired,
                                        yaw_rate_desired, self.height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        motor_power[0] *= -1 # propellor 1 and 3 have to be reversed for stable lift
        motor_power[2] *= -1

        for motor, power in zip(self.motors, motor_power):
            motor.setVelocity(power)

        self.past_time = self.robot.getTime()
        self.past_x = x
        self.past_y = y

        return [x, y], [v_x, v_y]

    def get_velocity(self):
        pos = self.gps.getValues()[0:1]
        v_ = (pos - [self.past_x, self.past_y])
        return v_

class pid_velocity_fixed_height_controller():
    def __init__(self):
        self.past_vx_error = 0.0
        self.past_vy_error = 0.0
        self.past_alt_error = 0.0
        self.past_pitch_error = 0.0
        self.past_roll_error = 0.0
        self.altitude_integrator = 0.0
        self.last_time = 0.0
        print('PID initialised')

    def pid(self, dt, desired_vx, desired_vy, desired_yaw_rate, desired_altitude, actual_roll, actual_pitch, actual_yaw_rate,
            actual_altitude, actual_vx, actual_vy):
        # Velocity PID control (converted from Crazyflie c code)
        gains = {"kp_att_y": 1, "kd_att_y": 0.5, "kp_att_rp": 0.5, "kd_att_rp": 0.1,
                 "kp_vel_xy": 2, "kd_vel_xy": 0.5, "kp_z": 10, "ki_z": 5, "kd_z": 5}

        # Velocity PID control
        vx_error = desired_vx - actual_vx
        vx_deriv = (vx_error - self.past_vx_error) / dt
        vy_error = desired_vy - actual_vy
        vy_deriv = (vy_error - self.past_vy_error) / dt
        desired_pitch = gains["kp_vel_xy"] * np.clip(vx_error, -1, 1) + gains["kd_vel_xy"] * vx_deriv
        desired_roll = -gains["kp_vel_xy"] * np.clip(vy_error, -1, 1) - gains["kd_vel_xy"] * vy_deriv
        self.past_vx_error = vx_error
        self.past_vy_error = vy_error

        # Altitude PID control
        alt_error = desired_altitude - actual_altitude
        alt_deriv = (alt_error - self.past_alt_error) / dt
        self.altitude_integrator += alt_error * dt
        alt_command = gains["kp_z"] * alt_error + gains["kd_z"] * alt_deriv + \
            gains["ki_z"] * np.clip(self.altitude_integrator, -2, 2) + 48
        self.past_alt_error = alt_error

        # Attitude PID control
        pitch_error = desired_pitch - actual_pitch
        pitch_deriv = (pitch_error - self.past_pitch_error) / dt
        roll_error = desired_roll - actual_roll
        roll_deriv = (roll_error - self.past_roll_error) / dt
        yaw_rate_error = desired_yaw_rate - actual_yaw_rate
        roll_command = gains["kp_att_rp"] * np.clip(roll_error, -1, 1) + gains["kd_att_rp"] * roll_deriv
        pitch_command = -gains["kp_att_rp"] * np.clip(pitch_error, -1, 1) - gains["kd_att_rp"] * pitch_deriv
        yaw_command = gains["kp_att_y"] * np.clip(yaw_rate_error, -1, 1)
        self.past_pitch_error = pitch_error
        self.past_roll_error = roll_error

        # Motor mixing
        m1 = alt_command - roll_command + pitch_command + yaw_command
        m2 = alt_command - roll_command - pitch_command - yaw_command
        m3 = alt_command + roll_command - pitch_command + yaw_command
        m4 = alt_command + roll_command + pitch_command - yaw_command

        # Limit the motor command
        m1 = np.clip(m1, 0, 600)
        m2 = np.clip(m2, 0, 600)
        m3 = np.clip(m3, 0, 600)
        m4 = np.clip(m4, 0, 600)

        return [m1, m2, m3, m4]
