import numpy as np
from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller
from math import cos, sin

class controller():
    def __init__(self, robot_, pid_model_, FLYING_ATTITUDE = 1):
        self.robot = robot_
        self.pid_model = pid_model_
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
        self.first_time = False
        self.height_desired = FLYING_ATTITUDE

        print('Controller initialised')

    def update(self, direction, yaw_change):
        dt = self.robot.getTime() - self.past_time
        actual_state = {}

        if False:
            past_x = self.gps.getValues()[0]
            past_y = self.gps.getValues()[1]
            past_time = self.robot.getTime()
            first_time = False

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
        
        yaw_desired = yaw + yaw_change # if change is zero, stay at current yaw

        height_diff_desired = 0      
        self.height_desired += height_diff_desired * dt

        # PID velocity controller with fixed height
        motor_power = self.pid_model.pid(dt, forward_desired, sideways_desired,
                                        yaw_desired, self.height_desired,
                                        roll, pitch, yaw_rate,
                                        altitude, v_x, v_y)

        motor_power[0] *= -1 # propellor 1 and 3 have to be reversed for stable lift
        motor_power[2] *= -1

        for motor, power in zip(self.motors, motor_power):
            motor.setVelocity(power)

        self.past_time = self.robot.getTime()
        self.past_x = x
        self.past_y = y

        return motor_power, [v_x, v_y]

    def get_velocity(self):
        pos = self.gps.getValues()[0:1]
        v_ = (pos - [self.past_x, self.past_y])
        return v_