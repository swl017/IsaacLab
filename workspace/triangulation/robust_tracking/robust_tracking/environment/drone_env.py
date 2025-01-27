"""
@file drone_env.py
@description Environment for drone control tasks.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..simulator.launcher import simulator


class DroneEnv(gym.Env):
    def __init__(self, num_drones=2):
        self.world = simulator.world
        self.timeline = simulator.timeline
        # Group related state variables for better organization and clarity
        self.observation_space = spaces.Dict({
            f'drone_{i}': spaces.Dict({
                # Local state observations
                'orientation': spaces.Box(low=-1.0, high=1.0, shape=(4,)),  # quaternion
                'angular_velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                'linear_velocity': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                'gimbal_orientation': spaces.Box(low=-np.pi, high=np.pi, shape=(3,)),
                
                # Target-related observations
                'target_in_camera': spaces.Box(low=-1.0, high=1.0, shape=(2,)),  # Normalized image coordinates
                'target_visible': spaces.Box(low=0, high=1, shape=(1,)),
                'target_relative_pos': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
                
                # Formation-related observations
                'other_drone_relative': spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            }) for i in range(num_drones)
        })

        # Combine linear velocity, yaw rate, and gimbal commands
        self.action_space = spaces.Box(
            low=np.array([-1.0] * (4 + 2) * num_drones),  # vx, vy, vz, vyaw + gimbal(yaw, pitch)
            high=np.array([1.0] * (4 + 2) * num_drones),
            dtype=np.float32
        )

    def step(self, action):
        simulator.step()