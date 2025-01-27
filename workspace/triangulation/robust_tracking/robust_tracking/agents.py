
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
from robust_tracking.ros2.ros2_node import ROS2Node
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from vision_msgs.msg import Detection2DArray
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self, num_agents, ros2_node: ROS2Node):
        # Each drone's observation includes:
        # - Own orientation (3D), velocity (6D) | topic: local_odom
        # - gimbal orientation (3D) | topic: gimbal_orientation
        # - Target position in camera frame if visible (2D) | topic: yolo_detection2darray
        # - Target visibility flag (1D) | topic: yolo_detection2darray
        # - Target drone's relative position (3D) | topic: target_odom_gt & local_odom
        # - Other drone's relative position (3D) | topic: relative_odom
        self.ros2_node = ros2_node
        self.num_ego = num_agents
        self.observation_space = spaces.Dict({
            f'drone{i}': spaces.Dict({
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
            }) for i in range(num_agents)
        })
        # Velocity commands in world frame, gimbal orientation commands
        """
            [vx1, vy1, vz1, vyaw1, yaw1, pitch1, 
             vx2, vy2, vz2, vyaw2, yaw2, pitch2]
        """
        # Combine linear velocity, yaw rate, and gimbal commands
        self.action_space = spaces.Box(
            low=np.array([-1.0] * (4 + 2) * num_agents),  # vx, vy, vz, vyaw + gimbal(yaw, pitch)
            high=np.array([1.0] * (4 + 2) * num_agents),
            dtype=np.float32
        )
        self.last_action = None

        class Weight:
            def __init__(self):
                self.tri = 1.0
                self.det = 1.0
                self.dist_to_target = 1.0
                self.dist_to_self = 1.0
                self.smoothness = 0.3
                self.tri_acc = 1.0
        self.w = Weight()
        self.min_distance = 1.0
        self.max_invisible_steps = 10
        self.invisible_steps = 0
        self.required_success_steps = 10
        self.tracking_success_steps = 0
        self.last_observation = None
        class NormFactor:
            def __init__(self):
                self.velocity = 10.0
                self.angular_velocity = 3.0
                self.gimbal_orientation = np.pi
        self.norm_factor = NormFactor()


    def _get_obs(self):
        """Process raw sensor data into normalized observations."""
        observations = {}
        
        for i in range(self.num_ego):
            # Get latest sensor data
            odom = self.ros2_node.ego_agents[i].odom
            gimbal = self.ros2_node.ego_agents[i].gimbal_state
            detection = self.ros2_node.ego_agents[i].detection
            
            if odom is None or gimbal is None:
                # Handle missing data
                return self.last_observation
            
            # Process into normalized observations
            obs = {
                'orientation': np.array([
                    odom.pose.pose.orientation.x,
                    odom.pose.pose.orientation.y,
                    odom.pose.pose.orientation.z,
                    odom.pose.pose.orientation.w,
                ]),
                'angular_velocity': np.array([
                    odom.twist.twist.angular.x / self.norm_factor.angular_velocity,
                    odom.twist.twist.angular.y / self.norm_factor.angular_velocity,
                    odom.twist.twist.angular.z / self.norm_factor.angular_velocity,
                ]),
                'linear_velocity': np.array([
                    odom.twist.twist.linear.x / self.norm_factor,
                    odom.twist.twist.linear.y / self.norm_factor,
                    odom.twist.twist.linear.z / self.norm_factor,
                ]),
                'gimbal_orientation': np.array([
                    gimbal.x / self.norm_factor.gimbal_orientation,  # roll
                    gimbal.y / self.norm_factor.gimbal_orientation,  # pitch
                    gimbal.z / self.norm_factor.gimbal_orientation,  # yaw
                ]),
            }
            
            # Process target detection if available
            if detection is not None and len(detection.detections) > 0:
                target_det = detection.detections[0]  # Assume first detection is target
                obs['target_in_camera'] = np.array([
                    target_det.bbox.center.x / self.ros2_node.ego_agents[i].camera_info.width,  # Normalized x (-1 to 1)
                    target_det.bbox.center.y / self.ros2_node.ego_agents[i].camera_info.height,  # Normalized y (-1 to 1)
                ])
                obs['target_visible'] = np.array([1.0])
            else:
                obs['target_in_camera'] = np.zeros(2)
                obs['target_visible'] = np.array([0.0])
            
            # Calculate relative positions
            target_pos = self.ros2_node.target_agents[0].odom.pose.pose.position
            ego_pos = odom.pose.pose.position
            other_ego_pos = self.ros2_node.ego_agents[1-i].odom.pose.pose.position
            
            obs['target_relative_pos'] = np.array([
                target_pos.x - ego_pos.x,
                target_pos.y - ego_pos.y,
                target_pos.z - ego_pos.z,
            ])
            
            obs['other_drone_relative'] = np.array([
                other_ego_pos.x - ego_pos.x,
                other_ego_pos.y - ego_pos.y,
                other_ego_pos.z - ego_pos.z,
            ])
            
            observations[f'drone{i}'] = obs
        
        self.last_observation = observations
        return observations

    def step(self, action):
        """Execute action and return new state."""
        # Store action for smoothness reward
        self.last_action = action
        
        # Split action into commands for each drone
        action_size = 6  # 4 velocity + 2 gimbal
        for i in range(self.num_ego):
            drone_action = action[i*action_size:(i+1)*action_size]
            
            # Create velocity command
            cmd_vel = TwistStamped()
            cmd_vel.header.stamp = self.ros2_node.node.get_clock().now().to_msg()
            cmd_vel.twist.linear.x = drone_action[0] * 10.0
            cmd_vel.twist.linear.y = drone_action[1] * 10.0
            cmd_vel.twist.linear.z = drone_action[2] * 10.0
            cmd_vel.twist.angular.z = drone_action[3] * np.pi  # yaw rate
            
            # Create gimbal command
            gimbal_cmd = Vector3()
            gimbal_cmd.y = drone_action[4] * 180.0  # pitch
            gimbal_cmd.z = drone_action[5] * 180.0  # yaw
            
            # Set commands
            self.ros2_node.ego_agents[i].set_cmd_vel(cmd_vel)
            self.ros2_node.ego_agents[i].set_gimbal_command(gimbal_cmd)
        
        # Publish all commands
        self.ros2_node.timer_callback()
        
        # Get new state
        observation = self._get_obs()
        reward = self.compute_reward()
        terminated = self._check_collision  # Define your termination conditions
        truncated = self._check_tracking_lost()  # Define your truncation conditions
        info = {}

        self.ros2_node.update_last_states()
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        # Wait for all drones to reach initial positions
        self.ros2_node.is_node_ready()
        
        # Get initial observation
        observation = self._get_obs()
        info = {}
        
        return observation, info

    def close(self):
        """Cleanup ROS node."""
        self.ros2_node.node.destroy_node()

    def compute_reward(self):
        reward = 0.0

        # Triangulation reward (w[0])
        reward += self.w.tri * 1 if self.ros2_node.is_triangulation_updated() else 0

        # Triangulation accuracy reward (w[0])
        reward += self.w.tri_acc * self.ros2_node.calculate_distance(self.ros2_node.triangulation, self.ros2_node.target_odom_gt) * 1 if self.ros2_node.is_triangulation_updated() else 0

        for i in range(self.num_ego):
            # Detection reward (w[1]])
            # reward += self.w.det * self.is_target_detected2d(drone_id)
            # Distance to target penalty (w[2])
            reward -= self.w.dist_to_target * self._barrier_function(thres=1.0, value=self.ros2_node.calculate_distance(self.ros2_node.drones[i].odom, self.ros2_node.triangulation)) * 1 if self.ros2_node.is_triangulation_updated() else 0

        
        # Distance to other drone penalty (w[3])
        reward -= self.w.dist_to_self * self._barrier_function(thres=5.0, value=self.ros2_node.calculate_distance(self.drones[1].odom.position, self.drones[0].odom.position))
                
        # Smoothness penalty (w4 = 0.3)
        action_norm = np.linalg.norm(self.last_action)
        smoothness_penalty = self.w.smoothness * action_norm
        reward -= smoothness_penalty
        
        return reward
    
    def _barrier_function(self, thres, value):
        return -np.log(value / thres)
    
    def _check_collision(self):
        """Check if drones are too close"""
        for i in range(self.num_ego):
            for j in range(i+1, self.num_ego):
                dist = self.ros2_node.calculate_distance(self.ros2_node.drones[i].odom, self.ros2_node.drones[j].odom)
                if dist < self.min_distance:
                    return True
        return False

    def _check_tracking_lost(self):
        """Check if target was invisible for too long"""
        return True if self.ros2_node.get_message_age(self.ros2_node.triangulation) > self.max_invisible_steps else False

    # def _check_tracking_success(self):
    #     """Check if target was tracked successfully"""
    #     return self.tracking_success_steps > self.required_success_steps

####################################################################################################
### @todo normalize
### @todo truncate
### @todo write functions in reward function