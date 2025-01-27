
import gymnasium as gym
from gymnasium import spaces

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
import numpy as np

class ROS2Node(Node):
    def __init__(self):
        try:
            rclpy.init()
        except:
            # If rclpy is already initialized, just ignore the exception
            pass        
        self.node = rclpy.create_node("rl_node")
        qos_profile_trainsient = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_profile_durability = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        class ROSAgent:
            def __init__(self, node):
                self.node = node
                self.cmd_vel_pub = None
                self.gimbal_command_pub = None
                self.gimbal_command = None
                self.gimbal_state_sub = None
                self.gimbal_state = None
                self.odom = None
            def gimbal_state_sub_callback(self, msg):
                self.gimbal_state = msg
            def odom_sub_callback(self, msg):
                self.odom = msg
            def set_gimbal_command(self, gimbal_command):
                self.gimbal_command = gimbal_command
            def set_cmd_vel(self, cmd_vel):
                self.cmd_vel = cmd_vel
            def publish_commands(self):
                if self.cmd_vel_pub is not None:
                    self.cmd_vel_pub.publish(self.cmd_vel if self.cmd_vel is not None else TwistStamped())
                if self.gimbal_command_pub is not None:
                    self.gimbal_command_pub.publish(self.gimbal_command if self.cmd_vel is not None else TwistStamped())
                
        num_ego_agents = 2
        num_target_agents = 1
        self.drones = []
        for i in range(1, num_ego_agents + num_target_agents + 2):
            self.drones[i] = ROSAgent(self.node)
            self.drones[i].cmd_vel_pub = self.node.create_publisher(TwistStamped, f"/px4_{i}/cmd_vel", qos_profile_trainsient)
            self.drones[i].gimbal_command_pub = self.node.create_publisher(TwistStamped, f"/px4_{i}/gimbal_command_rpy_deg", qos_profile_trainsient)
            self.drones[i].gimbal_state_sub = self.node.create_subscription(TwistStamped, f"/px4_{i}/gimbal_state_rpy_rad", self.drones[i].gimbal_state_sub_callback, qos_profile_durability)
            self.drones[i].odom_sub = self.node.create_subscription(Odometry, f"/px4_{i}/local_odom", self.drones[i].odom_sub_callback, qos_profile_durability)

        self.ego_agents = self.drones[:num_ego_agents]
        self.target_agents = self.drones[num_ego_agents:num_ego_agents+num_target_agents]

        self.timer = self.node.create_timer(10.0, self.timer_callback)
        self.triangulation_sub = self.node.create_subscription(Odometry, "/target/odom/triangulation", self.sub_callback, 10)

    def timer_callback(self):
        for drone in self.drones:
            drone.publish_commands()

    def sub_callback(self, msg):
        carb.log_warn(msg.data)

    def spin_once(self):
        rclpy.spin_once(self.node, timeout_sec=0.1)
        
class ROS2SpinCallback(BaseCallback):
    def __init__(self, node):
        super().__init__()
        self.node = node

    def _on_step(self):
        try:
            self.node.spin_once()
            return True
        except Exception as e:
            print(e)
            return False
        
class DroneEnv():
    def __init__(self):
        # Each drone's observation includes:
        # - Own orientation (3D), velocity (6D) | topic: local_odom
        # - gimbal orientation (3D) | topic: gimbal_orientation
        # - Target position in camera frame if visible (2D) | topic: yolo_detection2darray
        # - Target visibility flag (1D) | topic: yolo_detection2darray
        # - Target drone's relative position (3D) | topic: target_odom_gt & local_odom
        # - Other drone's relative position (3D) | topic: relative_odom
        self.observation_space = spaces.Dict({
            'drone1': spaces.Box(low=-np.inf, high=np.inf, shape=(12,)),
            'drone2': spaces.Box(low=-np.inf, high=np.inf, shape=(12,))
        })
        
        # Velocity commands in world frame, gimbal orientation commands
        """
            [vx1, vy1, vz1, vyaw1, yaw1, pitch1, 
             vx2, vy2, vz2, vyaw2, yaw2, pitch2]
        """
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,)  
        )

        class Weight:
            def __init__(self):
                self.tri = 1.0
                self.det = 1.0
                self.dist_to_target = 1.0
                self.dist_to_self = 1.0
                self.smoothness = 0.3
                self.tri_acc = 1.0
        self.w = Weight()

    def compute_reward(self):
        reward = 0.0

        # Triangulation reward (w[0])
        # reward += w[0] * self.is_triangulation_updated()

        # Triangulation accuracy reward (w[0])
        reward += self.w.tri_acc * self.is_triangulation_updated() * np.norm(self.triangulation.position - self.target_odom_gt.position)

        for drone_id in [1, 2]:
            # Detection reward (w[1]])
            reward += self.w.det * self.is_target_detected2d(drone_id)
            # Distance to target penalty (w[2])
            reward -= self.w.dist_to_target * self.is_triangulation_updated() * self.barrier_function(thres=1.0, value=np.norm(self.drones[i].odom.position - self.triangulation.position))

        
        # Distance to other drone penalty (w[3])
        reward -= self.w.dist_to_self * self.barrier_function(thres=5.0, value=np.norm(self.drones[1].odom.position - self.drones[2].odom.position))
                
        # Smoothness penalty (w4 = 0.3)
        action_norm = np.linalg.norm(self.last_action)
        smoothness_penalty = self.w.smoothness * action_norm
        reward -= smoothness_penalty
        
        return reward