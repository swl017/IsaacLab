from stable_baselines3.common.callbacks import BaseCallback
import carb
# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray
import numpy as np

class ROS2Node(Node):
    def __init__(self, num_ego=2, num_target=1):
        try:
            rclpy.init()
        except:
            # If rclpy is already initialized, just ignore the exception
            pass        
        # self.node = rclpy.create_node("rl_node")
        super().__init__("rl_node")

        self.rate = self.create_rate(10)

        # Create more reliable QoS profile for critical messages
        self.qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Less strict QoS for high-frequency data
        self.qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        class ROSAgent:
            def __init__(self, id=0):
                self.id = id
                self.cmd_vel_pub = None
                self.cmd_vel = None
                self.gimbal_command_pub = None
                self.gimbal_command = None
                self.gimbal_state_sub = None
                self.gimbal_state = None
                self.last_gimbal_state = None
                self.odom = None
                self.last_odom = None
                self.waypoint = None
                self.last_waypoint = None
                self.detection = None
                self.last_detection = None
                self.camera_info_sub = None
                self.camera_info = None
                self.is_position_ready = False

            def gimbal_state_sub_callback(self, msg):
                self.gimbal_state = msg
            def odom_sub_callback(self, msg):
                self.odom = msg
                carb.log_warn(f"Odom {self.id}: {self.odom.pose.pose.position.x}")
            def detection_sub_callback(self, msg):
                self.detection = msg
            def camera_info_sub_callback(self, msg):
                self.camera_info = msg
            def waypoint_sub_callback(self, msg):
                self.waypoint = msg

            def set_gimbal_command(self, gimbal_command):
                self.gimbal_command = gimbal_command
            def set_cmd_vel(self, cmd_vel):
                self.cmd_vel = cmd_vel

            def is_gimbal_state_updated(self):
                if self.gimbal_state is not None and self.last_gimbal_state is not None:
                    return True if self.gimbal_state != self.last_gimbal_state else False
            def is_detection_updated(self):
                if self.detection is not None and self.last_detection is not None:
                    return True if self.detection != self.last_detection else False
            def is_odom_updated(self):
                if self.odom is not None and self.last_odom is not None:
                    return True if self.odom != self.last_odom else False
                
            def publish_commands(self):
                if self.cmd_vel_pub is not None:
                    self.cmd_vel_pub.publish(self.cmd_vel if self.cmd_vel is not None else TwistStamped())
                if self.gimbal_command_pub is not None:
                    self.gimbal_command_pub.publish(self.gimbal_command if self.cmd_vel is not None else Vector3())
            def is_ready(self, i=0):
                # carb.log_warn(f"Drone: {i}")
                # carb.log_warn(f"Odom: {self.odom.pose.pose.position.x if self.odom is not None else -1}")
                # carb.log_warn(f"Waypoint: {self.waypoint.pose.pose.position.x if self.waypoint is not None else -1}")
                if self.odom is not None and self.waypoint is not None:
                    dist = np.linalg.norm(np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z]) - np.array([self.waypoint.pose.pose.position.x, self.waypoint.pose.pose.position.y, self.waypoint.pose.pose.position.z]))
                    yaw_error_deg = np.rad2deg(np.arccos(2 * (self.odom.pose.pose.orientation.w**2) - 1)) - np.rad2deg(np.arccos(2 * (self.waypoint.pose.pose.orientation.w**2) - 1))
                    self.is_position_ready = True if dist < 5.0 else False
                return self.is_position_ready
            def update_last_states(self):
                self.last_gimbal_state = self.gimbal_state
                self.last_odom = self.odom
                self.last_waypoint = self.waypoint
                self.last_detection = self.detection

        self.num_ego_agents = num_ego
        self.num_target_agents = num_target
        self.num_agents = self.num_ego_agents + self.num_target_agents
        self.drones = []
        for i in range(self.num_agents):
            self.drones += [ROSAgent(i)]
            self.drones[i].cmd_vel_pub = self.create_publisher(TwistStamped, f"/px4_{i+1}/cmd_vel", self.qos_reliable)
            self.drones[i].gimbal_command_pub = self.create_publisher(Vector3, f"/px4_{i+1}/gimbal_command_rpy_deg", self.qos_reliable)
            self.drones[i].gimbal_state_sub = self.create_subscription(Vector3, f"/px4_{i+1}/gimbal_state_rpy_rad", self.drones[i].gimbal_state_sub_callback, self.qos_sensor)
            self.drones[i].odom_sub = self.create_subscription(Odometry, f"/px4_{i+1}/local_odom", self.drones[i].odom_sub_callback, self.qos_sensor)
            self.drones[i].waypoint_sub = self.create_subscription(Odometry, f"/px4_{i+1}/initial_waypoint", self.drones[i].waypoint_sub_callback, self.qos_sensor)
            self.drones[i].detection_sub = self.create_subscription(Detection2DArray, f"/px4_{i+1}/yolo_result_vision", self.drones[i].detection_sub_callback, self.qos_sensor)
            self.drones[i].camera_info_sub = self.create_subscription(CameraInfo, f"/px4_{i+1}/camera/color/camera_info", self.drones[i].camera_info_sub_callback, self.qos_sensor)

        self.ego_agents = self.drones[:self.num_ego_agents]
        self.target_agents = self.drones[self.num_ego_agents:self.num_agents]

        # self.timer = self.create_timer(10.0, self.timer_callback)
        self.triangulation_sub = self.create_subscription(Odometry, "/target/odom/triangulation", self.triangulation_sub_callback, self.qos_sensor)
        self.triangulation = None
        self.clock_sub = self.create_subscription(Clock, "/clock", self.clock_sub_callback, self.qos_sensor)
        self.clock = None
        self.last_clock = Clock()
        self.last_triangulation = None
        self.target_odom_gt = None
        self.last_target_odom_gt = None

    def timer_callback(self):
        for drone in self.drones:
            drone.publish_commands()

    def clock_sub_callback(self, msg):
        self.clock = msg
        carb.log_warn(f"Clock: {self.clock.clock.sec + self.clock.clock.nanosec * 1e-9 if self.clock is not None else -1}")

    def get_message_age(self, msg):
        if msg is None or msg.header.stamp is None:
            return float('inf')
        msg_time = Time.from_msg(msg.header.stamp)
        current_time = self.get_clock().now()
        return (current_time - msg_time).nanoseconds / 1e9
    
    def triangulation_sub_callback(self, msg):
        self.triangulation = msg
        carb.log_warn(f"triangulation: {self.triangulation.pose.pose.position.x}")

    def get_target_odom_gt(self):
        return self.drones[-1].odom

    def update_last_states(self):
        for drone in self.drones:
            drone.update_last_states()
        self.last_triangulation = self.triangulation
        self.last_clock = self.clock

    def is_triangulation_updated(self):
        return (self.triangulation is not None and 
                        self.last_triangulation is not None and 
                        self.triangulation != self.last_triangulation)
    
    def calculate_distance(self, odom1, odom2):
        return np.linalg.norm(np.array([odom1.pose.pose.position.x, odom1.pose.pose.position.y, odom1.pose.pose.position.z]) - np.array([odom2.pose.pose.position.x, odom2.pose.pose.position.y, odom2.pose.pose.position.z]))

    def spin(self):
        rclpy.spin()

    def spin_once(self):
        rclpy.spin_once(self)

    def is_node_ready(self):
        carb.log_warn("Checking if node is ready")
        # carb.log_warn(f"Curr time: {self.get_clock().now()}")
        # carb.log_warn(f"Clock: {self.clock.clock.sec + self.clock.clock.nanosec * 1e-9 if self.clock is not None else -1}")
        ready_drone = False
        ready_triangulation = False
        for i in range(len(self.drones)):
            if not self.drones[i].is_ready(i):
                pass
            else:
                ready_drone = True
        ready_triangulation = True if self.triangulation is not None else False
        return ready_drone and ready_triangulation


    def ok(self):
        return rclpy.ok()

    def shutdown(self):
        self.destroy_node()
        rclpy.shutdown()
        
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
        
class ISAACStepCallback(BaseCallback):
    def __init__(self, world):
        super().__init__()
        self.world = world

    def _on_step(self):
        try:
            self.world.step(render=True)
            return True
        except Exception as e:
            print(e)
            return False  