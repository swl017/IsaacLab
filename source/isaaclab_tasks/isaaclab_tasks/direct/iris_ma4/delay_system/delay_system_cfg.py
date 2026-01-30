from isaaclab.utils import configclass

@configclass
class DelaySystemCfg:
    """
    Configuration for the delay system.

    This config defines all timing parameters for realistic sim-to-real delays:
    - Motion filtering (first-order lag)
    - Detection delays (FPS, latency, dropout)
    - Communication delays (ego vs inter-agent)
    """

    # Simulation parameters
    dt_sim: float = 0.01  # Simulation timestep (100 Hz)

    # Motion filter (first-order lag)
    motion_time_constant: float = 0.1  # 100ms time constant
    orientation_time_constant: float = 0.1  # 100ms for orientation
    joint_time_constant: float = 0.05  # 50ms for joints (faster response)

    # Detection sampler
    detection_fps_mean: float = 200.0  # Mean detection rate (Hz)
    detection_fps_std: float = 5.0  # Std deviation of detection rate (Hz)
    detection_latency_mean: float = 0.03  # 30ms mean processing latency
    detection_latency_std: float = 0.005  # 5ms std deviation
    detection_dropout_rate: float = 0.005  # 1% frame dropout
    # Ego communication (local, fast)
    ego_comm_latency: float = 0.005  # 5ms latency (nearly instant)

    # Inter-agent communication
    inter_agent_comm_rate_min: float = 200.0  # Min comm rate (Hz)
    inter_agent_comm_rate_max: float = 400.0  # Max comm rate (Hz) → 30 ± 10 Hz
    inter_agent_comm_latency_mean: float = 0.01  # 100ms mean latency
    inter_agent_comm_latency_std: float = 0.002  # 20ms std deviation
    inter_agent_comm_dropout_rate: float = 0.005  # 5% packet loss