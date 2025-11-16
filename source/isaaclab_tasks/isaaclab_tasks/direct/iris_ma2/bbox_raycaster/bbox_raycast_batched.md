# Creating a Bounding Box Raycast Module with Batched Operations
Since create_bbox_cache from isaacsim.core.utils.bound doesn't support batched operations, I am building a batched (axis-aligned) bounding box extractor that accounts for occlusions that doesn't require querying from image data or transfering it between CPU and GPU for my parallel multi agent RL environment. Each agent(drone) is equipped with a gimbaled zoom camera.

## Parameters and dimensions
- Number of parallel envs: N
- Number of targets per env: T
- Number of agents("camera") per env: C
- Target pose (world): [N, T, 7] # position + orientation
- Camera pose (world): [N, C, 7]
- Camera parameters: [N, C, 7] # fx, fy, cx, cy, zoom, image width, image height can all differ for all cameras (Can we make it more compact or not?)

## Extracting the bounding boxes
- Get the target(s)'s 3D bounding box from create_bbox_cache from isaacsim.core.utils.bounds only once when we setup the scene (world frame).
- Initial 3D corners (create_bbox_cache output): min_x, min_y, min_z, max_z, max_y, max_z
- Transform the 3D vertices to the target's pose at the time step using the target's pose
- Project the transformed 3D vertices to each agent's camera image plane.
- Invalidate for any partial detections (any corners out of FOV) or occlusion or bboxes with too small or too large sizes relative to image width and height.
- Input
    - Target pose
    - Camera pose
    - Camera parameters
    - Initial 3D corners of the target
- Output
    - bboxes: [N, C, T, 4] # bbox center x, y, width, height in pixels
    - bboxes_normalized: [N, C, T, 4] # bbox center x, y, width, height devided by image width and height
    - valid_mask: [N, C, T]

## Implementing occlusions
- Select 9 points from the bbox(4 corners, 4 middle-of-corners, 1 center) and cast a ray from the camera origin to those points.
- Use RTX/PhysX raycaster to check for any collisions other than the target.
- Use IsaacLab raycaster code.

## Edge cases and solutions
- Division by Zero - Depth Normalization
    - Problem: When projecting points, dividing by depth (Z) can fail if point is at camera origin.
    - Solution: 
        ```
        # Mark points behind camera as invalid
        behind_camera = z.squeeze(-1) <= 0
        ```
- Points Behind Camera
    - Problem: Points with negative Z in camera frame project to incorrect pixel coordinates.
    - Solution: Same as depth normalization
- Degenerate Bounding Boxes. 
    - Problem: When all corners project to nearly the same pixel, bbox becomes degenerate (zero area).
    - Solution: Filter out using minimum bbox size
- Gimbal lock
    - Problem: Extreme pitch angles (~±90°) can cause numerical instability in transformation matrices.
    - Solution: Check if z-axis (forward) is nearly vertical

## Code optimization (Memory management)
- Pre-allocate all buffers in __init__ to avoid dynamic allocation
    - Allocate all tensors once at initialization with maximum expected size.
- Use tensor indexing, .view(), .reshape(), and .unsqueeze() which create views sharing the same underlying data.
    - .view(), .reshape() (when contiguous)
    - .unsqueeze(), .squeeze()
    - .expand() (creates virtual copies)
    - .transpose(), .permute() (changes stride)
    - Slicing: tensor[..., :3]
- Use pre-allocated buffers for intermediate tensors if possible.


## IsaacLab integration
- See attached files.
- IsaacLab 2.1.0
- IsaacSim 4.5.0

## Details for my setup
Here are some of my env configs
```
class IrisMAEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 30.0
    decimation = 2
    
    # Define agents - The environment is now built from this list
    possible_agents = ["drone_0", "drone_1"]
    
    # Define spaces for each agent
    action_spaces = {
        "drone_0": 7,  # [vx, vy, vz, yaw_rate, gimbal_yaw_rate, gimbal_pitch_rate, zoom_rate]
        "drone_1": 7,
    }
    observation_spaces = {
        "drone_0": 27,  # Added formation-related observations
        "drone_1": 27,
    }
    state_space = -1  # Concatenate all observations
    
    debug_vis = True
    ui_window_class_type = IrisMAEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=2 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Camera configuration template (dynamically applied to each agent)
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/{robot_name}/pitch_link/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "semantic_segmentation"],
        colorize_semantic_segmentation=True,
        semantic_segmentation_mapping={
            "class:target": (255, 36, 66, 255),
            "class:robot": (255, 36, 255, 255),
        },
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955, 
            clipping_range=(0.1, 1.0e5)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.5, -0.5, 0.5, -0.5), 
            convention="ros"
        ),
    )

    # Target configuration (shared between agents)
    target_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/target",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
                enable_gyroscopic_forces=False,
                rigid_body_enabled=True,
            ),
            copy_from_source=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(5.0, 0.0, 3.5), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=15.0, replicate_physics=True)
    
    # Robot configuration template (dynamically applied to each agent)
    robot: ArticulationCfg = IRIS_GIMBAL2_CFG.replace(prim_path="/World/envs/env_.*/{robot_name}")

...
```
```
class IrisMAEnv(DirectMARLEnv):
    cfg: IrisMAEnvCfg

    def __init__(self, cfg: IrisMAEnvCfg, render_mode: str | None = None, **kwargs):
        # -- Dynamically generate agent-specific robot and camera configs --
        # This must be done BEFORE super().__init__ because _setup_scene is called from there.
        self.agent_robot_cfgs = {}
        self.agent_camera_cfgs = {}
        for agent_id in cfg.possible_agents:
            robot_index = agent_id.split("_")[-1]
            robot_name = f"Robot_{robot_index}"

            # Create and store robot config
            robot_cfg = copy.deepcopy(cfg.robot)
            robot_cfg.prim_path = robot_cfg.prim_path.format(robot_name=robot_name)
            self.agent_robot_cfgs[agent_id] = robot_cfg
            
            # Create and store camera config
            camera_cfg = copy.deepcopy(cfg.camera)
            camera_cfg.prim_path = camera_cfg.prim_path.format(robot_name=robot_name)
            self.agent_camera_cfgs[agent_id] = camera_cfg
...

        # Camera tracking for each agent
        self.camera_pos_world = {
            agent: torch.zeros(self.num_envs, 3, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.camera_quat_world = {
            agent: torch.zeros(self.num_envs, 4, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.bboxes = {
            agent: torch.zeros(self.num_envs, 4, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.bboxes_normalized = {
            agent: torch.zeros(self.num_envs, 4, device=self.device)
            for agent in self.cfg.possible_agents
        }
        self.bbox_valid_mask = {
            agent: torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
            for agent in self.cfg.possible_agents
        }
        
        # Gimbal tracking
        self.gimbal_dof_targets = {
            agent: torch.zeros(self.num_envs, self._robots[agent].num_joints, device=self.device)
            for agent in self.cfg.possible_agents
        }
        
        # Zoom tracking
        self.zoom_level = {
            agent: torch.ones(self.num_envs, device=self.device) * 1.0
            for agent in self.cfg.possible_agents
        }
        self.base_focal_length = self.cfg.camera.spawn.focal_length
        
        # Camera configuration batch for each agent
        self.camera_cfg_batch = {
            agent_id: create_camera_cfg_tensor(agent_cfg, self.num_envs, device=self.device)
            for agent_id, agent_cfg in self.agent_camera_cfgs.items()
        }
        camera_offset_rot_single = torch.tensor(cfg.camera.offset.rot, device=self.device)
        self.camera_offset_rot_batch = camera_offset_rot_single.unsqueeze(0).expand(self.num_envs, -1)
        camera_offset_pos_single = torch.tensor(cfg.camera.offset.pos, device=self.device)
        self.camera_offset_pos_batch = camera_offset_pos_single.unsqueeze(0).expand(self.num_envs, -1)
        
        # Action weights
        self.action_weight = torch.tensor(self.cfg.action_weight, device=self.device).repeat(self.num_envs, 1)
        self.action_delta_weight = torch.tensor(self.cfg.action_delta_weight, device=self.device).repeat(self.num_envs, 1)
        
        # Get body and joint indices for each robot
        self._body_ids = {}
        self.gimbal_joint_idx = {}
        self._robot_mass = {}
        self._stabilizers = {}
        
        for agent in self.cfg.possible_agents:
            robot = self._robots[agent]
            self._body_ids[agent] = robot.find_bodies("body")[0]
            self.gimbal_joint_idx[agent] = {
                "yaw": robot.find_joints("yaw_joint")[0][0],
                "roll": robot.find_joints("roll_joint")[0][0],
                "pitch": robot.find_joints("pitch_joint")[0][0],
            }
            self._robot_mass[agent] = robot.root_physx_view.get_masses()[0].sum()
            robot_weight = (self._robot_mass[agent] * torch.tensor(self.sim.cfg.gravity, device=self.device).norm()).item()
            self._stabilizers[agent] = PointMass(0, robot_weight, self.num_envs, self.device)
        
```

