from dataclasses import dataclass, field

from ..config import RobotConfig
from .protocol import DEFAULT_CMD_PORT, DEFAULT_OBS_PORT, DEFAULT_RPC_PORT


@RobotConfig.register_subclass("agilex_cobot_zmq")
@dataclass
class AgilexCobotZmqConfig(RobotConfig):
    """Python 3.12 client for the AgileX cobot. Hardware I/O stays on the ZMQ bridge."""

    remote_ip: str = "127.0.0.1"
    port_zmq_cmd: int = DEFAULT_CMD_PORT
    port_zmq_observations: int = DEFAULT_OBS_PORT
    port_zmq_rpc: int = DEFAULT_RPC_PORT

    connect_timeout_s: float = 15.0
    polling_timeout_ms: int = 50
    # Returned observations older than this are still served, and a warning is logged.
    obs_stale_timeout_s: float = 1.0
    command_send_timeout_ms: int = 100
    rpc_timeout_ms: int = 30000
    # Log obs age / image stats / outgoing joint cmds (INFO, rate-limited).
    debug_zmq: bool = False
    debug_zmq_every_s: float = 1.0

    use_external_commands: bool = False

    with_mobile_base: bool = False
    with_l_arm: bool = True
    with_r_arm: bool = True
    with_left_camera: bool = True
    with_right_camera: bool = True
    with_front_camera: bool = True
    image_height: int = 480
    image_width: int = 640
    # Keep false when Rerun colors look correct (this ROS/JPEG path). Set true
    # only if red/blue are swapped in Rerun relative to the real cameras.
    convert_bgr_to_rgb: bool = False

    left_arm_joints: list[str] = field(
        default_factory=lambda: [
            "left_joint0",
            "left_joint1",
            "left_joint2",
            "left_joint3",
            "left_joint4",
            "left_joint5",
            "left_joint6",
        ]
    )
    right_arm_joints: list[str] = field(
        default_factory=lambda: [
            "right_joint0",
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
            "right_joint6",
        ]
    )
    mobile_base_joints: list[str] = field(default_factory=lambda: ["vx", "vtheta"])

    urdf_path: str = "/home/xlk/work/lerobot/examples/hil-serl/aloha_new_description/urdf"
    ik_target_frame_name: str = "gripper_frame_link"
    ik_joint_names: list[str] = field(
        default_factory=lambda: [
            "right_joint0",
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
            "gripper_frame_joint",
        ]
    )
    eef_input_fps: float = 30.0
    eef_command_timeout_s: float = 0.2
    velocity_filter_cutoff_hz: float = 5.0

    def __post_init__(self):
        if self.connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be > 0")
        if self.eef_command_timeout_s <= 0:
            raise ValueError("eef_command_timeout_s must be > 0")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image_height and image_width must be > 0")
        super().__post_init__()
