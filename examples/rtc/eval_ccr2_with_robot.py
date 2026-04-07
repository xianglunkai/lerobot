import logging
import sys
import time
import traceback
from dataclasses import dataclass, field
from queue import Queue, Empty, Full
from threading import Event, Lock, Thread
from collections import deque
from typing import Optional, Tuple

import torch
from torch import Tensor
import numpy as np

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)

from lerobot.rl.process import ProcessSignalHandler
from lerobot.robots.utils import make_robot_from_config
from lerobot.utils.hub import HubMixin
from lerobot.utils.utils import init_logging
from lerobot.utils.bspline import BSplineFitter

from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    so_follower,
    agilex_cobot,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


@dataclass
class DemoConfig(HubMixin):
    # Policy configuration
    policy: Optional[PreTrainedConfig] = None
    
    # Robot configuration
    robot: Optional[RobotConfig] = None
    
    # 核心参数
    fps: float = 30.0
    device: Optional[str] = None
    duration: float = 60.0
    
    # 动作预测参数
    action_history_horizon: int = 8   # 历史动作长度（用于CCR）
    
    # CCR参数（可选）
    use_ccr: bool = True              # 是否使用CCR平滑
    ccr_k: int = 3                    # B-spline阶数
    ccr_n_ctrl: int = 8               # 控制点数量
    ccr_n_free: int = 4               # 优化控制点数
    ccr_last_pt_weight: float = 0.05  # 连续性权重
    
    # Task to execute
    task: str = field(default="", metadata={"help": "Task to execute"})
    
    # Torch compile configuration
    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    torch_compile_backend: str = field(
        default="inductor",
        metadata={"help": "Backend for torch.compile (inductor, aot_eager, cudagraphs)"},
    )

    torch_compile_mode: str = field(
        default="max-autotune",
        metadata={"help": "Compilation mode (default, reduce-overhead, max-autotune)"},
    )

    torch_compile_disable_cudagraphs: bool = field(
        default=True,
        metadata={
            "help": "Disable CUDA graphs in torch.compile. Required due to in-place tensor "
            "operations in denoising loop (x_t += dt * v_t) which cause tensor aliasing issues."
        },
    )

    # Visualization configuration
    enable_visualization: bool = field(
        default=False,
        metadata={"help": "Enable real-time action visualization plot"},
    )

    viz_history_size: int = field(
        default=1200,  # 40 seconds of history at 30 FPS
        metadata={"help": "Number of action steps to display in visualization"},
    )

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path
        else:
            raise ValueError("Policy path is required")
        
        if self.robot is None:
            raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class ThreadSafeActionQueue:
    """线程安全的动作队列"""
    
    def __init__(self, maxlen: int = 1):
        self._queue = deque(maxlen=maxlen)
        self._lock = Lock()
        self._not_empty = Event()
    
    def append(self, item: Tuple[float, Tensor]) -> bool:
        """添加新动作，如果队列满则覆盖"""
        with self._lock:
            is_full = len(self._queue) >= self._queue.maxlen
            self._queue.append(item)
            self._not_empty.set()
            return is_full
    
    def popleft(self, timeout: Optional[float] = None) -> Optional[Tuple[float, Tensor]]:
        """获取动作，支持超时"""
        if not self._not_empty.wait(timeout=timeout):
            return None
        
        with self._lock:
            if len(self._queue) == 0:
                return None
            item = self._queue.popleft()
            if len(self._queue) == 0:
                self._not_empty.clear()
            return item
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)
    
    def clear(self):
        with self._lock:
            self._queue.clear()
            self._not_empty.clear()


def inference_thread(
    policy,
    robot: RobotWrapper,
    robot_observation_processor,
    action_queue: ThreadSafeActionQueue,
    shutdown_event: Event,
    cfg: DemoConfig,
):
    """
    纯推理线程：获取观测 -> 模型推理 -> 放入原始预测结果
    """
    try:
        logger.info("[INFERENCE] Starting inference thread")
        
        dataset_features = hw_to_dataset_features(robot.observation_features(), "observation")
        policy_device = policy.config.device

        # Load preprocessor and postprocessor from pretrained files
        # The stats are embedded in the processor .safetensors files
        logger.info(f"[GET_ACTIONS] Loading preprocessor/postprocessor from {cfg.policy.pretrained_path}")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            dataset_stats=None,  # Will load from pretrained processor files
            preprocessor_overrides={
                "device_processor": {"device": cfg.policy.device},
            },
        )       
        logger.info("[GET_ACTIONS] Preprocessor/postprocessor loaded successfully with embedded stats")

        
        inference_count = 0
        last_log_time = time.time()
        
        while not shutdown_event.is_set():
            t_infer_start = time.perf_counter()
            
            # 1. 获取观测
            obs = robot.get_observation()
            
            # Apply robot observation processor
            obs_processed = robot_observation_processor(obs)

            obs_with_policy_features = build_dataset_frame(
                dataset_features, obs_processed, prefix="observation"
            )

            for name in obs_with_policy_features:
                obs_with_policy_features[name] = torch.from_numpy(obs_with_policy_features[name])
                if "image" in name:
                    obs_with_policy_features[name] = (
                        obs_with_policy_features[name].type(torch.float32) / 255
                    )
                    obs_with_policy_features[name] = (
                        obs_with_policy_features[name].permute(2, 0, 1).contiguous()
                    )
                obs_with_policy_features[name] = obs_with_policy_features[name].unsqueeze(0)
                obs_with_policy_features[name] = obs_with_policy_features[name].to(policy_device)

            obs_with_policy_features["task"] = [cfg.task]  # Task should be a list, not a string!
            obs_with_policy_features["robot_type"] = (
                robot.robot.name if hasattr(robot.robot, "name") else ""
            )

            preproceseded_obs = preprocessor(obs_with_policy_features)

            # 3. 模型推理
            with torch.no_grad():
                actions = policy.predict_action_chunk(
                    preproceseded_obs,
                    smoothing_method=None,
                )
                # 4. 后处理
                postprocessed_actions = postprocessor(actions)

                postprocessed_actions = postprocessed_actions.squeeze(0)
            
                            
            inference_count += 1
            if inference_count < 3:
                continue
            
            # 5. 放入队列
            infer_latency = time.perf_counter() - t_infer_start
            dropped = action_queue.append((t_infer_start, postprocessed_actions.clone()))
            
            if dropped:
                print("[INFERENCE] Dropped old action chunk (queue full)")
            
            # 日志控制（每5秒一次）
            if time.time() - last_log_time > 10.0:
                print(f"[INFERENCE] Speed: {inference_count/10:.1f} chunks/sec, "
                           f"latency: {infer_latency*1000:.1f}ms")
                inference_count = 0
                last_log_time = time.time()
            
    except Exception as e:
        logger.error(f"[INFERENCE] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


def control_thread(
    robot: RobotWrapper,
    robot_action_processor,
    action_queue: ThreadSafeActionQueue,
    viz_queue: Optional[Queue],
    shutdown_event: Event,
    cfg: DemoConfig,
):
    """
    控制线程：获取预测 -> CCR优化 -> 执行动作
    """
    try:
        logger.debug("[CONTROL] Starting control thread")
        
        # 预分配BSplineFitter（避免重复创建）
        bspline_fitter = BSplineFitter(
            T=cfg.policy.n_action_steps or 16,  # 从配置获取
            k=cfg.ccr_k, 
            n_ctrl=cfg.ccr_n_ctrl
        ) if cfg.use_ccr else None
        
        # 执行历史（用于CCR）
        actions_dim = len(robot.action_features())
        zero_action = np.zeros(actions_dim, dtype=np.float32)
        latest_actions_queue = deque([zero_action.copy() for _ in range(50)], maxlen=50)
        
        # 动作缓冲
        action_buffer = deque(maxlen=50)
        
        control_interval = 1.0 / cfg.fps
        control_count = 0
        empty_count = 0
        
        while not shutdown_event.is_set():
            loop_start = time.perf_counter()
            
            # 1. 获取新预测（非阻塞，超时1ms）
            new_prediction = action_queue.popleft(timeout=0.001)
            
            if new_prediction is not None:
                t_obs, raw_actions = new_prediction
                raw_actions = raw_actions.cpu().numpy()  # (T, action_dim)
                
                t_now = time.perf_counter()
                delay_s = t_now - t_obs
                delay_steps = int(round(delay_s * cfg.fps))
                
                if cfg.use_ccr:
                    # CCR平滑处理
                    n_prefix = min(cfg.action_history_horizon + delay_steps, len(latest_actions_queue))
                    y_prefix = np.array(list(latest_actions_queue)[-n_prefix:], dtype=np.float32)
                    
                    T = raw_actions.shape[0]
                    # 重用fitter，避免重新分配内存
                    if bspline_fitter.T != T:
                        bspline_fitter = BSplineFitter(T=T, k=cfg.ccr_k, n_ctrl=cfg.ccr_n_ctrl)
                        print(f"[CONTROL] Recreated BSplineFitter for T={T}")
                    
                    ctrl_y = bspline_fitter.fit(raw_actions)
                    ctrl_y_refit = bspline_fitter.refit_prefix_w(
                        y_prefix=y_prefix,
                        ctrl_y=ctrl_y,
                        n_prefix=n_prefix,
                        n_free=cfg.ccr_n_free,
                        last_pt_weight=cfg.ccr_last_pt_weight
                    )
                    
                    optimized_actions, _ = bspline_fitter.rebuild(ctrl_y_refit)
                    
                    # 关键修复：正确处理边界
                    if n_prefix < len(optimized_actions):
                        actions_to_exec = optimized_actions[n_prefix:]
                    else:
                        # 延迟过大，使用原始预测的最后部分
                        print(f"[CONTROL] High delay ({delay_steps} steps), "f"skipping CCR prefix")
                        actions_to_exec = raw_actions[max(0, delay_steps):]
                else:
                    # 无CCR：直接应用延迟补偿
                    start_idx = min(delay_steps, raw_actions.shape[0] - 1)
                    actions_to_exec = raw_actions[start_idx:]
                
                # 更新缓冲
                action_buffer.clear()
                action_buffer.extend(actions_to_exec)
                
                print(f"[CONTROL] New chunk: delay={delay_steps} steps, "
                           f"buffer={len(action_buffer)}")
            
            # 2. 获取动作执行
            if len(action_buffer) > 0:
                action = action_buffer.popleft()
                latest_actions_queue.append(action.copy())
                empty_count = 0
            else:
                action = None
                empty_count += 1
                if empty_count % 100 == 0:  # 每100次空循环警告一次
                    print(f"[CONTROL] Action buffer empty ({empty_count} cycles)")
            
            # 3. 执行动作
            if action is not None:
                action_dict = {key: float(action[i]) 
                              for i, key in enumerate(robot.action_features())}
                action_processed = robot_action_processor((action_dict, None))
                robot.send_action(action_processed)
                
                # 可视化（发送numpy数组）
                if viz_queue is not None and cfg.enable_visualization:
                    try:
                        viz_queue.put_nowait((time.time(), action.copy()))
                    except Full:
                        pass  # 丢弃旧数据
                
                control_count += 1
            
            # 4. 精确周期控制
            elapsed = time.perf_counter() - loop_start
            sleep_time = control_interval - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"[CONTROL] Overrun: {elapsed*1000:.1f}ms > "f"{control_interval*1000:.1f}ms")
            
                
    except Exception as e:
        logger.error(f"[CONTROL] Fatal error: {e}")
        logger.error(traceback.format_exc())
        shutdown_event.set()


def update_visualization(
    fig, axes, lines, history_buffer: deque, cfg: DemoConfig
):
    """Update the visualization plot with new data."""
    try:
        if len(history_buffer) == 0:
            return

        # 修复：统一处理numpy和tensor
        timestamps = [h[0] for h in history_buffer]
        
        # 安全转换：支持numpy数组和tensor
        actions_list = []
        for h in history_buffer:
            act = h[1]
            if isinstance(act, torch.Tensor):
                act = act.cpu().numpy()
            actions_list.append(np.asarray(act))
        
        actions_array = np.stack(actions_list, axis=0)  # (N, action_dim)
        
        x_data = np.arange(len(timestamps))
        
        # 更新线条数据
        for i, line in enumerate(lines):
            if i < actions_array.shape[1]:
                line.set_data(x_data, actions_array[:, i])
        
        # 动态调整Y轴
        for i, ax in enumerate(axes):
            if i < actions_array.shape[1]:
                y_min, y_max = np.min(actions_array[:, i]), np.max(actions_array[:, i])
                margin = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
                ax.set_ylim(y_min - margin, y_max + margin)
        
        # 调整X轴
        x_max = max(cfg.viz_history_size, len(x_data))
        for ax in axes:
            ax.set_xlim(max(0, len(x_data) - cfg.viz_history_size), len(x_data))
        
        fig.canvas.draw()
        fig.canvas.flush_events()

    except Exception as e:
        logger.warning(f"[VIZ] Error updating plot: {e}")


def save_visualization(fig, save_path: str = "action_history.png"):
    """Save the final visualization to file."""
    try:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"[VIZ] Visualization saved to {save_path}")
    except Exception as e:
        logger.error(f"[VIZ] Failed to save visualization: {e}")


def _apply_torch_compile(policy, cfg: DemoConfig):
    """Apply torch.compile to the policy's predict_action_chunk method."""
    # PI models handle their own compilation
    if policy.type in ["pi05", "pi0"]:
        return policy

    try:
        if not hasattr(torch, "compile"):
            logger.warning(f"torch.compile requires PyTorch 2.0+. Current: {torch.__version__}")
            return policy

        logger.info("Applying torch.compile to predict_action_chunk...")
        logger.info(f"  Backend: {cfg.torch_compile_backend}, Mode: {cfg.torch_compile_mode}")

        torch.set_float32_matmul_precision("high")
        compile_kwargs = {
            "backend": cfg.torch_compile_backend,
            "mode": cfg.torch_compile_mode,
        }
        
        if cfg.torch_compile_disable_cudagraphs:
            compile_kwargs["options"] = {"triton.cudagraphs": False}

        original_method = policy.predict_action_chunk
        compiled_method = torch.compile(original_method, **compile_kwargs)
        policy.predict_action_chunk = compiled_method
        logger.info("✓ Successfully compiled predict_action_chunk")

    except Exception as e:
        logger.error(f"Failed to apply torch.compile: {e}")

    return policy


@parser.wrap()
def demo_cli(cfg: DemoConfig):
    """主入口"""
    init_logging()
    logger.info(f"Using device: {cfg.device}")
    
    # 信号处理
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    shutdown_event = signal_handler.shutdown_event
    
    policy = None
    robot = None
    infer_thread = None
    ctrl_thread = None
    
    policy_class = get_policy_class(cfg.policy.type)
    
    # Load config and set compile_model for pi0/pi05 models
    config = PreTrainedConfig.from_pretrained(cfg.policy.pretrained_path)
    
    if cfg.policy.type == "pi05" or cfg.policy.type == "pi0":
        config.compile_model = cfg.use_torch_compile
        
    if config.use_peft:
        from peft import PeftConfig, PeftModel

        peft_pretrained_path = cfg.policy.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        policy = policy_class.from_pretrained(
            pretrained_name_or_path=peft_config.base_model_name_or_path, config=config
        )
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)
    else:
        policy = policy_class.from_pretrained(cfg.policy.pretrained_path, config=config)

    policy.type = cfg.policy.type


    assert policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"

    policy = policy.to(cfg.device)
    policy.eval()
    
    if cfg.use_torch_compile:
        policy = _apply_torch_compile(policy, cfg)
    
    # 初始化机器人
    logger.info(f"Initializing robot: {cfg.robot.type}")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    robot.reset_to_default_positions()
    time.sleep(3)
    robot_wrapper = RobotWrapper(robot)
    
    # 处理器
    obs_processor = make_default_robot_observation_processor()
    act_processor = make_default_robot_action_processor()
    
    # Prepare visualization queue if enabled
    viz_queue = None
    fig = None
    axes = None
    lines = None
    history_buffer = None
    
    if cfg.enable_visualization:
        try:
            import matplotlib.pyplot as plt
            plt.ion()  # Enable interactive mode
            
            action_features = []
            for i, key in enumerate(robot_wrapper.action_features()):
                action_features.append(key)
        
            num_actions = len(action_features)
            fig, axes = plt.subplots(num_actions, 1, figsize=(10, 2 * num_actions), sharex=True)
            if num_actions == 1:
                axes = [axes]
            fig.suptitle("Executed Actions (Sent to Robot)", fontsize=14)

            lines = []
            y_min, y_max = 0, 1
            for i, ax in enumerate(axes):
                line, = ax.plot([], [], linewidth=1.5, color='red', label='Executed')
                lines.append(line)
                ax.set_ylabel(action_features[i], fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(y_min, y_max)

            axes[-1].set_xlabel("Time Steps", fontsize=10)
            plt.tight_layout()
            plt.show(block=False)
            
            cfg.viz_history_size = max(int(cfg.duration * cfg.fps),  cfg.viz_history_size)
            
            history_buffer = deque(maxlen=cfg.viz_history_size)
            viz_queue = Queue(maxsize=cfg.viz_history_size * 2)
            logger.info("[VIZ] Real-time visualization initialized")
        except ImportError:
            logger.warning("[VIZ] matplotlib not available, visualization disabled")
            cfg.enable_visualization = False
    
    # 线程安全的动作队列
    action_queue = ThreadSafeActionQueue(maxlen=1)
    
    # 启动推理线程
    infer_thread = Thread(
        target=inference_thread,
        args=(policy, robot_wrapper, obs_processor, action_queue, shutdown_event, cfg),
        daemon=True,
        name="Inference"
    )
    
    # 启动控制线程
    ctrl_thread = Thread(
        target=control_thread,
        args=(robot_wrapper, act_processor, action_queue, viz_queue, shutdown_event, cfg),
        daemon=True,
        name="Control"
    )
  
    infer_thread.start()
    logger.info("Started get actions thread")
    
    time.sleep(3)
    
    ctrl_thread.start()
    logger.info("Started actor thread")
    
    # 主循环
    logger.info(f"Running for {cfg.duration}s...")
    start_time = time.time()

    while not shutdown_event.is_set() and (time.time() - start_time) < cfg.duration:
        time.sleep(10)

        if time.time() - start_time > cfg.duration:
            break

    logger.info("Demo duration reached or shutdown requested")
    
    # 清理
    shutdown_event.set()
    
    # 等待线程结束（带超时）
    if infer_thread and infer_thread.is_alive():
        logger.info("Waiting for inference thread...")
        infer_thread.join()
        if infer_thread.is_alive():
            logger.warning("Inference thread did not terminate in time")
    
    if ctrl_thread and ctrl_thread.is_alive():
        logger.info("Waiting for control thread...")
        ctrl_thread.join()
        if ctrl_thread.is_alive():
            logger.warning("Control thread did not terminate in time")
    
    # Final visualization update and save
    if cfg.enable_visualization and fig is not None:
        # Process any remaining data
        if viz_queue is not None:
            try:
                while not viz_queue.empty():
                    item = viz_queue.get_nowait()
                    history_buffer.append(item)
            except Exception:
                pass
        
        # Final update
        update_visualization(fig, axes, lines, history_buffer, cfg)
        
        # Save the figure
        save_visualization(fig)
        
        # Keep plot open for a moment then close
        plt.pause(2.0)
        plt.close(fig)
    
    # 机器人安全状态
    if robot is not None:
        try:
            logger.info("Returning robot to default position...")
            robot.reset_to_default_positions()
            time.sleep(1.0)
            robot.disconnect()
            logger.info("Robot disconnected")
        except Exception as e:
            logger.error(f"Robot cleanup error: {e}")
    
    logger.info("Cleanup completed")
    


    

if __name__ == "__main__":
    demo_cli()
    logger.info("RTC demo finished")