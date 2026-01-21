#!/usr/bin/env python
"""
Convert Dobot HDF5 robot data to LeRobot v3.0 dataset format.

This script handles Dobot robot data with 3 cameras (cam_left_wrist, cam_right_wrist, cam_high).
Updated for LeRobot v3.0 dataset format.

Usage:
    python hdf5_to_lerobot_dobot_v3.py \
        --input_dir /path/to/hdf5/files \
        --repo_id username/dataset_name \
        --fps 30 \
        --push_to_hub
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import subprocess
import cv2

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_huggingface_login() -> Optional[str]:
    """
    Verify HuggingFace login and return username.
    
    Returns:
        Username if logged in, None otherwise
    """
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        username = result.stdout.strip()
        logger.info(f"✓ Logged in as: {username}")
        return username
    except subprocess.CalledProcessError:
        logger.error("✗ Not logged in to HuggingFace")
        logger.error("  Please run: huggingface-cli login")
        return None
    except FileNotFoundError:
        logger.error("✗ huggingface-cli not found")
        logger.error("  Please install: pip install huggingface_hub[cli]")
        return None


def create_features_dict() -> Dict[str, Dict[str, Any]]:
    """
    Create the features dictionary for LeRobot dataset v3.0.
    
    Features must match the structure of frames you'll be adding.
    
    Returns:
        Dictionary defining the data structure for LeRobot dataset
    """
    features = {
        # === Actions and States ===
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": {"axes": ["action_" + str(i) for i in range(14)]}
        },
        "observation.state": {
            "dtype": "float32", 
            "shape": (14,),
            "names": {"axes": ["qpos_" + str(i) for i in range(14)]}
        },
        "observation.qvel": {
            "dtype": "float32",
            "shape": (14,),
            "names": {"axes": ["qvel_" + str(i) for i in range(14)]}
        },
        
        # === Camera Images ===
        # Images will be stored as videos (MP4 format)
        # Note: LeRobot expects (height, width, channels) format
        # This will be automatically converted to (channels, height, width) for policy
        "observation.images.cam_left_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]  # Must be "channels" (plural)
        },
        "observation.images.cam_right_wrist": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]  # Must be "channels" (plural)
        },
        "observation.images.cam_high": {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]  # Must be "channels" (plural)
        },
    }
    
    return features


def validate_hdf5_structure(file_path: Path) -> Dict[str, Any]:
    """
    Validate Dobot HDF5 file structure and return metadata.
    
    Args:
        file_path: Path to HDF5 file
        
    Returns:
        Dictionary with file metadata and validation results
    """
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "metadata": {}
    }
    
    try:
        with h5py.File(file_path, 'r') as f:
            # Check required datasets
            required_datasets = ['action', 'observations']
            for dataset in required_datasets:
                if dataset not in f:
                    validation_result["errors"].append(f"Missing required dataset: {dataset}")
                    validation_result["valid"] = False
            
            if not validation_result["valid"]:
                return validation_result
            
            # Check observations structure
            obs_group = f['observations']
            required_obs = ['qpos', 'qvel', 'images']
            for obs in required_obs:
                if obs not in obs_group:
                    validation_result["errors"].append(f"Missing required observation: {obs}")
                    validation_result["valid"] = False
            
            if not validation_result["valid"]:
                return validation_result
            
            # Check images structure
            images_group = obs_group['images']
            required_cameras = ['cam_left_wrist', 'cam_right_wrist', 'cam_high']
            for camera in required_cameras:
                if camera not in images_group:
                    validation_result["errors"].append(f"Missing required camera: {camera}")
                    validation_result["valid"] = False
            
            if not validation_result["valid"]:
                return validation_result
            
            # Extract metadata
            validation_result["metadata"] = {
                "episode_length": f['action'].shape[0],
                "action_dim": f['action'].shape[1],
                "qpos_dim": obs_group['qpos'].shape[1],
                "qvel_dim": obs_group['qvel'].shape[1],
            }
            
            # Check if images are compressed
            for camera in required_cameras:
                image_data = images_group[camera]
                if len(image_data.shape) == 2:  # Compressed format
                    validation_result["metadata"]["images_compressed"] = True
                    break
            else:
                validation_result["metadata"]["images_compressed"] = False
    
    except Exception as e:
        validation_result["errors"].append(f"Error reading HDF5 file: {str(e)}")
        validation_result["valid"] = False
    
    return validation_result


def decompress_jpeg_image(compressed_data: np.ndarray) -> np.ndarray:
    """
    Decompress JPEG image data.
    
    Args:
        compressed_data: Compressed JPEG data as numpy array
        
    Returns:
        Decompressed image as numpy array (H, W, C) in RGB format
    """
    jpeg_bytes = compressed_data.tobytes()
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode JPEG image")
    
    # Convert BGR to RGB (OpenCV loads as BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image



def load_hdf5_episode1(file_path: Path) -> Dict[str, np.ndarray]:
    """
    Load a single Dobot HDF5 episode file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing all data from the episode
    """
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Load actions and states
        data['action'] = f['action'][:]
        
        obs_group = f['observations']
        data['observation.state'] = obs_group['qpos'][:]
        data['observation.qvel'] = obs_group['qvel'][:]
        
        # Load and decompress images
        images_group = obs_group['images']
        print(images_group['cam_left_wrist'])
        episode_length = data['action'].shape[0]
        
        # Initialize image arrays
        data['observation.images.cam_left_wrist'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
        data['observation.images.cam_right_wrist'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
        data['observation.images.cam_high'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
        
        # Decompress images frame by frame
        logger.info(f"  Decompressing {episode_length} frames...")
        for frame_idx in range(episode_length):
            if frame_idx % 100 == 0 and frame_idx > 0:
                logger.info(f"    Processed {frame_idx}/{episode_length} frames")
            
            data['observation.images.cam_left_wrist'][frame_idx] = decompress_jpeg_image(
                images_group['cam_left_wrist'][frame_idx]
            )
            data['observation.images.cam_right_wrist'][frame_idx] = decompress_jpeg_image(
                images_group['cam_right_wrist'][frame_idx]
            )
            data['observation.images.cam_high'][frame_idx] = decompress_jpeg_image(
                images_group['cam_high'][frame_idx]
            )
    
    return data

def load_hdf5_episode(file_path: Path) -> Dict[str, np.ndarray]:
    """
    加载HDF5episode文件，适配已解压缩的图像数据
    """
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        # 加载动作和状态数据
        data['action'] = f['action'][:]
        
        obs_group = f['observations']
        data['observation.state'] = obs_group['qpos'][:]
        data['observation.qvel'] = obs_group['qvel'][:]
        
        # 加载图像数据
        images_group = obs_group['images']
        episode_length = data['action'].shape[0]
        
        # 检查图像数据格式
        cam_left_data = images_group['cam_left_wrist']
        logger.info(f"相机数据格式: {cam_left_data.shape}, 类型: {cam_left_data.dtype}")
        
        # 根据数据格式处理图像
        if len(cam_left_data.shape) == 4:  # 已经是解压缩格式 (frames, height, width, channels)
            # 直接使用数据，不需要解码
            data['observation.images.cam_left_wrist'] = cam_left_data[:]
            data['observation.images.cam_right_wrist'] = images_group['cam_right_wrist'][:]
            data['observation.images.cam_high'] = images_group['cam_high'][:]
            
            # 确保数据格式正确
            for camera in ['cam_left_wrist', 'cam_right_wrist', 'cam_high']:
                if data[f'observation.images.{camera}'].dtype != np.uint8:
                    logger.warning(f"{camera} 数据类型不是uint8，进行转换")
                    data[f'observation.images.{camera}'] = data[f'observation.images.{camera}'].astype(np.uint8)
                    
        else:  # 压缩格式，需要解码
            logger.info("检测到压缩图像数据，使用解码流程")
            # 初始化图像数组
            data['observation.images.cam_left_wrist'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
            data['observation.images.cam_right_wrist'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
            data['observation.images.cam_high'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
            
            # 使用增强的解码函数
            logger.info(f"  解码 {episode_length} 帧...")
            for frame_idx in range(episode_length):
                if frame_idx % 100 == 0 and frame_idx > 0:
                    logger.info(f"    已处理 {frame_idx}/{episode_length} 帧")
                
                data['observation.images.cam_left_wrist'][frame_idx] = decompress_jpeg_image_enhanced(
                    images_group['cam_left_wrist'][frame_idx]
                )
                data['observation.images.cam_right_wrist'][frame_idx] = decompress_jpeg_image_enhanced(
                    images_group['cam_right_wrist'][frame_idx]
                )
                data['observation.images.cam_high'][frame_idx] = decompress_jpeg_image_enhanced(
                    images_group['cam_high'][frame_idx]
                )
    
    return data

def convert_hdf5_to_lerobot(
    input_dir: Path,
    repo_id: str,
    root: Optional[Path] = None,
    fps: int = 30,
    task_name: str = "peg_in_hole",
    robot_type: str = "dobot",
    push_to_hub: bool = False,
    tags: Optional[list] = None,
    license: str = "apache-2.0",
    private: bool = False,
) -> None:
    """
    Convert HDF5 files to LeRobot v3.0 dataset format.
    
    Args:
        input_dir: Directory containing HDF5 files
        repo_id: Repository ID (e.g., 'username/dataset_name')
        root: Root directory to save dataset locally (optional)
        fps: Frames per second for the dataset
        task_name: Name of the task being performed
        robot_type: Type of robot used
        push_to_hub: Whether to push dataset to HuggingFace Hub
        tags: Tags for the dataset
        license: License for the dataset
        private: Whether to make the repository private
    """
    # Verify HuggingFace login if pushing to hub
    if push_to_hub:
        username = verify_huggingface_login()
        if username is None:
            raise ValueError("HuggingFace login required for pushing to hub")
    
    # Find all HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")
    
    logger.info(f"Found {len(hdf5_files)} HDF5 files")
    
    # Validate first file
    logger.info("Validating HDF5 file structure...")
    validation_result = validate_hdf5_structure(hdf5_files[0])
    
    if not validation_result["valid"]:
        logger.error("✗ HDF5 validation failed:")
        for error in validation_result["errors"]:
            logger.error(f"  - {error}")
        raise ValueError("HDF5 file validation failed")
    
    if validation_result["warnings"]:
        logger.warning("⚠ HDF5 validation warnings:")
        for warning in validation_result["warnings"]:
            logger.warning(f"  - {warning}")
    
    logger.info(f"✓ Validation passed. Episode length: {validation_result['metadata']['episode_length']}")
    
    # Create features dictionary
    features = create_features_dict()
    
    # Create LeRobot dataset
    logger.info("Creating LeRobot v3.0 dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type=robot_type,
        features=features,
        use_videos=True,  # Store images as videos (MP4 format)
        batch_encoding_size=1,  # Encode videos immediately after each episode
    )
    
    logger.info(f"✓ Dataset created at: {dataset.root}")
    
    # Process each episode
    total_frames = 0
    for episode_idx, hdf5_file in enumerate(hdf5_files):
        logger.info(f"\n[Episode {episode_idx + 1}/{len(hdf5_files)}] Processing: {hdf5_file.name}")
        
        # Load episode data
        episode_data = load_hdf5_episode(hdf5_file)
        episode_length = episode_data['action'].shape[0]
        
        logger.info(f"  Episode length: {episode_length} frames")
        
        # Create episode buffer
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_idx)
        
        # Add frames to episode buffer
        logger.info(f"  Adding frames to dataset...")
        for frame_idx in range(episode_length):
            if frame_idx % 100 == 0 and frame_idx > 0:
                logger.info(f"    Added {frame_idx}/{episode_length} frames")
            
            # Create frame dictionary
            # Note: 'task' must be included in the frame dictionary
            frame = {
                'action': episode_data['action'][frame_idx],
                'observation.state': episode_data['observation.state'][frame_idx],
                'observation.qvel': episode_data['observation.qvel'][frame_idx],
                'observation.images.cam_left_wrist': episode_data['observation.images.cam_left_wrist'][frame_idx],
                'observation.images.cam_right_wrist': episode_data['observation.images.cam_right_wrist'][frame_idx],
                'observation.images.cam_high': episode_data['observation.images.cam_high'][frame_idx],
                'task': task_name,  # Task is included in the frame itself
            }
            
            # Add frame to dataset
            dataset.add_frame(frame)
        
        # Save episode
        logger.info(f"  Saving episode...")
        dataset.save_episode()
        total_frames += episode_length
        logger.info(f"  ✓ Episode {episode_idx + 1} saved")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Conversion complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Dataset location: {dataset.root}")
    logger.info(f"Repository ID: {repo_id}")
    logger.info(f"Total episodes: {len(hdf5_files)}")
    logger.info(f"Total frames: {total_frames}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Features: {list(features.keys())}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        logger.info(f"\n{'='*60}")
        logger.info("Pushing dataset to HuggingFace Hub...")
        logger.info(f"{'='*60}")
        try:
            dataset.push_to_hub(
                tags=tags or ["lerobot", "robotics", "dobot", "demonstration"],
                license=license,
                private=private,
            )
            logger.info(f"\n✓ Dataset successfully pushed to Hub!")
            logger.info(f"  View at: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            logger.error(f"\n✗ Failed to push to HuggingFace Hub: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert HDF5 Dobot robot data to LeRobot v3.0 dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert and save locally
  python hdf5_to_lerobot_dobot_v3.py \\
      --input_dir ./hdf5_data \\
      --repo_id username/dobot_peg_in_hole \\
      --root ./lerobot_datasets

  # Convert and push to Hub
  python hdf5_to_lerobot_dobot_v3.py \\
      --input_dir ./hdf5_data \\
      --repo_id username/dobot_peg_in_hole \\
      --push_to_hub \\
      --tags robotics dobot peg-insertion

  # Convert with custom settings
  python hdf5_to_lerobot_dobot_v3.py \\
      --input_dir ./hdf5_data \\
      --repo_id username/dobot_dataset \\
      --fps 30 \\
      --task_name "peg_insertion" \\
      --robot_type "dobot_mg400" \\
      --push_to_hub \\
      --private
        """
    )
    
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing HDF5 files"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Repository ID (format: username/dataset_name)"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory to save dataset locally (default: ~/.cache/huggingface/lerobot)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="peg_in_hole",
        help="Name of the task (default: peg_in_hole)"
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="dobot",
        help="Type of robot (default: dobot)"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--tags",
        nargs="+",
        help="Tags for the dataset (e.g., robotics dobot)"
    )
    parser.add_argument(
        "--license",
        type=str,
        default="apache-2.0",
        help="License for the dataset (default: apache-2.0)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dir.exists():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    if "/" not in args.repo_id:
        raise ValueError(
            f"Invalid repo_id format: {args.repo_id}\n"
            "Expected format: username/dataset_name"
        )
    
    # Convert dataset
    convert_hdf5_to_lerobot(
        input_dir=args.input_dir,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        task_name=args.task_name,
        robot_type=args.robot_type,
        push_to_hub=args.push_to_hub,
        tags=args.tags,
        license=args.license,
        private=args.private,
    )


if __name__ == "__main__":
    main()

# python src/lerobot/datasets/v30/hdf5_to_lerobot_dobot_v3.py --input_dir ../datasets/fold_towel/ --repo_id lerobot/fold_towel_v3_0 --root ../huggingface/data