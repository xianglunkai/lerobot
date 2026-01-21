#!/usr/bin/env python
"""
Convert dobot HDF5 robot data to LeRobot dataset format.
This script handles dobot robot data with 3 cameras (left_wrist, right_wrist, top).

This script converts HDF5 files containing dobot robot demonstrations to the LeRobot dataset format.
Based on the dobot data structure with compressed JPEG images.

Usage:
    python hdf5_to_lerobot_dobot.py --input_dir /path/to/hdf5/files --output_dir /path/to/output --fps 30
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import subprocess
import sys
import cv2
from huggingface_hub import HfApi, login
from huggingface_hub.utils import HfHubHTTPError

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.buffer import guess_feature_info

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_huggingface_login() -> Optional[str]:
    """
    Verify HuggingFace login and return username.
    
    Returns:
        Username if logged in, None otherwise
    """
    try:
        # Try to get current user
        result = subprocess.run(
            ["huggingface-cli", "whoami"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        username = result.stdout.strip()
        logger.info(f"Logged in as: {username}")
        return username
    except subprocess.CalledProcessError:
        logger.error("Not logged in to HuggingFace. Please run: huggingface-cli login")
        return None
    except FileNotFoundError:
        logger.error("huggingface-cli not found. Please install huggingface_hub with CLI support.")
        return None


def create_features_dict() -> Dict[str, Dict[str, Any]]:
    """
    Create the features dictionary for LeRobot dataset based on the dobot HDF5 structure.
    
    Returns:
        Dictionary defining the data structure for LeRobot dataset
    """
    features = {
        # Robot actions and states (matching the dobot data structure)
        "action": {"dtype": "float32", "shape": (14,), "names": None},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": None},  # qpos
        "observation.qvel": {"dtype": "float32", "shape": (14,), "names": None},  # qvel (from HDF5)
        
        # Camera images (matching the dobot data format)
        "observation.images.left_wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "observation.images.right_wrist": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "observation.images.top": {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        
        # Required metadata fields (matching the example dataset)
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
    }
    
    return features


def validate_hdf5_structure(file_path: Path) -> Dict[str, Any]:
    """
    Validate dobot HDF5 file structure and return metadata.
    
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
            required_cameras = ['left_wrist', 'right_wrist', 'top']
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
                "image_shape": images_group['left_wrist'].shape[1:],  # (H, W, C) or compressed size
            }
            
            # Validate dimensions
            expected_dims = {
                "action": 14,
                "qpos": 14,
                "qvel": 14,
            }
            
            for key, expected_dim in expected_dims.items():
                if key == "action":
                    actual_dim = f['action'].shape[1]
                else:
                    actual_dim = obs_group[key].shape[1]
                
                if actual_dim != expected_dim:
                    validation_result["warnings"].append(
                        f"Dimension mismatch for {key}: expected {expected_dim}, got {actual_dim}"
                    )
            
            # Check if images are compressed (JPEG format)
            for camera in required_cameras:
                image_data = images_group[camera]
                if len(image_data.shape) == 2:  # Compressed format (frames, compressed_bytes)
                    validation_result["warnings"].append(
                        f"Images in {camera} are in compressed format (JPEG), will be decompressed during loading"
                    )
    
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
        Decompressed image as numpy array (H, W, C)
    """
    # Convert to bytes
    jpeg_bytes = compressed_data.tobytes()
    
    # Decode JPEG
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode JPEG image")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def load_hdf5_episode(file_path: Path) -> Dict[str, np.ndarray]:
    """
    Load a single dobot HDF5 episode file.
    
    Args:
        file_path: Path to the HDF5 file
        
    Returns:
        Dictionary containing all data from the episode
    """
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Load actions
        data['action'] = f['action'][:]  # Shape: (697, 14)
        
        # Load observations
        obs_group = f['observations']
        data['observation.state'] = obs_group['qpos'][:]  # Shape: (697, 14)
        data['observation.qvel'] = obs_group['qvel'][:]  # Shape: (697, 14)
        
        # Load and decompress images
        images_group = obs_group['images']
        episode_length = data['action'].shape[0]
        
        # Initialize image arrays
        data['observation.images.left_wrist'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
        data['observation.images.right_wrist'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
        data['observation.images.top'] = np.zeros((episode_length, 480, 640, 3), dtype=np.uint8)
        
        # Decompress images frame by frame
        for frame_idx in range(episode_length):
            # Decompress left_wrist image
            compressed_left = images_group['left_wrist'][frame_idx]
            data['observation.images.left_wrist'][frame_idx] = decompress_jpeg_image(compressed_left)
            
            # Decompress right_wrist image
            compressed_right = images_group['right_wrist'][frame_idx]
            data['observation.images.right_wrist'][frame_idx] = decompress_jpeg_image(compressed_right)
            
            # Decompress top image
            compressed_top = images_group['top'][frame_idx]
            data['observation.images.top'][frame_idx] = decompress_jpeg_image(compressed_top)
    
    return data


def convert_hdf5_to_lerobot(
    input_dir: Path,
    output_dir: Path,
    fps: int = 30,
    repo_id: str = "dobot_robot_dataset",
    task_name: str = "peg_in_hole",
    robot_type: str = "dobot",
    overwrite: bool = False,
    push_to_hub: bool = False,
    private: bool = False,
    tags: Optional[list] = None,
    license: str = "apache-2.0"
) -> None:
    """
    Convert HDF5 files to LeRobot dataset format.
    
    Args:
        input_dir: Directory containing HDF5 files
        output_dir: Directory to save the LeRobot dataset
        fps: Frames per second for the dataset
        repo_id: Repository ID for the dataset
        task_name: Name of the task being performed
        robot_type: Type of robot used
        overwrite: Whether to overwrite existing output directory
        push_to_hub: Whether to push dataset to HuggingFace Hub
        private: Whether to make the repository private
        tags: Tags for the dataset
        license: License for the dataset
    """
    # Verify HuggingFace login if pushing to hub
    if push_to_hub:
        username = verify_huggingface_login()
        if username is None:
            raise ValueError("HuggingFace login required for pushing to hub")
        
        # Update repo_id with username if not provided
        if "/" not in repo_id:
            repo_id = f"{username}/{repo_id}"
    
    # Find all HDF5 files
    hdf5_files = sorted(list(input_dir.glob("*.hdf5")))
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")
    
    logger.info(f"Found {len(hdf5_files)} HDF5 files")
    
    # Validate first file to ensure consistent structure
    logger.info("Validating HDF5 file structure...")
    validation_result = validate_hdf5_structure(hdf5_files[0])
    
    if not validation_result["valid"]:
        logger.error("HDF5 validation failed:")
        for error in validation_result["errors"]:
            logger.error(f"  - {error}")
        raise ValueError("HDF5 file validation failed")
    
    if validation_result["warnings"]:
        logger.warning("HDF5 validation warnings:")
        for warning in validation_result["warnings"]:
            logger.warning(f"  - {warning}")
    
    logger.info(f"Validation passed. Episode length: {validation_result['metadata']['episode_length']}")
    
    # Create features dictionary
    features = create_features_dict()
    
    # Handle existing directory
    if output_dir.exists():
        if overwrite:
            import shutil
            logger.info(f"Removing existing directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} already exists. Use --overwrite to overwrite it.")
    
    # Create LeRobot dataset
    logger.info("Creating LeRobot dataset...")
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features=features,
        use_videos=True,  # Store images as videos
        image_writer_processes=0,
        image_writer_threads=4,
    )
    
    # Process each episode
    total_frames = 0
    for episode_idx, hdf5_file in enumerate(hdf5_files):
        logger.info(f"Processing episode {episode_idx + 1}/{len(hdf5_files)}: {hdf5_file.name}")
        
        # Load episode data
        episode_data = load_hdf5_episode(hdf5_file)
        
        # Get episode length
        episode_length = episode_data['action'].shape[0]
        logger.info(f"Episode length: {episode_length} frames")
        
        # Create episode buffer
        dataset.episode_buffer = dataset.create_episode_buffer(episode_index=episode_idx)
        
        # Process each frame
        for frame_idx in range(episode_length):
            # Create frame dictionary with required data fields and 'next.done'
            frame = {
                'action': episode_data['action'][frame_idx],
                'observation.state': episode_data['observation.state'][frame_idx],
                'observation.qvel': episode_data['observation.qvel'][frame_idx],
                'observation.images.left_wrist': episode_data['observation.images.left_wrist'][frame_idx],
                'observation.images.right_wrist': episode_data['observation.images.right_wrist'][frame_idx],
                'observation.images.top': episode_data['observation.images.top'][frame_idx],
                'next.done': np.array([frame_idx == episode_length - 1], dtype=np.bool_),
            }
            # Add frame to dataset
            dataset.add_frame(frame, task=task_name)
        
        # Save episode
        dataset.save_episode()
        total_frames += episode_length
        logger.info(f"Saved episode {episode_idx + 1}")
    
    # Stop image writer
    dataset.stop_image_writer()
    
    logger.info(f"Conversion complete! Dataset saved to {output_dir}")
    logger.info(f"Total episodes: {len(hdf5_files)}")
    logger.info(f"Total frames: {total_frames}")
    
    # Print dataset info
    print(f"\nDataset created successfully!")
    print(f"Location: {output_dir}")
    print(f"Features: {list(dataset.features.keys())}")
    print(f"FPS: {dataset.fps}")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Frames: {dataset.num_frames}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        logger.info("Pushing dataset to HuggingFace Hub...")
        try:
            dataset.push_to_hub(
                tags=tags or ["lerobot", "robotics", "demonstration"],
                license=license,
                private=private
            )
            logger.info(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            logger.error(f"Failed to push to HuggingFace Hub: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 robot data to LeRobot dataset format")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory containing HDF5 files")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output directory for LeRobot dataset")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--repo_id", type=str, default="dobot_robot_dataset", help="Repository ID for dataset")
    parser.add_argument("--task_name", type=str, default="peg_in_hole", help="Name of the task")
    parser.add_argument("--robot_type", type=str, default="dobot", help="Type of robot")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output directory")
    parser.add_argument("--push_to_hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--tags", nargs="+", help="Tags for the dataset")
    parser.add_argument("--license", type=str, default="apache-2.0", help="License for the dataset")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_dir.exists():
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert dataset
    convert_hdf5_to_lerobot(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        repo_id=args.repo_id,
        task_name=args.task_name,
        robot_type=args.robot_type,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
        private=args.private,
        tags=args.tags,
        license=args.license
    )


if __name__ == "__main__":
    main() 