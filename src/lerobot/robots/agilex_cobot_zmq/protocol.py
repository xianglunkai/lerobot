# Shared by the Python 3.8 ROS bridge and the Python 3.12 robot client.
# Keep this module free of rospy and of lerobot imports.
from __future__ import annotations

import json
import struct

MAGIC = b"ACZ1"
VERSION = 1

# Bridge binds these. The inference client connects.
DEFAULT_CMD_PORT = 5600
DEFAULT_OBS_PORT = 5601
DEFAULT_RPC_PORT = 5602


def encode_observation(joints, jpeg_images, timestamp, endpose=None) -> bytes:
    """Pack one observation into a single buffer so ZMQ CONFLATE can keep only the latest."""
    cameras = [{"name": name, "nbytes": len(blob)} for name, blob in jpeg_images.items()]
    header = {
        "v": VERSION,
        "t": float(timestamp),
        "joints": {str(name): float(value) for name, value in joints.items()},
        "cameras": cameras,
    }
    if endpose:
        header["endpose"] = {
            str(side): [float(value) for value in pose] for side, pose in endpose.items()
        }
    header_bytes = json.dumps(header).encode("utf-8")
    payload = [MAGIC, struct.pack("<I", len(header_bytes)), header_bytes]
    payload.extend(jpeg_images.values())
    return b"".join(payload)


def decode_observation(blob: bytes) -> dict:
    if len(blob) < 8 or blob[:4] != MAGIC:
        raise ValueError("observation does not start with ACZ1")
    (header_len,) = struct.unpack_from("<I", blob, 4)
    header_end = 8 + header_len
    if header_end > len(blob):
        raise ValueError("truncated observation header")
    header = json.loads(blob[8:header_end].decode("utf-8"))
    if header.get("v") != VERSION:
        raise ValueError(f"unsupported observation version {header.get('v')}")
    offset = header_end
    images = {}
    for camera in header.get("cameras", []):
        size = int(camera["nbytes"])
        next_offset = offset + size
        if next_offset > len(blob):
            raise ValueError("truncated observation image")
        images[camera["name"]] = blob[offset:next_offset]
        offset = next_offset
    if offset != len(blob):
        raise ValueError("observation payload size mismatch")
    header["images"] = images
    return header


def encode_command(payload: dict) -> str:
    return json.dumps(payload)


def decode_command(raw: str) -> dict:
    message = json.loads(raw)
    if not isinstance(message, dict):
        raise ValueError("command must be a JSON object")
    return message
