#!/usr/bin/env python3
"""Fake AgileX ZMQ bridge for remote-inference dry runs.

It speaks the same three ports as run_agilex_cobot_zmq_bridge.sh and does not
import ROS. Joint targets from the rollout client are applied immediately and
printed. Camera frames are colored placeholders.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq

HERE = Path(__file__).resolve().parent
PROTOCOL_DIR = HERE.parent / "src" / "lerobot" / "robots" / "agilex_cobot_zmq"
if str(PROTOCOL_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOCOL_DIR))

from protocol import (  # noqa: E402
    DEFAULT_CMD_PORT,
    DEFAULT_OBS_PORT,
    DEFAULT_RPC_PORT,
    decode_command,
    encode_command,
    encode_observation,
)

logger = logging.getLogger("zmq_robot_sim")

JOINTS = [f"left_joint{i}" for i in range(7)] + [f"right_joint{i}" for i in range(7)]
CAMERAS = ("high", "left", "right")
# Solid bright colors make this FineART checkpoint emit empty language; black
# frames match the server warmup distribution and keep /autosteer usable offline.
CAMERA_COLORS = {"high": (0, 0, 0), "left": (0, 0, 0), "right": (0, 0, 0)}
RESET_LEFT = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.07]
RESET_RIGHT = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.07]


def _parse_args():
    parser = argparse.ArgumentParser(description="Simulated AgileX ZMQ robot")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=DEFAULT_OBS_PORT)
    parser.add_argument("--rpc-port", type=int, default=DEFAULT_RPC_PORT)
    parser.add_argument("--obs-hz", type=float, default=30.0)
    parser.add_argument("--print-every", type=float, default=0.5)
    return parser.parse_args()


def _encode_images(step: int) -> dict[str, bytes]:
    del step  # kept so callers can pass a frame counter without changing the API
    encoded = {}
    for name in CAMERAS:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = CAMERA_COLORS[name]
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            encoded[name] = buffer.tobytes()
    return encoded


def _format_arm(values: list[float]) -> str:
    return " ".join(f"{value:+.3f}" for value in values)


class SimulatedRobot:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._joints = {name: 0.0 for name in JOINTS}
        self._last_print = 0.0

    def reset(self) -> None:
        with self._lock:
            for index, name in enumerate(JOINTS[:7]):
                self._joints[name] = RESET_LEFT[index]
            for index, name in enumerate(JOINTS[7:]):
                self._joints[name] = RESET_RIGHT[index]
        logger.info("Reset arms to the bridge default pose")

    def apply_arm(self, message: dict, print_every: float) -> None:
        with self._lock:
            left = message.get("left")
            right = message.get("right")
            if left is not None:
                for index, value in enumerate(left):
                    self._joints[JOINTS[index]] = float(value)
            if right is not None:
                for index, value in enumerate(right):
                    self._joints[JOINTS[7 + index]] = float(value)
            snapshot = (
                [self._joints[name] for name in JOINTS[:7]],
                [self._joints[name] for name in JOINTS[7:]],
            )
        now = time.monotonic()
        if now - self._last_print >= print_every:
            self._last_print = now
            logger.info("action left [%s] right [%s]", _format_arm(snapshot[0]), _format_arm(snapshot[1]))

    def joints(self) -> dict[str, float]:
        with self._lock:
            return dict(self._joints)


def _observation_loop(robot: SimulatedRobot, obs_socket, stop: threading.Event, hz: float) -> None:
    period = 1.0 / hz
    step = 0
    while not stop.is_set():
        started = time.monotonic()
        packet = encode_observation(
            joints=robot.joints(),
            jpeg_images=_encode_images(step),
            timestamp=time.time(),
            endpose={"left": [0.0] * 7, "right": [0.0] * 7},
        )
        try:
            obs_socket.send(packet, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        step += 1
        remaining = period - (time.monotonic() - started)
        if remaining > 0:
            stop.wait(remaining)


def _command_loop(robot: SimulatedRobot, cmd_socket, rpc_socket, stop: threading.Event, print_every: float) -> None:
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)
    poller.register(rpc_socket, zmq.POLLIN)
    while not stop.is_set():
        events = dict(poller.poll(50))
        if rpc_socket in events:
            message = decode_command(rpc_socket.recv_string())
            kind = message.get("type")
            if kind == "reset":
                robot.reset()
                reply = {"ok": True}
            elif kind == "continuous":
                robot.apply_arm(message, print_every)
                reply = {"ok": True}
            else:
                reply = {"ok": False, "error": f"unknown rpc {kind}"}
            rpc_socket.send_string(encode_command(reply))
        if cmd_socket in events:
            message = decode_command(cmd_socket.recv_string())
            if message.get("type") == "arm":
                robot.apply_arm(message, print_every)
            else:
                logger.warning("Ignored command type %s", message.get("type"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    robot = SimulatedRobot()
    robot.reset()
    context = zmq.Context()
    cmd_socket = context.socket(zmq.PULL)
    cmd_socket.setsockopt(zmq.CONFLATE, 1)
    cmd_socket.setsockopt(zmq.LINGER, 0)
    cmd_socket.bind(f"tcp://{args.bind_host}:{args.cmd_port}")
    obs_socket = context.socket(zmq.PUSH)
    obs_socket.setsockopt(zmq.CONFLATE, 1)
    obs_socket.setsockopt(zmq.LINGER, 0)
    obs_socket.bind(f"tcp://{args.bind_host}:{args.obs_port}")
    rpc_socket = context.socket(zmq.REP)
    rpc_socket.setsockopt(zmq.LINGER, 0)
    rpc_socket.bind(f"tcp://{args.bind_host}:{args.rpc_port}")
    logger.info(
        "Simulated robot listening on cmd=%s obs=%s rpc=%s",
        args.cmd_port,
        args.obs_port,
        args.rpc_port,
    )
    stop = threading.Event()
    observer = threading.Thread(
        target=_observation_loop,
        args=(robot, obs_socket, stop, args.obs_hz),
        name="sim-obs",
        daemon=True,
    )
    observer.start()
    try:
        _command_loop(robot, cmd_socket, rpc_socket, stop, args.print_every)
    except KeyboardInterrupt:
        logger.info("Stopping simulated robot")
    finally:
        stop.set()
        observer.join(timeout=2.0)
        cmd_socket.close(linger=0)
        obs_socket.close(linger=0)
        rpc_socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
