#!/usr/bin/env python3
"""ROS1 Python 3.8 process that exposes the AgileX cobot over ZMQ.

Run this after sourcing ROS. The Python 3.12 inference process connects with
``--robot.type=agilex_cobot_zmq`` and never imports rospy.

Observations are JPEG frames, one ZMQ message each, latest-only.
Arm targets are joint positions. If the client stops sending, this process
stops publishing. A position controller then holds the last target. Velocity
commands for the mobile base are zeroed after the watchdog expires.
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

import cv2
import rospy
import zmq

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from ros_manager import AgilexCobotROSManager  # noqa: E402
from ros_manager_config import AgilexCobotROSManagerConfig  # noqa: E402

PROTOCOL_PATH = HERE / "protocol.py"


def _load_protocol():
    import importlib.util

    spec = importlib.util.spec_from_file_location("agilex_cobot_zmq_protocol", PROTOCOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("agilex_cobot_zmq_bridge")

RESET_LEFT = [
    -0.00133514404296875,
    0.00209808349609375,
    0.01583099365234375,
    -0.032616615295410156,
    -0.00286102294921875,
    0.00095367431640625,
    0.07,
]
RESET_RIGHT = [
    -0.00133514404296875,
    0.00438690185546875,
    0.034523963928222656,
    -0.053597450256347656,
    -0.00476837158203125,
    -0.00209808349609375,
    0.07,
]


def _parse_args():
    parser = argparse.ArgumentParser(description="AgileX cobot ROS1 to ZMQ bridge")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--cmd-port", type=int, default=protocol.DEFAULT_CMD_PORT)
    parser.add_argument("--obs-port", type=int, default=protocol.DEFAULT_OBS_PORT)
    parser.add_argument("--rpc-port", type=int, default=protocol.DEFAULT_RPC_PORT)
    parser.add_argument("--obs-hz", type=float, default=30.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--watchdog-s", type=float, default=0.5)
    parser.add_argument("--with-mobile-base", action="store_true")
    parser.add_argument("--ros-timeout-s", type=float, default=30.0)
    return parser.parse_args()


def _encode_images(cameras, quality):
    encoded = {}
    for name, image in cameras.items():
        if image is None:
            continue
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if ok:
            encoded[name] = buffer.tobytes()
    return encoded


def _observation_loop(ros, obs_socket, stop, args):
    period = 1.0 / args.obs_hz
    while not stop.is_set() and not rospy.is_shutdown():
        started = time.monotonic()
        try:
            joints, cameras, endpose = ros.get_synchronized_observation()
            packet = protocol.encode_observation(
                joints=joints,
                jpeg_images=_encode_images(cameras, args.jpeg_quality),
                timestamp=time.time(),
                endpose=endpose,
            )
            try:
                obs_socket.send(packet, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
        except Exception as exc:
            logger.warning("Failed to build an observation: %s", exc)
        remaining = period - (time.monotonic() - started)
        if remaining > 0:
            stop.wait(remaining)


def _apply_arm_command(ros, message):
    left = message.get("left")
    right = message.get("right")
    base = message.get("base")
    if left is not None:
        ros.publish_left_arm_command([float(value) for value in left])
    if right is not None:
        ros.publish_right_arm_command([float(value) for value in right])
    if base is not None:
        ros.publish_mobile_base_command([float(value) for value in base])


def _apply_rpc(ros, message):
    kind = message.get("type")
    if kind == "reset":
        ros.publish_continuous_arm_commands(list(RESET_LEFT), list(RESET_RIGHT))
        return {"ok": True}
    if kind == "continuous":
        ros.publish_continuous_arm_commands(message.get("left"), message.get("right"))
        return {"ok": True}
    return {"ok": False, "error": f"unknown rpc {kind}"}


def _command_loop(ros, cmd_socket, rpc_socket, stop, args):
    poller = zmq.Poller()
    poller.register(cmd_socket, zmq.POLLIN)
    poller.register(rpc_socket, zmq.POLLIN)
    last_command = None
    watchdog_latched = False
    while not stop.is_set() and not rospy.is_shutdown():
        events = dict(poller.poll(50))
        if rpc_socket in events:
            raw = rpc_socket.recv_string()
            try:
                reply = _apply_rpc(ros, protocol.decode_command(raw))
            except Exception as exc:
                logger.exception("RPC command failed")
                reply = {"ok": False, "error": str(exc)}
            rpc_socket.send_string(protocol.encode_command(reply))
            last_command = time.monotonic()
            watchdog_latched = False
        if cmd_socket in events:
            try:
                message = protocol.decode_command(cmd_socket.recv_string())
                if message.get("type") != "arm":
                    raise ValueError(f"unknown command {message.get('type')}")
                _apply_arm_command(ros, message)
                last_command = time.monotonic()
                watchdog_latched = False
            except Exception as exc:
                logger.warning("Ignored a bad arm command: %s", exc)
        if (
            args.with_mobile_base
            and last_command is not None
            and not watchdog_latched
            and time.monotonic() - last_command > args.watchdog_s
        ):
            logger.warning("No command for %.2fs. Stopping the mobile base.", args.watchdog_s)
            ros.publish_mobile_base_command([0.0, 0.0])
            watchdog_latched = True
        elif (
            last_command is not None
            and not watchdog_latched
            and not args.with_mobile_base
            and time.monotonic() - last_command > args.watchdog_s
        ):
            logger.info("No command for %.2fs. Holding the last arm target.", args.watchdog_s)
            watchdog_latched = True


def main():
    import zmq

    args = _parse_args()
    ros_config = AgilexCobotROSManagerConfig()
    ros_config.with_mobile_base = args.with_mobile_base
    ros = AgilexCobotROSManager(ros_config)

    logger.info("Waiting up to %.0fs for joint states and cameras", args.ros_timeout_s)
    deadline = time.time() + args.ros_timeout_s
    while not ros.is_connected() and time.time() < deadline and not rospy.is_shutdown():
        time.sleep(0.1)
    if not ros.is_connected():
        logger.warning("ROS topics are not all ready. The bridge will keep trying to publish observations.")

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
        "ZMQ bridge listening on cmd=%s obs=%s rpc=%s",
        args.cmd_port,
        args.obs_port,
        args.rpc_port,
    )
    stop = threading.Event()
    observer = threading.Thread(
        target=_observation_loop, args=(ros, obs_socket, stop, args), name="zmq-obs", daemon=True
    )
    observer.start()
    try:
        _command_loop(ros, cmd_socket, rpc_socket, stop, args)
    finally:
        stop.set()
        observer.join(timeout=2.0)
        cmd_socket.close(linger=0)
        obs_socket.close(linger=0)
        rpc_socket.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
