#!/usr/bin/env python3
"""End-to-end dry run: ZMQ sim + remote FineART language/action."""
from __future__ import annotations

import time
from uuid import uuid4

import numpy as np

from lerobot.inference import ExecutionMode, ObservationSnapshot
from lerobot.inference.execution import ChunkRequest
from lerobot.inference.factory import RemoteInferenceConfig
from lerobot.policies.rtc.action_queue import QueueSnapshot
from lerobot.remote_inference.client import RemoteClient
from lerobot.robots.agilex_cobot_zmq.agilex_cobot_zmq import AgilexCobotZmq
from lerobot.robots.agilex_cobot_zmq.config_agilex_cobot_zmq import AgilexCobotZmqConfig


def _obs_from_robot(robot: AgilexCobotZmq, features, task: str) -> ObservationSnapshot:
    raw = robot.get_observation()
    mapped = {}
    for feature in features:
        if feature.kind == "rgb":
            name = feature.name.removeprefix("observation.images.")
            mapped[feature.name] = np.asarray(raw[name], dtype=np.uint8)
        else:
            # joint order follows feature.names
            mapped[feature.name] = np.asarray(
                [float(raw[name]) for name in feature.names], dtype=np.float32
            )
    return ObservationSnapshot(mapped, time.time(), task, 1, uuid4().hex)


def main() -> None:
    robot = AgilexCobotZmq(AgilexCobotZmqConfig(remote_ip="127.0.0.1", id="zip_tie"))
    robot.connect(calibrate=True)
    cfg = RemoteInferenceConfig(
        endpoint="tcp/127.0.0.1:7447",
        deployment="zip_tie",
        mode="rtc_trained",
        semantics="agilex-bimanual-joint-rad-v1",
        handshake_timeout_s=30,
        language_timeout_s=120,
        action_timeout_s=180,
    )
    client = RemoteClient.connect(cfg)
    features = client.capabilities.features
    client.admit(
        features=features,
        action_feature=client.capabilities.action_feature,
        semantics=cfg.semantics,
        action_interval=1 / 30,
        mode=cfg.mode,
    )
    print("admitted", client.session_id)

    # On blank sim cameras only the short warmup-style goal stays readable.
    goal = "Describe the next task."
    subtask = client.query_language(
        _obs_from_robot(robot, features, ""),
        kind="next_subtask",
        text=goal,
        intent_generation=1,
        generation=client.generation,
        cancelled=lambda: False,
    )
    print("LANGUAGE goal", repr(goal), "->", repr(subtask))
    if sum(ch.isascii() and ch.isprintable() for ch in subtask) / max(len(subtask), 1) < 0.7:
        subtask = "pick up scissors"
        print("LANGUAGE fallback subtask", repr(subtask))

    obs = _obs_from_robot(robot, features, subtask)
    request = ChunkRequest(
        observation=obs,
        continuation=QueueSnapshot(
            generation=client.generation,
            cursor=0,
            index=0,
            model_actions=None,
            canonical_actions=None,
            provenance=(),
        ),
        request_id=uuid4().hex,
        generation=client.generation,
        mode=ExecutionMode.RTC_TRAINED,
        delay=0,
        submitted_at=time.monotonic(),
    )
    started = time.time()
    chunk = client.infer(request, cancelled=lambda: False)
    elapsed = time.time() - started
    actions = chunk.canonical_actions
    print("ACTION shape", tuple(actions.shape), f"turnaround={elapsed:.2f}s")
    print("ACTION[0]", np.round(actions[0], 3).tolist())

    # Second chunk with trained delay + continuation, as the RTC client does.
    follow = ChunkRequest(
        observation=_obs_from_robot(robot, features, subtask),
        continuation=QueueSnapshot(
            generation=client.generation,
            cursor=8,
            index=0,
            model_actions=chunk.model_actions,
            canonical_actions=chunk.canonical_actions,
            provenance=(),
        ),
        request_id=uuid4().hex,
        generation=client.generation,
        mode=ExecutionMode.RTC_TRAINED,
        delay=8,
        submitted_at=time.monotonic(),
    )
    started = time.time()
    chunk2 = client.infer(follow, cancelled=lambda: False)
    print(
        "ACTION2 shape",
        tuple(chunk2.canonical_actions.shape),
        f"turnaround={time.time() - started:.2f}s",
    )

    names = list(client.capabilities.action_feature.names)
    for step in actions[:8]:
        robot.send_action({name: float(value) for name, value in zip(names, step, strict=True)})
        time.sleep(1 / 30)
    print("sent 8 actions to the ZMQ sim")

    client.close()
    robot.disconnect()
    print("PASS")


if __name__ == "__main__":
    main()
