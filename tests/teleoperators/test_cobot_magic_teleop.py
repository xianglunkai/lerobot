#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest

from lerobot.teleoperators.cobot_magic_teleop import (
    COBOTMAGIC_L_ARM_JOINTS,
    COBOTMAGIC_R_ARM_JOINTS,
    COBOTMAGIC_VEL,
    CobotMagicTeleoperator,
    CobotMagicTeleoperatorConfig,
)

# {lerobot_keys: cobotmagic_sdk_keys}
COBOTMAGIC_JOINTS = {
    **COBOTMAGIC_L_ARM_JOINTS,
    **COBOTMAGIC_R_ARM_JOINTS,
}

PARAMS = [
    {},  # default config
    {"with_mobile_base": False},
    {"with_mobile_base": False, "with_l_arm": False},
    {"with_r_arm": False},
    {"use_present_position": True},
]


def _make_cobotmagic_sdk_mock():
    r = MagicMock(name="CobotMagicSDKMock")
    r.is_connected.return_value = True

    def _connect():
        r.is_connected.return_value = True

    def _disconnect():
        r.is_connected.return_value = False

    # Mock joints with some dummy positions
    joints = {
        k: MagicMock(
            present_position=float(i),
            goal_position=float(i) + 0.5,
        )
        for i, k in enumerate(COBOTMAGIC_JOINTS.values())
    }
    r.joints = joints

    # Mock mobile base with some dummy odometry
    r.mobile_base = MagicMock()
    r.mobile_base.last_cmd_vel = {
        "vx": -0.2,
        "vy": 0.2,
        "vtheta": 11.0,
    }
    r.mobile_base.odometry = {
        "x": 1.0,
        "y": 2.0,
        "theta": 20.0,
        "vx": 0.1,
        "vy": -0.1,
        "vtheta": 8.0,
    }

    r.connect = MagicMock(side_effect=_connect)
    r.disconnect = MagicMock(side_effect=_disconnect)

    return r


@pytest.fixture(params=PARAMS, ids=lambda p: "default" if not p else ",".join(p.keys()))
def cobotmagic(request):
    with (
        patch(
            "lerobot.teleoperators.cobot_magic_teleop.cobot_magic_teleoperator.CobotMagicSDK",
            side_effect=lambda *a, **k: _make_cobotmagic_sdk_mock(),
        ),
    ):
        overrides = request.param
        cfg = CobotMagicTeleoperatorConfig(ip_address="192.168.0.200", **overrides)
        robot = CobotMagicTeleoperator(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(cobotmagic):
    """测试连接和断开功能"""
    assert not cobotmagic.is_connected

    cobotmagic.connect()
    assert cobotmagic.is_connected

    cobotmagic.disconnect()
    assert not cobotmagic.is_connected

    cobotmagic.cobotmagic.disconnect.assert_called_once()


def test_get_action(cobotmagic):
    """测试获取动作状态功能"""
    cobotmagic.connect()
    action = cobotmagic.get_action()

    expected_keys = set(cobotmagic.joints_dict)
    expected_keys.update(f"{v}" for v in COBOTMAGIC_VEL if cobotmagic.config.with_mobile_base)
    assert set(action.keys()) == expected_keys

    for motor in cobotmagic.joints_dict:
        if cobotmagic.config.use_present_position:
            assert action[motor] == cobotmagic.cobotmagic.joints[COBOTMAGIC_JOINTS[motor]].present_position
        else:
            assert action[motor] == cobotmagic.cobotmagic.joints[COBOTMAGIC_JOINTS[motor]].goal_position
    if cobotmagic.config.with_mobile_base:
        if cobotmagic.config.use_present_position:
            for vel in COBOTMAGIC_VEL:
                assert action[vel] == cobotmagic.cobotmagic.mobile_base.odometry[COBOTMAGIC_VEL[vel]]
        else:
            for vel in COBOTMAGIC_VEL:
                assert action[vel] == cobotmagic.cobotmagic.mobile_base.last_cmd_vel[COBOTMAGIC_VEL[vel]]


def test_no_part_declared():
    """测试未声明任何部件时的异常情况"""
    with pytest.raises(ValueError):
        _ = CobotMagicTeleoperatorConfig(
            ip_address="192.168.0.200",
            with_mobile_base=False,
            with_l_arm=False,
            with_r_arm=False,
        )


def test_mobile_base_state_callback(cobotmagic):
    """测试移动底座状态回调功能"""
    cobotmagic.connect()
    
    # 模拟Twist消息
    msg = MagicMock()
    msg.linear.x = 0.5
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 1.0
    
    # 调用回调函数
    cobotmagic._mobile_base_state_callback(msg)
    
    # 验证状态更新
    with cobotmagic._state_lock:
        assert cobotmagic._mobile_base_state is not None
        assert cobotmagic._mobile_base_state.linear["x"] == 0.5
        assert cobotmagic._mobile_base_state.linear["y"] == 0.0
        assert cobotmagic._mobile_base_state.linear["z"] == 0.0
        assert cobotmagic._mobile_base_state.angular["x"] == 0.0
        assert cobotmagic._mobile_base_state.angular["y"] == 0.0
        assert cobotmagic._mobile_base_state.angular["z"] == 1.0
        assert cobotmagic._mobile_base_state.last_update > 0


def test_mobile_base_state_callback_with_null_values(cobotmagic):
    """测试移动底座状态回调功能（空值情况）"""
    cobotmagic.connect()
    
    # 模拟Twist消息（空值）
    msg = MagicMock()
    msg.linear.x = 0.0
    msg.linear.y = 0.0
    msg.linear.z = 0.0
    msg.angular.x = 0.0
    msg.angular.y = 0.0
    msg.angular.z = 0.0
    
    # 调用回调函数
    cobotmagic._mobile_base_state_callback(msg)
    
    # 验证状态更新
    with cobotmagic._state_lock:
        assert cobotmagic._mobile_base_state is not None
        assert cobotmagic._mobile_base_state.linear["x"] == 0.0
        assert cobotmagic._mobile_base_state.linear["y"] == 0.0
        assert cobotmagic._mobile_base_state.linear["z"] == 0.0
        assert cobotmagic._mobile_base_state.angular["x"] == 0.0
        assert cobotmagic._mobile_base_state.angular["y"] == 0.0
        assert cobotmagic._mobile_base_state.angular["z"] == 0.0
        assert cobotmagic._mobile_base_state.last_update > 0
