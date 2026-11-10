"""The gripper force-control example's input safeguards: what a typed force may do."""

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "gripper_force_control" / "gripper_force_control.py"


def _example() -> Any:
    # The example imports its sibling hardware_session module, as it does when run as a script.
    if str(EXAMPLE.parent) not in sys.path:
        sys.path.insert(0, str(EXAMPLE.parent))
    spec = importlib.util.spec_from_file_location("gripper_force_control_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), -float("inf"), None, "-", ""])
def test_a_partly_typed_force_keeps_the_last_good_value(raw: Any) -> None:
    assert _example().sanitize_force(raw, 12.0, 20.0) == (12.0, False)


def test_a_typed_force_is_clamped_to_the_slider_range() -> None:
    ex = _example()
    assert ex.sanitize_force(500.0, 0.0, 20.0) == (20.0, True)
    assert ex.sanitize_force(-500.0, 0.0, 20.0) == (-20.0, True)
    assert ex.sanitize_force("7.5", 0.0, 20.0) == (7.5, True)


def test_pushing_harder_is_ramped() -> None:
    ramp = _example().ramp_force
    assert ramp(5.0, 15.0, 2.0) == 7.0
    assert ramp(-5.0, -15.0, 2.0) == -7.0
    assert ramp(0.0, 50.0, 0.8) == 0.8, "one keystroke must not land 50 N at once"


def test_backing_off_is_immediate() -> None:
    ramp = _example().ramp_force
    assert ramp(15.0, 5.0, 2.0) == 5.0
    assert ramp(15.0, 0.0, 2.0) == 0.0, "Zero force must act at once"
    # A reversal releases first, then ramps up the other way.
    assert ramp(15.0, -10.0, 2.0) == -2.0


# ---- the 3-D view ---------------------------------------------------------------------

import asyncio
import socket
import threading
import time
from typing import Dict

import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import ArmType, GripperType


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def _hold_client(port: int, seconds: float, result: Dict[str, Any]) -> None:
    """Keep a viser client connected; the server dropping it means a message it could not encode."""
    viser = pytest.importorskip("viser")
    websockets = pytest.importorskip("websockets")

    async def run() -> None:
        # max_size=None: the arm meshes arrive as one ~2 MB message; a browser has no such cap.
        uri = f"ws://localhost:{port}"
        async with websockets.connect(uri, subprotocols=[f"viser-v{viser.__version__}"], max_size=None) as ws:
            result["connected"] = True
            deadline = time.time() + seconds
            while time.time() < deadline:
                try:
                    await asyncio.wait_for(ws.recv(), 0.1)
                except asyncio.TimeoutError:
                    pass

    try:
        asyncio.run(run())
    except Exception as e:
        result["error"] = repr(e)


@pytest.mark.parametrize("arm", [ArmType.YAM_ULTRA_2, ArmType.NO_ARM, ArmType.YAM_7DOF_LEFT], ids=lambda a: a.value)
def test_the_view_follows_the_robot_with_a_client_connected(arm: ArmType) -> None:
    viser = pytest.importorskip("viser")
    port = _free_port()
    server = viser.ViserServer(port=port, verbose=False)
    # The robot sim reports joint positions exactly as the hardware robot does, which is all the view reads.
    robot = get_yam_robot(arm_type=arm, gripper_type=GripperType.LINEAR_4310, sim=True)
    result: Dict[str, Any] = {}
    try:
        view = _example()._RobotView(server, robot, close_up=arm == ArmType.NO_ARM)
        assert view._geoms, "the view should draw the robot's meshes"
        client = threading.Thread(target=_hold_client, args=(port, 2.0, result))
        client.start()
        deadline = time.time() + 3.0
        while not result.get("connected") and time.time() < deadline:
            time.sleep(0.02)

        before = {g: view._handles[g].position.copy() for g in view._geoms}
        q = robot.get_joint_pos().copy()
        q[: len(q) - 1] += 0.3  # every arm joint (none on a gripper alone)
        q[-1] = 1.0 - q[-1]  # and the gripper across its stroke
        robot.command_joint_pos(q)
        for _ in range(20):
            view.mirror()
            time.sleep(0.02)
        moved = [g for g in view._geoms if not np.allclose(view._handles[g].position, before[g])]
        assert moved, "the meshes should follow the robot's joints"
        client.join()
    finally:
        robot.close()
        server.stop()
    assert result.get("connected"), "the client never connected"
    assert "error" not in result, f"the server dropped the client: {result['error']}"
