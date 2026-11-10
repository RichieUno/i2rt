"""The examples' ``hardware_session``: on every exit the gripper is released and then disabled.

Ctrl+C and SIGTERM take the same path: zero torque first, then ``close(disable_gripper=True)``, which
also works when the control loop has already died, since the bus is still open.
"""

import os
import signal
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, cast

import numpy as np
import pytest

from i2rt.robots.motor_chain_robot import MotorChainRobot

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "gripper_force_control"))
import hardware_session


def _fake_robot(log: List[str]) -> Any:
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))  # passes isinstance, opens nothing
    robot.motor_chain = SimpleNamespace(running=True)
    robot.get_joint_pos = lambda: np.array([0.3, 0.5])
    sent: Dict[str, Any] = {}

    def command(state: Dict[str, Any]) -> None:
        sent.update(state)
        log.append("released")

    robot.command_joint_state = command
    robot.close = lambda disable_gripper=False: log.append("closed, gripper disabled" if disable_gripper else "closed")
    robot.sent = sent
    return robot


def test_the_gripper_is_released_before_closing_on_ctrl_c() -> None:
    log: List[str] = []
    robot = _fake_robot(log)
    with pytest.raises(KeyboardInterrupt):
        with hardware_session.hardware_session(lambda: robot):
            raise KeyboardInterrupt
    assert log == ["released", "closed, gripper disabled"]
    assert float(np.asarray(robot.sent["gripper_torque"]).reshape(-1)[0]) == 0.0, "let go: zero torque"
    assert np.allclose(robot.sent["pos"], [0.3, 0.5]), "the arm is held where it is"


def test_sigterm_takes_the_same_clean_exit() -> None:
    log: List[str] = []
    before = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt):
        with hardware_session.hardware_session(lambda: _fake_robot(log)):
            os.kill(os.getpid(), signal.SIGTERM)
            threading.Event().wait(1.0)  # the signal arrives here
    assert log == ["released", "closed, gripper disabled"]
    assert signal.getsignal(signal.SIGTERM) is before, "the previous SIGTERM handler must be restored"


def test_a_dead_chain_is_still_closed_without_a_release() -> None:
    log: List[str] = []
    robot = _fake_robot(log)
    robot.motor_chain.running = False
    with hardware_session.hardware_session(lambda: robot):
        pass
    assert log == ["closed, gripper disabled"], "a dead loop's bus is still open, so the gripper is still disabled"
