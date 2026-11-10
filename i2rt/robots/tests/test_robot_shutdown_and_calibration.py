"""Shutdown, startup and calibration of the hardware robot: nothing outlives it, nothing fakes a limit.

* ``get_yam_robot`` must close a chain whose robot it could not finish building (an error, or
  Ctrl+C during enable and calibration), or the chain's non-daemon thread keeps the process alive,
  holds the bus and keeps sending its last frame.
* ``MotorChainRobot.close(disable_gripper=True)`` disables the gripper motor by its CAN id; a plain
  close disables nothing, so an arm keeps holding its pose.
* ``detect_gripper_limits`` refuses a stopped motor chain and a sweep that saw no travel, instead of
  returning both stops at the starting angle.
"""

import threading
from types import SimpleNamespace
from typing import Any, List, cast

import numpy as np
import pytest

import i2rt.robots.get_robot as get_robot_module
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmType, GripperType, detect_gripper_limits


class _FakeChain:
    """Enough of DMChainCanInterface for get_yam_robot to get as far as building the robot."""

    def __init__(self, motor_list: List[Any], motor_offset: Any, motor_direction: Any, *_: Any, **__: Any) -> None:
        self.motor_list = motor_list
        self.motor_offset = np.array(motor_offset, dtype=float)
        self.motor_direction = np.array(motor_direction)
        self.running = True
        self.closed = False
        _FakeChain.last = self

    def __len__(self) -> int:
        return len(self.motor_list)

    def read_states(self, torques: Any = None) -> List[Any]:
        return [SimpleNamespace(pos=0.0, vel=0.0, eff=0.0) for _ in self.motor_list]

    def start_thread(self) -> None:
        pass

    def close(self) -> None:
        self.running = False
        self.closed = True


@pytest.mark.parametrize("arm", [ArmType.NO_ARM, ArmType.YAM_ULTRA_2], ids=lambda a: a.value)
@pytest.mark.parametrize("failure", [KeyboardInterrupt, RuntimeError], ids=["ctrl_c", "error"])
def test_a_robot_that_fails_to_build_closes_its_chain(
    arm: ArmType, failure: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: Any, **__: Any) -> None:
        raise failure()

    monkeypatch.setattr(get_robot_module, "DMChainCanInterface", _FakeChain)
    monkeypatch.setattr(get_robot_module, "MotorChainRobot", fail)
    with pytest.raises(failure):
        get_yam_robot(arm_type=arm, gripper_type=GripperType.LINEAR_4310)
    assert _FakeChain.last.closed, "the chain must not outlive a robot that was never built"


def _closable_robot(gripper_index: Any) -> Any:
    calls: List[Any] = []
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot._stop_event = threading.Event()
    robot._server_thread = threading.Thread(target=lambda: None)
    robot._server_thread.start()
    robot._gripper_index = gripper_index
    robot._mcap_lock = threading.Lock()
    robot._mcap_recorder = None

    class Chain:
        motor_list = [[0x01, "DM4340"], [0x07, "DM4310"]]

        def close(self, **kwargs: Any) -> None:
            calls.append(kwargs)

    robot.motor_chain = Chain()
    robot.calls = calls
    return robot


def test_close_disables_the_gripper_motor_by_its_can_id() -> None:
    robot = _closable_robot(gripper_index=1)
    robot.close(disable_gripper=True)
    assert robot.calls == [{"disable_motor_ids": [0x07]}]


def test_plain_close_disables_nothing() -> None:
    robot = _closable_robot(gripper_index=1)
    robot.close()
    assert robot.calls == [{}], "the arm (and by default the gripper) must keep its last frame"


def test_disable_gripper_on_a_gripperless_robot_is_a_plain_close() -> None:
    robot = _closable_robot(gripper_index=None)
    robot.close(disable_gripper=True)
    assert robot.calls == [{}]


class _CalibrationChain:
    """A gripper motor that slides between two stops under torque -- or never moves, or is dead."""

    def __init__(self, stuck: bool = False, running: bool = True) -> None:
        self.motor_list = [[0x07, "DM4310"]]
        self.motor_direction = np.array([1])
        self.running = running
        self.stuck = stuck
        self.pos = 0.0

    def set_commands(self, torques: Any, **_: Any) -> None:
        if not self.stuck:
            self.pos = float(np.clip(self.pos + np.sign(torques[0]) * 1.0, -3.0, 3.0))

    def read_states(self) -> List[Any]:
        return [SimpleNamespace(pos=self.pos)]


def _calibrate(chain: _CalibrationChain) -> Any:
    return detect_gripper_limits(
        chain, 0, hold_kp=np.zeros(1), hold_kd=np.zeros(1), max_duration=0.5, check_interval=0.01
    )


def test_calibration_finds_the_stops_of_a_gripper_that_moves() -> None:
    assert _calibrate(_CalibrationChain()) == (3.0, -3.0)


def test_calibration_refuses_a_gripper_that_never_moved() -> None:
    with pytest.raises(RuntimeError, match="travel"):
        _calibrate(_CalibrationChain(stuck=True))


def test_calibration_refuses_a_dead_motor_chain() -> None:
    with pytest.raises(RuntimeError, match="motor chain stopped"):
        _calibrate(_CalibrationChain(running=False))
