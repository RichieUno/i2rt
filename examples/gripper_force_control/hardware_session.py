"""Start and stop a hardware gripper session so the motor is let go and the process can exit.

Used by ``gripper_force_control.py`` and ``characterize_gripper.py``.
Three things go wrong otherwise:

* ``close()`` only stops sending. The motor keeps executing its last frame until its CAN timeout
  (if one is set), so a gripper last told to squeeze keeps squeezing -- and a gripper motor left
  enabled makes the *next* launch fail ("disabled" on its first control frame). So the session
  sends zero torque first (:func:`release_gripper`), then closes with the gripper motor disabled,
  which also works when the control loop has already died.
* ``kill`` (SIGTERM) skips every ``finally``; it is turned into the same clean exit as Ctrl+C.
* A second Ctrl+C during shutdown interrupts it half done; signals are ignored until it finishes.

A Ctrl+C during startup (motor checks, enable, the gripper calibration) is handled by
``get_yam_robot`` itself, which closes the chain it was building.
"""

import signal
import time
from contextlib import contextmanager
from typing import Any, Callable, Generator

import numpy as np

from i2rt.robots.motor_chain_robot import MotorChainRobot


def _interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def chain_alive(robot: MotorChainRobot) -> bool:
    """True while the motor chain's control loop is still running."""
    return bool(getattr(robot.motor_chain, "running", False))


def release_gripper(robot: MotorChainRobot, settle_s: float = 0.15) -> None:
    """Let go of the gripper: zero torque and no position hold, with the arm held where it is.

    This is the last frame the motors keep executing once the process stops sending, so it decides
    what the robot does after exit: the gripper goes limp, the arm holds its pose.
    """
    if not chain_alive(robot):
        print("[session] motor chain already stopped; disabling the gripper motor directly")
        return
    q = robot.get_joint_pos().copy()
    robot.command_joint_state({"pos": q, "vel": np.zeros_like(q), "gripper_torque": np.array([0.0])})
    time.sleep(settle_s)  # several control cycles, so the frame reaches the motor


@contextmanager
def hardware_session(build: Callable[[], Any]) -> Generator[MotorChainRobot, None, None]:
    """Build the robot, yield it, and on any exit release the gripper and close the robot."""
    previous_term = signal.signal(signal.SIGTERM, _interrupt)
    try:
        try:
            robot = build()
        except KeyboardInterrupt:
            print("[session] interrupted during startup; the motor chain was closed")
            raise
        if not isinstance(robot, MotorChainRobot):
            robot.close()
            raise SystemExit("this needs a motorized gripper on real hardware")
        try:
            yield robot
        finally:
            # Finish shutting down even if Ctrl+C is pressed again or the process is told to terminate.
            previous_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("[session] releasing the gripper and closing; please wait")
            try:
                release_gripper(robot)
            except Exception as e:  # never let a failed release skip the close
                print(f"[session] could not release the gripper: {e}")
            finally:
                robot.close(disable_gripper=True)
                signal.signal(signal.SIGINT, previous_int)
    finally:
        signal.signal(signal.SIGTERM, previous_term)
