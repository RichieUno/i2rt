"""Torque/force control of the gripper motor, with the hard-stop guards it needs.

The gripper motor runs in MIT mode like every other motor, so torque control needs no mode change:
with the gripper's ``kp`` at zero the frame's feedforward torque *is* the command. What it does
need is a new way to keep the jaws off their mechanical stops. Position mode does that by clipping
the position target to the calibrated limits, and with ``kp = 0`` the target is ignored -- that
clip still runs, but it no longer limits anything. The guards here act on the torque instead:

* **Force clamp** -- the requested force is clamped to ``max_force`` before it is mapped to a
  torque, and the resulting torque to ``max_torque`` after, so a map that misbehaves near a
  singularity cannot ask for more than the motor is allowed.
* **Virtual wall** -- inside ``wall_margin`` of a stop, torque pushing *toward* that stop tapers
  linearly to zero at the stop, and is zero past it. Torque pushing away always passes, so a
  closed gripper can always open.
* **Speed limit** -- torque pushing *along* the current motion fades from full at ``max_speed``
  to zero at twice it, so jaws closing on nothing do not reach the wall at full speed.

Everything is computed in the normalized gripper space the robot already exposes (``s = 0``
closed, ``s = 1`` open, per the calibrated limits), so one law covers every gripper type and both
calibrated limit orders. Sign convention throughout: **a positive force or torque closes**.

The function is pure -- no threads, no CAN -- so every edge is unit-testable without hardware.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# The force -> motor-torque map from the gripper config: called with ``gripper_force`` (N) and
# ``current_angle`` (raw motor rad). See ``GripperType.get_gripper_limiter_params``.
ForceTorqueMap = Callable[..., float]


@dataclass(frozen=True)
class GripperForceControlParams:
    """Safety envelope for gripper torque control, from the gripper config's ``force_control`` block."""

    max_force: float = 50.0  # N, the largest force a command may ask for
    max_torque: float = 1.5  # Nm at the motor, applied after the force map
    wall_margin: float = 0.05  # normalized stroke, the taper zone at each stop
    max_speed: float = 2.0  # normalized stroke / s; torque along the motion is zero at twice this
    kd: Optional[float] = None  # MIT damping on the gripper motor; None = the gripper's own kd
    # m of jaw travel between the *calibrated* stops, for a linear force map; None = the limiter's
    # gripper_stroke. Force control only: position mode's force limiter keeps the limiter block's.
    gripper_stroke: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.max_force > 0 or not self.max_torque > 0:
            raise ValueError(f"max_force and max_torque must be positive, got {self}")
        if not 0 < self.wall_margin < 0.5:
            raise ValueError(f"wall_margin must be in (0, 0.5), got {self.wall_margin}")
        if not self.max_speed > 0:
            raise ValueError(f"max_speed must be positive, got {self.max_speed}")
        if self.kd is not None and self.kd < 0:
            raise ValueError(f"kd must be non-negative, got {self.kd}")
        if self.gripper_stroke is not None and not self.gripper_stroke > 0:
            raise ValueError(f"gripper_stroke must be positive, got {self.gripper_stroke}")


@dataclass(frozen=True)
class GripperForceRequest:
    """A gripper command in force (N) or motor-torque (Nm) units. Exactly one is set; positive closes."""

    force_n: Optional[float] = None
    torque_nm: Optional[float] = None

    def __post_init__(self) -> None:
        if (self.force_n is None) == (self.torque_nm is None):
            raise ValueError("give exactly one of gripper_force and gripper_torque")
        value = self.force_n if self.force_n is not None else self.torque_nm
        if not np.isfinite(value):
            raise ValueError(f"gripper force/torque must be finite, got {value}")


def closing_sign(raw_limits: np.ndarray) -> float:
    """The sign of a motor torque that closes the gripper, from the calibrated ``[closed, open]`` limits.

    ``detect_gripper_limits`` orders the pair by motor direction, so the closed end can be the
    larger or the smaller raw angle; the sign has to come from the pair, not from a constant.
    """
    closed, opened = float(raw_limits[0]), float(raw_limits[1])
    if closed == opened:
        raise ValueError(f"gripper limits have zero stroke: {raw_limits}")
    return float(np.sign(closed - opened))


def torque_per_newton(force_torque_map: ForceTorqueMap, raw_pos: float) -> float:
    """Motor torque (Nm) per newton of grip force at this motor angle. Constant on a linear gripper."""
    return abs(float(force_torque_map(gripper_force=1.0, current_angle=raw_pos)))


def gripper_force_torque(
    request: GripperForceRequest,
    s: float,
    s_dot: float,
    raw_pos: float,
    raw_limits: np.ndarray,
    force_torque_map: ForceTorqueMap,
    params: GripperForceControlParams,
) -> float:
    """The gripper motor torque, in the robot's raw motor frame, for one control cycle.

    Args:
        request: The commanded force or torque; positive closes.
        s: Gripper position, normalized (0 closed, 1 open).
        s_dot: Gripper velocity, normalized units per second.
        raw_pos: Gripper motor angle (rad), for the position-dependent crank map.
        raw_limits: Calibrated ``[closed, open]`` motor angles.
        force_torque_map: The gripper's force -> torque map.
        params: The safety envelope.

    Returns:
        The feedforward torque to send with ``kp = 0``.
    """
    if request.force_n is not None:
        force = float(np.clip(request.force_n, -params.max_force, params.max_force))
        closing = np.sign(force) * abs(force) * torque_per_newton(force_torque_map, raw_pos)
    else:
        closing = float(request.torque_nm)
    closing = float(np.clip(closing, -params.max_torque, params.max_torque))

    # Virtual wall: taper whatever pushes toward the stop it is heading for.
    if closing > 0:
        closing *= float(np.clip(s / params.wall_margin, 0.0, 1.0))
    elif closing < 0:
        closing *= float(np.clip((1.0 - s) / params.wall_margin, 0.0, 1.0))

    # Speed limit: closing drives s down, so it pushes along the motion when s is falling.
    if closing * -s_dot > 0:
        closing *= float(np.clip(2.0 - abs(s_dot) / params.max_speed, 0.0, 1.0))

    return closing * closing_sign(raw_limits)


def estimate_grip_force(
    raw_eff: float,
    raw_pos: float,
    raw_limits: np.ndarray,
    force_torque_map: ForceTorqueMap,
) -> float:
    """The grip force (N, positive closing) implied by the gripper motor's measured torque.

    The inverse of the force map, so it is only as good as the map: it includes friction and,
    while the jaws move, the damping torque. It is meaningful at a stall on an object.
    """
    per_newton = torque_per_newton(force_torque_map, raw_pos)
    if per_newton < 1e-9:
        return 0.0
    return float(raw_eff) * closing_sign(raw_limits) / per_newton
