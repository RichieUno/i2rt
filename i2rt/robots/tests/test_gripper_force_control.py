"""Gripper torque/force control: the pure torque law, and how ``MotorChainRobot`` routes it.

The law (``gripper_force_control.py``) is exercised directly, edge by edge. The robot side is driven
through ``update()`` on a fixture built with ``__new__`` -- the idiom of ``test_torque_modes.py`` --
with the frame that would have gone to the motors captured instead of sent.
"""

import logging
import threading
from types import SimpleNamespace
from typing import Any, List, Tuple, cast

import numpy as np
import pytest

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.gripper_force_control import (
    GripperForceControlParams,
    GripperForceRequest,
    closing_sign,
    estimate_grip_force,
    gripper_force_torque,
)
from i2rt.robots.motor_chain_robot import JointCommands, MotorChainRobot
from i2rt.robots.utils import ArmType, GripperForceLimiter, GripperType, JointMapper

PARAMS = GripperForceControlParams(max_force=50.0, max_torque=1.5, wall_margin=0.05, max_speed=2.0)
# linear_4310's map: torque = force * gripper_stroke / motor_stroke.
NM_PER_N = 0.096 / 6.57
FORCE_GRIPPERS = [GripperType.LINEAR_4310, GripperType.LINEAR_3507, GripperType.FLEXIBLE_4310, GripperType.CRANK_4310]


def _linear_map(gripper_force: float, current_angle: float) -> float:
    return gripper_force * NM_PER_N


# Both orders detect_gripper_limits can produce: [closed, open] with closed above or below open.
LIMIT_ORDERS = [np.array([2.0, -4.57]), np.array([-4.57, 2.0])]


def _torque(request: GripperForceRequest, s: float = 0.5, s_dot: float = 0.0, limits: Any = None) -> float:
    limits = LIMIT_ORDERS[0] if limits is None else limits
    raw = limits[0] + s * (limits[1] - limits[0])
    return gripper_force_torque(request, s, s_dot, raw, limits, _linear_map, PARAMS)


# ---- the torque law ---------------------------------------------------------


@pytest.mark.parametrize("limits", LIMIT_ORDERS, ids=["closed_high", "closed_low"])
def test_positive_force_turns_the_motor_toward_the_closed_stop(limits: np.ndarray) -> None:
    tau = _torque(GripperForceRequest(force_n=20.0), limits=limits)
    toward_closed = np.sign(limits[0] - limits[1])
    assert np.sign(tau) == toward_closed
    assert np.sign(_torque(GripperForceRequest(force_n=-20.0), limits=limits)) == -toward_closed
    assert closing_sign(limits) == toward_closed


def test_force_maps_through_the_gripper_force_torque_map() -> None:
    tau = _torque(GripperForceRequest(force_n=20.0))
    assert abs(tau) == pytest.approx(20.0 * NM_PER_N)


def test_force_is_clamped_to_max_force() -> None:
    assert abs(_torque(GripperForceRequest(force_n=1e4))) == pytest.approx(PARAMS.max_force * NM_PER_N)


def test_torque_is_clamped_to_max_torque() -> None:
    assert abs(_torque(GripperForceRequest(torque_nm=10.0))) == pytest.approx(PARAMS.max_torque)
    assert abs(_torque(GripperForceRequest(torque_nm=-10.0))) == pytest.approx(PARAMS.max_torque)


def test_wall_tapers_torque_toward_the_closed_stop() -> None:
    close = GripperForceRequest(torque_nm=1.0)
    m = PARAMS.wall_margin
    assert abs(_torque(close, s=m)) == pytest.approx(1.0)
    assert abs(_torque(close, s=m / 2)) == pytest.approx(0.5)
    assert _torque(close, s=0.0) == 0.0
    assert _torque(close, s=-0.02) == 0.0, "past the stop, torque toward it must be refused"


def test_wall_tapers_torque_toward_the_open_stop() -> None:
    open_ = GripperForceRequest(torque_nm=-1.0)
    m = PARAMS.wall_margin
    assert abs(_torque(open_, s=1.0 - m / 2)) == pytest.approx(0.5)
    assert _torque(open_, s=1.0) == 0.0
    assert _torque(open_, s=1.02) == 0.0


def test_torque_away_from_a_stop_always_passes() -> None:
    """A closed gripper must always be able to open, and an open one to close."""
    assert abs(_torque(GripperForceRequest(torque_nm=-1.0), s=0.0)) == pytest.approx(1.0)
    assert abs(_torque(GripperForceRequest(torque_nm=-1.0), s=-0.02)) == pytest.approx(1.0)
    assert abs(_torque(GripperForceRequest(torque_nm=1.0), s=1.0)) == pytest.approx(1.0)


def test_speed_limit_fades_torque_along_the_motion() -> None:
    close = GripperForceRequest(torque_nm=1.0)
    v = PARAMS.max_speed
    # Closing drives s down, so a negative s_dot is motion along the closing torque.
    assert abs(_torque(close, s_dot=-v)) == pytest.approx(1.0)
    assert abs(_torque(close, s_dot=-1.5 * v)) == pytest.approx(0.5)
    assert _torque(close, s_dot=-3.0 * v) == 0.0
    # Against the motion it is braking, and passes in full.
    assert abs(_torque(close, s_dot=3.0 * v)) == pytest.approx(1.0)


def test_estimate_inverts_the_commanded_force_mid_stroke() -> None:
    for limits in LIMIT_ORDERS:
        raw = float(np.mean(limits))
        tau = gripper_force_torque(GripperForceRequest(force_n=30.0), 0.5, 0.0, raw, limits, _linear_map, PARAMS)
        assert estimate_grip_force(tau, raw, limits, _linear_map) == pytest.approx(30.0)


def test_request_needs_exactly_one_finite_value() -> None:
    with pytest.raises(ValueError):
        GripperForceRequest()
    with pytest.raises(ValueError):
        GripperForceRequest(force_n=1.0, torque_nm=1.0)
    with pytest.raises(ValueError):
        GripperForceRequest(force_n=float("nan"))


@pytest.mark.parametrize(
    "bad", [{"max_force": 0.0}, {"max_torque": -1.0}, {"wall_margin": 0.0}, {"wall_margin": 0.5}, {"max_speed": 0}]
)
def test_params_reject_a_degenerate_envelope(bad: dict) -> None:
    with pytest.raises(ValueError):
        GripperForceControlParams(**bad)


# ---- config -----------------------------------------------------------------


@pytest.mark.parametrize("gripper", FORCE_GRIPPERS, ids=lambda g: g.value)
def test_every_motorized_gripper_ships_a_force_control_envelope(gripper: GripperType) -> None:
    params = gripper.get_force_control_params(ArmType.YAM)
    assert params is not None
    force_map = gripper.get_gripper_limiter_params(ArmType.YAM)[3]
    # The whole force range must fit the torque clamp, or max_force would be a lie at mid-stroke.
    assert abs(force_map(gripper_force=params.max_force, current_angle=0.5)) <= params.max_torque


@pytest.mark.parametrize("gripper", [GripperType.NO_GRIPPER, GripperType.YAM_TEACHING_HANDLE_V1])
def test_motorless_end_effectors_have_no_force_control(gripper: GripperType) -> None:
    assert gripper.get_force_control_params(ArmType.YAM) is None


def test_crank_map_depends_on_position() -> None:
    force_map = GripperType.CRANK_4310.get_gripper_limiter_params(ArmType.YAM)[3]
    assert force_map(gripper_force=10.0, current_angle=-0.5) != pytest.approx(
        force_map(gripper_force=10.0, current_angle=-1.5)
    )


# ---- the robot --------------------------------------------------------------

N = 2  # one arm joint + the gripper
GI = 1
RAW_LIMITS = np.array([2.0, -4.57])  # [closed, open]


def _robot(s: float = 0.5, force_params: Any = PARAMS) -> Any:
    """A one-joint arm with a gripper, carrying only what update() and the command methods touch."""
    robot = cast(Any, MotorChainRobot.__new__(MotorChainRobot))
    robot.motor_chain = [None] * N
    robot._arm_type = ArmType.YAM
    robot._gripper_type = GripperType.LINEAR_4310
    robot._kp = np.array([80.0, 20.0])
    robot._kd = np.array([5.0, 0.5])
    robot._grav_comp_kd = np.zeros(N)
    robot.use_gravity_comp = False
    robot.use_coulomb_friction = False
    robot._coulomb_friction = np.zeros(N)
    robot.gravity_comp_factor = np.ones(N)
    robot._clip_motor_torque = np.inf
    robot._joint_limits = None
    robot._gripper_index = GI
    robot._gripper_limits = RAW_LIMITS
    robot.remapper = JointMapper({GI: RAW_LIMITS}, N)
    robot._limit_gripper_force = 50.0
    robot._gripper_force_limiter = GripperForceLimiter(
        max_force=50.0, gripper_type=GripperType.LINEAR_4310, arm_type=ArmType.YAM, kp=20.0
    )
    robot._gripper_force_params = force_params
    robot._gripper_force_map = _linear_map
    robot._last_gripper_command_qpos = 1
    robot._joint_state = SimpleNamespace(pos=np.array([0.1, s]), vel=np.zeros(N), eff=np.zeros(N))
    robot._command_lock = threading.Lock()
    robot._state_lock = threading.Lock()
    robot._commands = JointCommands.init_all_zero(N)
    return robot


def _step(robot: Any) -> Tuple[np.ndarray, JointCommands]:
    """One update(), returning the torques and command frame that would have gone to the motors."""
    sent: List[Tuple[np.ndarray, JointCommands]] = []
    robot._update_joint_state = lambda torques, commands: sent.append((torques, commands))
    robot.update()
    return sent[0]


def _state(gripper: float = 0.5, **extra: Any) -> dict:
    return {"pos": np.array([0.3, gripper]), "vel": np.zeros(N), **extra}


def test_force_command_torque_controls_the_gripper_and_leaves_the_arm_alone() -> None:
    robot = _robot()
    robot.command_joint_state(_state(gripper_force=20.0))
    torques, commands = _step(robot)

    assert commands.kp[GI] == 0.0, "the gripper's position loop must be off"
    assert commands.kd[GI] == pytest.approx(0.5), "kd falls back to the gripper's own"
    assert torques[GI] == pytest.approx(20.0 * NM_PER_N * closing_sign(RAW_LIMITS))
    assert commands.kp[0] == 80.0 and commands.pos[0] == pytest.approx(0.3), "the arm stays position-controlled"
    assert torques[0] == 0.0
    # The target sits at the current position, inside the stops.
    assert commands.pos[GI] == pytest.approx(float(np.mean(RAW_LIMITS)))


def test_torque_command_goes_through_the_same_guards() -> None:
    robot = _robot(s=0.0)  # at the closed stop
    robot.command_joint_state(_state(gripper_torque=1.0))
    torques, _ = _step(robot)
    assert torques[GI] == 0.0, "the wall must refuse closing torque at the closed stop"


def test_the_controller_does_not_modify_the_stored_command() -> None:
    """update() deep-copies the command, so the gains it zeroes are not the robot's own."""
    robot = _robot()
    robot.command_joint_state(_state(gripper_force=20.0))
    _step(robot)
    assert robot._kp[GI] == 20.0


def test_position_command_after_force_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    robot = _robot()
    robot.command_joint_state(_state(gripper_force=20.0))
    robot.command_joint_state(_state(gripper_force=25.0))
    assert not caplog.records, "force to force is not a mode switch"

    with caplog.at_level(logging.WARNING):
        robot.command_joint_pos(np.array([0.3, 0.5]))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "leaves force control" in warnings[0].getMessage()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        robot.command_joint_pos(np.array([0.3, 0.5]))
        robot.command_joint_state(_state())
    assert not caplog.records, "position to position is not a mode switch"


def test_command_joint_state_without_the_key_also_warns(caplog: pytest.LogCaptureFixture) -> None:
    robot = _robot()
    robot.command_joint_state(_state(gripper_force=20.0))
    with caplog.at_level(logging.WARNING):
        robot.command_joint_state(_state())
    assert any("command_joint_state" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("idle", ["enter_gravity_comp_idle", "enter_zero_torque_idle", "zero_torque_mode"])
def test_deliberate_idle_exits_force_control_quietly(idle: str, caplog: pytest.LogCaptureFixture) -> None:
    robot = _robot()
    robot._gravity_comp_available = False
    robot._coulomb_friction_configured = False
    robot.command_joint_state(_state(gripper_force=20.0))
    with caplog.at_level(logging.WARNING):
        getattr(robot, idle)()
        robot.command_joint_pos(np.array([0.3, 0.5]))
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_leaving_force_control_resets_the_position_limiter() -> None:
    robot = _robot()
    robot.command_joint_state(_state(gripper_force=20.0))
    _step(robot)
    robot._gripper_force_limiter._is_clogged = True  # stale state from before the force stretch
    robot.command_joint_pos(np.array([0.3, 0.5]))
    _, commands = _step(robot)
    assert robot._gripper_force_limiter._is_clogged is False
    assert commands.kp[GI] == 20.0, "position mode is back, with the gripper's gains"


def test_position_mode_still_clips_the_target_to_the_stops() -> None:
    robot = _robot()
    robot.command_joint_pos(np.array([0.3, 1.5]))  # past fully open
    _, commands = _step(robot)
    assert min(RAW_LIMITS) <= commands.pos[GI] <= max(RAW_LIMITS)


def test_force_and_torque_together_are_rejected() -> None:
    with pytest.raises(ValueError):
        _robot().command_joint_state(_state(gripper_force=1.0, gripper_torque=0.1))


def test_force_without_a_force_control_config_is_rejected() -> None:
    with pytest.raises(ValueError, match="force_control"):
        _robot(force_params=None).command_joint_state(_state(gripper_force=1.0))


def test_force_on_a_robot_without_a_gripper_is_rejected() -> None:
    robot = _robot()
    robot._gripper_index = None
    with pytest.raises(ValueError, match="no gripper"):
        robot.command_joint_state(_state(gripper_force=1.0))


def test_force_estimate_reads_the_measured_effort() -> None:
    robot = _robot()
    robot._joint_state.eff[GI] = 30.0 * NM_PER_N * closing_sign(RAW_LIMITS)
    assert robot.get_gripper_force_estimate() == pytest.approx(30.0)


def test_sim_robot_refuses_rather_than_ignores_a_force() -> None:
    robot = get_yam_robot(arm_type=ArmType.YAM, gripper_type=GripperType.LINEAR_4310, sim=True)
    try:
        q = robot.get_joint_pos()
        with pytest.raises(NotImplementedError):
            robot.command_joint_state({"pos": q, "vel": np.zeros_like(q), "gripper_force": 10.0})
    finally:
        robot.close()


# ---- non-finite commands ------------------------------------------------------------


@pytest.mark.parametrize("field", ["pos", "vel", "kp", "kd"])
def test_a_non_finite_command_is_refused_and_the_last_one_kept(field: str) -> None:
    robot = _robot()
    robot.command_joint_state(_state(gripper_force=20.0))
    before = robot._commands
    bad = _state(gripper_force=20.0)
    bad[field] = np.array([np.nan, 0.5]) if field in ("pos", "vel") else np.array([80.0, np.inf])
    with pytest.raises(ValueError, match="finite"):
        robot.command_joint_state(bad)
    assert robot._commands is before, "the robot must keep executing the last valid command"


def test_a_non_finite_position_command_is_refused() -> None:
    robot = _robot()
    robot.command_joint_pos(np.array([0.3, 0.5]))
    before = robot._commands
    with pytest.raises(ValueError, match="finite"):
        robot.command_joint_pos(np.array([np.nan, 0.5]))
    assert robot._commands is before


# ---- the force map over the calibrated stroke ------------------------------------------------

from i2rt.motor_drivers.utils import MotorInfo

NOMINAL_NM_PER_N = 0.096 / 6.57  # linear_4310's limiter block: gripper_stroke / motor_stroke


def test_force_control_divides_the_jaw_travel_by_the_calibrated_stroke() -> None:
    force_map = GripperType.LINEAR_4310.get_force_control_map(ArmType.YAM, calibrated_stroke=5.245)
    assert force_map(gripper_force=1.0, current_angle=0.0) == pytest.approx(0.096 / 5.245)


@pytest.mark.parametrize("stroke", [None, 0.0], ids=["none", "zero"])
def test_without_a_calibrated_stroke_the_config_map_stands(stroke: Any) -> None:
    force_map = GripperType.LINEAR_4310.get_force_control_map(ArmType.YAM, calibrated_stroke=stroke)
    assert force_map(gripper_force=1.0, current_angle=0.0) == pytest.approx(NOMINAL_NM_PER_N)


@pytest.mark.parametrize("stroke", [2.0, 12.0], ids=["blocked_short", "too_long"])
def test_an_implausible_stroke_keeps_the_config_map(stroke: float, caplog: pytest.LogCaptureFixture) -> None:
    """A calibration blocked short would raise the torque per newton; it must not reach force control."""
    with caplog.at_level(logging.WARNING):
        force_map = GripperType.LINEAR_4310.get_force_control_map(ArmType.YAM, calibrated_stroke=stroke)
    assert force_map(gripper_force=1.0, current_angle=0.0) == pytest.approx(NOMINAL_NM_PER_N)
    assert any("far from the nominal" in r.getMessage() for r in caplog.records)


def test_a_crank_gripper_keeps_its_own_map() -> None:
    config_map = GripperType.CRANK_4310.get_gripper_limiter_params(ArmType.YAM)[3]
    force_map = GripperType.CRANK_4310.get_force_control_map(ArmType.YAM, calibrated_stroke=2.8)
    for angle in (-0.5, -1.5, -2.5):
        assert force_map(gripper_force=10.0, current_angle=angle) == pytest.approx(
            config_map(gripper_force=10.0, current_angle=angle)
        )


class _GripperChain:
    """One gripper motor that holds still: enough for MotorChainRobot to build and run its loop."""

    def __init__(self) -> None:
        self.motor_list = [[0x07, "DM4310"]]
        self.motor_direction = np.array([1])
        self.running = True

    def __len__(self) -> int:
        return 1

    def read_states(self, torques: Any = None) -> List[MotorInfo]:
        return [MotorInfo(id=0x07, error_code=1, pos=-2.4, vel=0.0, eff=0.0, timestamp=0.0)]

    def set_commands(self, torques: Any, **_: Any) -> List[MotorInfo]:
        return self.read_states()

    def close(self, **_: Any) -> None:
        self.running = False


def test_position_mode_keeps_the_config_map_while_force_control_uses_the_calibration() -> None:
    limits = np.array([0.20237, -5.04215])  # the calibrated [closed, open] of the measured linear_4310
    robot = MotorChainRobot(
        motor_chain=cast(Any, _GripperChain()),
        xml_path=GripperType.LINEAR_4310.get_xml_path(),
        use_gravity_comp=False,
        kp=[20.0],
        kd=[0.5],
        gripper_index=0,
        gripper_limits=limits,
        limit_gripper_force=50.0,
        gripper_type=GripperType.LINEAR_4310,
        arm_type=ArmType.YAM,
        zero_gravity_mode=False,
    )
    try:
        stroke = abs(limits[1] - limits[0])
        assert robot._gripper_force_map(gripper_force=1.0, current_angle=0.0) == pytest.approx(0.096 / stroke)
        # The position-mode limiter, built from the limiter block, is untouched: 50 N at the nominal ratio.
        limiter_torque = robot._gripper_force_limiter.gripper_force_torque_map(current_angle=0.0)
        assert limiter_torque == pytest.approx(50.0 * NOMINAL_NM_PER_N)
    finally:
        robot.close()
