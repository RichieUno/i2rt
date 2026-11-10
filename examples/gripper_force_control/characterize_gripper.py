"""Characterize the gripper's dynamics under torque control: breakaway, friction, damping, inertia.

Runs two experiments on real hardware, logging every sample to CSV, then fits a model:

1. **Breakaway** -- from rest at several stroke positions, in both directions, repeated: ramp the
   motor torque slowly from zero until the jaws move. The torque at that moment is the static
   (breakaway) friction, and its spread over positions, directions, dwell and repeats is what makes
   the threshold vary.
2. **Torque steps** -- from one end of the free stroke, apply a constant torque and run to the other
   end. The steady speed against torque is the line

       omega = (tau - tau_c) / (b + kd)

   whose slope gives the viscous damping ``b`` (``kd`` is the known MIT damping force mode adds) and
   whose intercept gives the kinetic (moving) Coulomb friction ``tau_c``. The rise to that speed
   has the time constant ``T = J / (b + kd)``, which gives the inertia ``J``.

3. **Breakaway sweep** (``--experiments sweep``) -- the breakaway ramp at many positions in both
   directions, each approached from the far side so the mechanism's slack is already taken up in
   the test direction. Each position's two thresholds split into a **friction** (their mean) and a
   position-dependent **offset** (half their difference; positive = something pushes the jaws
   open), against stroke position and motor angle -- the shape the first experiment's three
   positions cannot show, and whether the offset repeats with motor angle (cogging) or not.
4. **Jaw travel** (``--experiments jaw_travel``) -- the jaws are held at a few positions and you
   type in the gap you measure (mm, between the jaw faces). The measured mm per motor radian is the
   real force-to-torque ratio; it is compared with the gripper config's force map, and corrected
   ``gripper_stroke`` / ``motor_stroke`` values are suggested.

All torques are motor torques in the closing-positive convention of ``command_joint_state``'s
``gripper_torque``, and they go through the same guards as any force command (clamps, virtual wall,
speed limit). Positions stay clear of the wall zones, and a trial that hits the speed limit is
flagged, since the limit changes the torque that was actually applied.

**Keep the jaws free -- nothing between them -- for the whole run.** The gripper moves on its own
between trials (repositioning in position mode). Ctrl+C at any time releases the gripper and exits.

Output, in ``<out>/<gripper>_<timestamp>/``: ``samples.csv`` (every sample of every trial),
``trials.csv`` (one row per trial with its derived values), ``meta.json`` (setup and limits) and
``fit.json`` (the fitted model).

Usage:
    python examples/gripper_force_control/characterize_gripper.py --channel can0
    python examples/gripper_force_control/characterize_gripper.py --channel can0 --arm yam_ultra_2
    python examples/gripper_force_control/characterize_gripper.py --channel can0 --step-torques 0.2 0.3 0.4
    python examples/gripper_force_control/characterize_gripper.py --channel can0 --experiments sweep jaw_travel
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import csv
import itertools
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import tyro
from hardware_session import chain_alive, hardware_session

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmName, ArmType, GripperType

Experiment = Literal["breakaway", "steps", "sweep", "jaw_travel"]

# ---- analysis: pure functions on logged samples ---------------------------------------


FIRST_MOVEMENT_TOL = 5e-4  # normalized: the first sample off the start, a ~1e-3 twitch included
SUSTAINED_TOL = 3e-3  # normalized: past such a twitch, where the jaws move for good


def breakaway_index(s: np.ndarray, threshold: float, onset_tol: float = FIRST_MOVEMENT_TOL) -> Optional[int]:
    """The sample where the jaws started to move, or None if they never left their start by ``threshold``.

    Triggers on ``threshold`` (robust to sensor noise), then walks back to the last sample still
    within ``onset_tol`` of the start -- about the encoder's noise (5e-4 of a ~6.6 rad stroke is
    ~9 counts). Reading the torque where the jaws had already travelled ``threshold`` would overstate
    breakaway by whatever the ramp added while they travelled it.
    """
    offset = np.abs(s - s[0])
    moved = np.nonzero(offset > threshold)[0]
    if not len(moved):
        return None
    still = np.nonzero(offset[: moved[0]] <= onset_tol)[0]
    return int(still[-1]) if len(still) else int(moved[0])


def steady_speed(t: np.ndarray, x: np.ndarray, window: Tuple[float, float]) -> Optional[float]:
    """Mean speed (dx/dt by a least-squares line) while ``x`` lies inside ``window``; None if too few samples."""
    lo, hi = window
    inside = (x >= lo) & (x <= hi)
    if inside.sum() < 5:
        return None
    slope, _ = np.polyfit(t[inside], x[inside], 1)
    return float(slope)


def rise_time_constant(t: np.ndarray, v: np.ndarray, v_steady: float) -> Optional[float]:
    """Time constant ``T`` of the rise to ``v_steady``, fitted to ``v = v_steady (1 - exp(-(t - t0) / T))``.

    Fitted over the whole rise rather than read off two threshold crossings: ``T = J / (b + kd)`` can
    be a few milliseconds, only a couple of samples, and a least-squares fit of the curve resolves it
    below the sample period. A grid search over ``T`` and the motion's start ``t0``, numpy only.
    """
    if v_steady <= 0:
        return None
    moving = np.nonzero(v > 0.05 * v_steady)[0]
    settled = np.nonzero(v >= 0.95 * v_steady)[0]
    if not len(moving) or not len(settled):
        return None
    dt = float(np.median(np.diff(t)))
    stop = min(len(t), settled[0] + 1 + max(5, settled[0] - moving[0]))  # the rise plus as much plateau
    tt, vv = t[:stop], v[:stop]
    best = (np.inf, None)
    for t0 in np.arange(t[moving[0]] - 3 * dt, t[moving[0]] + dt, dt / 20):
        elapsed = np.clip(tt - t0, 0.0, None)
        for T in np.geomspace(1e-4, 2.0, 400):
            err = float(np.sum((vv - v_steady * (1 - np.exp(-elapsed / T))) ** 2))
            if err < best[0]:
                best = (err, float(T))
    return best[1]


def fit_speed_line(torques: Sequence[float], speeds: Sequence[float], kd: float) -> Dict[str, float]:
    """Fit ``omega = (tau - tau_c) / (b + kd)`` to steady speeds; returns tau_c, b and the fit residual."""
    tau = np.asarray(torques, dtype=float)
    omega = np.asarray(speeds, dtype=float)
    if len(tau) < 2:
        raise ValueError("need steady speeds at two torques or more to fit the speed line")
    slope, intercept = np.polyfit(tau, omega, 1)  # omega = slope * tau + intercept
    damping = 1.0 / slope
    tau_c = -intercept * damping
    residual = float(np.sqrt(np.mean((omega - (slope * tau + intercept)) ** 2)))
    return {
        "tau_c_kinetic": float(tau_c),
        "b_viscous": float(damping - kd),
        "total_damping": float(damping),
        "rms_residual_rad_s": residual,
    }


def decompose_breakaway(close_nm: Sequence[float], open_nm: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Split paired breakaway thresholds into friction and a position-dependent offset torque.

    Friction resists motion either way; an offset ``g`` (a spring, a cable, cogging, gravity on the
    jaws) adds to one direction and subtracts from the other. With closing positive,
    ``close = friction + g`` and ``open = friction - g``, so ``g > 0`` pushes the jaws open.
    """
    c, o = np.asarray(close_nm, dtype=float), np.asarray(open_nm, dtype=float)
    return (c + o) / 2.0, (c - o) / 2.0


def dominant_period(angle: np.ndarray, values: np.ndarray, shortest: float) -> Optional[Dict[str, float]]:
    """The sinusoid period (rad of motor angle) that best explains ``values``, and how much it explains.

    A least-squares fit of ``a sin + b cos + c`` over candidate periods from ``shortest`` (twice the
    sample spacing: anything faster aliases) to the full span. A large ``variance_explained`` at a
    short period points at cogging; a long one at a slow mechanical effect.
    """
    if len(values) < 6:
        return None
    angle, values = np.asarray(angle, dtype=float), np.asarray(values, dtype=float)
    span = float(np.ptp(angle))
    total = float(np.sum((values - values.mean()) ** 2))
    if total == 0.0 or span <= shortest:
        return None
    best: Optional[Dict[str, float]] = None
    for period in np.geomspace(shortest, span, 300):
        k = 2 * np.pi / period
        basis = np.column_stack([np.sin(k * angle), np.cos(k * angle), np.ones_like(angle)])
        coef, *_ = np.linalg.lstsq(basis, values, rcond=None)
        explained = 1.0 - float(np.sum((values - basis @ coef) ** 2)) / total
        if best is None or explained > best["variance_explained"]:
            best = {
                "period_rad": float(period),
                "amplitude_nm": float(np.hypot(coef[0], coef[1])),
                "variance_explained": explained,
            }
    return best


@dataclass
class JawPoint:
    target_s: float
    s: float  # where the jaws actually were, normalized
    motor_rad: float  # ... as a motor angle
    gap_mm: float  # what was measured between the jaw faces


def analyze_jaw_travel(
    points: List[JawPoint], stroke_rad: float, force_map: Callable[..., float], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Measured mm per motor radian against the force map's own ratio.

    By virtual work ``tau * dtheta = F * dx``, so the force-to-torque ratio *is* the jaw travel per
    motor radian: the map is right exactly when ``|dx/dtheta|`` matches ``map(F=1)``. Compared per
    segment between neighbouring points, which holds for a position-dependent (crank) map as well.
    """
    pts = sorted(points, key=lambda p: p.motor_rad)
    angle = np.array([p.motor_rad for p in pts])
    gap_m = np.array([p.gap_mm for p in pts]) / 1000.0
    slope, intercept = np.polyfit(angle, gap_m, 1)
    linearity = float(np.sqrt(np.mean((gap_m - (slope * angle + intercept)) ** 2))) * 1000.0
    segments = []
    for a, b in itertools.pairwise(pts):
        measured = abs((b.gap_mm - a.gap_mm) / 1000.0 / (b.motor_rad - a.motor_rad))
        mid = 0.5 * (a.motor_rad + b.motor_rad)
        assumed = abs(float(force_map(gripper_force=1.0, current_angle=mid)))
        segments.append(
            {
                "from_s": a.s,
                "to_s": b.s,
                "measured_nm_per_n": measured,
                "map_nm_per_n": assumed,
                "map_error": assumed / measured - 1.0 if measured else None,
            }
        )
    measured_ratio = abs(float(slope))
    result: Dict[str, Any] = {
        "points": [asdict(p) for p in pts],
        "mm_per_rad": measured_ratio * 1000.0,
        "travel_over_calibrated_stroke_mm": measured_ratio * stroke_rad * 1000.0,
        "linearity_rms_mm": linearity,
        "segments": segments,
        "config": config,
    }
    if config.get("force_torque_map") == "linear":
        assumed = float(config["gripper_stroke"]) / float(config["motor_stroke"])
        result.update(
            measured_nm_per_n=measured_ratio,
            config_nm_per_n=assumed,
            # > 0: the config's map asks for too much torque per newton, so real forces run high.
            force_map_error=assumed / measured_ratio - 1.0,
            suggested={"gripper_stroke": round(measured_ratio * stroke_rad, 4), "motor_stroke": round(stroke_rad, 3)},
        )
    return result


# ---- the experiment ----------------------------------------------------------------------


@dataclass
class Setup:
    gi: int
    n: int
    closed: float  # calibrated raw motor angle at the closed stop (rad)
    opened: float  # ... and at the open stop
    kd: float  # MIT damping force mode applies (Nm s/rad)
    max_speed: float  # normalized stroke/s where the speed limit starts cutting torque
    hold: np.ndarray = field(repr=False)  # the arm's pose, held throughout

    @property
    def stroke(self) -> float:
        """Full stroke in motor radians."""
        return abs(self.opened - self.closed)


@dataclass
class Trial:
    kind: str  # "breakaway", "sweep" (a breakaway approached from the far side) or "step"
    direction: str  # "close" or "open"
    start_s: float
    repeat: int
    torque: float = 0.0  # step: the applied torque; breakaway: the ramp's ceiling
    samples: List[Tuple[float, float, float, float, float]] = field(default_factory=list)  # t, s, s_dot, eff, cmd
    result: Dict[str, Any] = field(default_factory=dict)


class _ForceControlSwitchFilter(logging.Filter):
    """Silence the robot's "leaves force control" warning: here every switch back is deliberate."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "leaves force control" not in record.getMessage()


def _command(robot: MotorChainRobot, setup: Setup, s: Optional[float] = None, torque: Optional[float] = None) -> None:
    """Position mode at normalized ``s``, or torque mode at ``torque`` (Nm, closing positive)."""
    q = setup.hold.copy()
    state: Dict[str, Any] = {"vel": np.zeros(setup.n)}
    if torque is None:
        q[setup.gi] = s
    else:
        q[setup.gi] = robot.get_joint_pos()[setup.gi]
        state["gripper_torque"] = np.array([torque])
    state["pos"] = q
    robot.command_joint_state(state)


def _sample(robot: MotorChainRobot, setup: Setup, t0: float, cmd: float) -> Tuple[float, float, float, float, float]:
    obs = robot.get_observations()
    closing = float(np.sign(setup.closed - setup.opened))
    eff_closing = float(obs["gripper_eff"][0]) * closing  # measured motor torque, closing positive
    return (time.perf_counter() - t0, float(obs["gripper_pos"][0]), float(obs["gripper_vel"][0]), eff_closing, cmd)


def _move_to(robot: MotorChainRobot, setup: Setup, target: float, duration: float = 1.0, settle: float = 0.5) -> None:
    """Reposition smoothly in position mode, then let it settle."""
    start = float(robot.get_joint_pos()[setup.gi])
    t0 = time.perf_counter()
    while (elapsed := time.perf_counter() - t0) < duration:
        alpha = 0.5 - 0.5 * np.cos(np.pi * elapsed / duration)
        _command(robot, setup, s=start + alpha * (target - start))
        time.sleep(0.005)
    _command(robot, setup, s=target)
    time.sleep(settle)


def _check(robot: MotorChainRobot) -> None:
    if not chain_alive(robot):
        raise RuntimeError("the motor chain stopped (see the first error above)")


def run_breakaway(
    robot: MotorChainRobot,
    setup: Setup,
    trial: Trial,
    ramp_rate: float,
    dwell: float,
    threshold: float,
    approach: float = 0.0,
) -> None:
    """Ramp torque from zero until the jaws move ``threshold`` (normalized) from rest, then stop.

    With ``approach``, the jaws first go ``approach`` beyond the start on the far side, so the last
    move into the start is in the test direction and the mechanism's slack is already taken up.
    """
    sign = 1.0 if trial.direction == "close" else -1.0
    if approach:
        _move_to(robot, setup, trial.start_s + sign * approach)  # closing lowers s: come from above
    _move_to(robot, setup, trial.start_s)
    # Rest at zero torque first: static friction grows with dwell, so dwell is a fixed part of the test.
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < dwell:
        _command(robot, setup, torque=0.0)
        trial.samples.append(_sample(robot, setup, t0, 0.0))
        time.sleep(0.002)
    t_ramp = time.perf_counter()
    while True:
        _check(robot)
        cmd = sign * min(ramp_rate * (time.perf_counter() - t_ramp), trial.torque)
        _command(robot, setup, torque=cmd)
        trial.samples.append(sample := _sample(robot, setup, t0, cmd))
        if abs(sample[1] - trial.samples[0][1]) > threshold:
            break
        if abs(cmd) >= trial.torque and time.perf_counter() - t_ramp > trial.torque / ramp_rate + 1.0:
            break  # never broke away below the ceiling
        time.sleep(0.002)
    _command(robot, setup, torque=0.0)
    arr = np.array(trial.samples)
    # Two readings: on the first hardware runs one trial in six first twitched ~0.1 % of the stroke
    # (taking up slack), held, and needed 0.02-0.10 Nm more to move for good. The breakaway is the
    # sustained one; the first movement is kept alongside.
    first = breakaway_index(arr[:, 1], threshold, FIRST_MOVEMENT_TOL)
    sustained = breakaway_index(arr[:, 1], threshold, SUSTAINED_TOL)
    trial.result = {
        "breakaway_torque": abs(float(arr[sustained, 4])) if sustained is not None else None,
        "first_movement_torque": abs(float(arr[first, 4])) if first is not None else None,
        "broke_away": sustained is not None,
    }


def run_step(robot: MotorChainRobot, setup: Setup, trial: Trial, timeout: float) -> None:
    """Constant torque from one end of the free stroke to the other."""
    sign = 1.0 if trial.direction == "close" else -1.0
    end = 0.12 if trial.direction == "close" else 0.88  # stop before the wall zone
    _move_to(robot, setup, trial.start_s)
    t0 = time.perf_counter()
    cmd = sign * trial.torque
    while time.perf_counter() - t0 < timeout:
        _check(robot)
        _command(robot, setup, torque=cmd)
        trial.samples.append(sample := _sample(robot, setup, t0, cmd))
        if (trial.direction == "close" and sample[1] <= end) or (trial.direction == "open" and sample[1] >= end):
            break
        time.sleep(0.002)
    _command(robot, setup, torque=0.0)
    arr = np.array(trial.samples)
    t, s = arr[:, 0], arr[:, 1]
    # In motor rad/s along the commanded direction, from position (cleaner than the velocity feedback).
    x = (s if trial.direction == "open" else -s) * setup.stroke
    window = (
        (0.3 * setup.stroke, 0.7 * setup.stroke)
        if trial.direction == "open"
        else (-0.7 * setup.stroke, -0.3 * setup.stroke)
    )
    omega = steady_speed(t, x, window)
    v = (arr[:, 2] if trial.direction == "open" else -arr[:, 2]) * setup.stroke
    time_constant = rise_time_constant(t, v, omega) if omega else None
    trial.result = {
        "steady_speed_rad_s": omega,
        "steady_speed_stroke_s": omega / setup.stroke if omega is not None else None,
        "time_constant_s": time_constant,
        # How many samples the rise spans: below ~3, read the fitted inertia as an order of magnitude.
        "rise_samples": time_constant / float(np.median(np.diff(t))) if time_constant else None,
        "speed_limited": bool(np.any(np.abs(arr[:, 2]) > setup.max_speed)),
        "reached_end": bool((s[-1] <= end) if trial.direction == "close" else (s[-1] >= end)),
    }


def run_jaw_travel(
    robot: MotorChainRobot, setup: Setup, targets: Sequence[float], ask: Callable[[str], str] = input
) -> List[JawPoint]:
    """Hold the jaws at each target and record the gap the operator measures there (mm)."""
    points = []
    for target in targets:
        _move_to(robot, setup, target)
        s_now = float(robot.get_joint_pos()[setup.gi])
        while True:
            answer = ask(f"[char] jaws held at {s_now:.3f} of the stroke. Measured gap between the jaw faces (mm): ")
            try:
                gap = float(answer)
            except ValueError:
                print("[char] not a number; try again")
                continue
            if gap < 0:
                print("[char] a gap cannot be negative; try again")
                continue
            break
        points.append(JawPoint(target, s_now, setup.closed + s_now * (setup.opened - setup.closed), gap))
    return points


def analyze_sweep(trials: List[Trial], setup: Setup) -> Optional[Dict[str, Any]]:
    """Friction and offset against position, from positions measured in both directions."""
    by_pos: Dict[float, Dict[str, List[float]]] = {}
    for t in trials:
        if t.kind == "sweep" and t.result.get("broke_away"):
            by_pos.setdefault(t.start_s, {"close": [], "open": []})[t.direction].append(t.result["breakaway_torque"])
    paired = sorted(p for p, v in by_pos.items() if v["close"] and v["open"])
    if not paired:
        return None
    close = [float(np.mean(by_pos[p]["close"])) for p in paired]
    open_ = [float(np.mean(by_pos[p]["open"])) for p in paired]
    friction, offset = decompose_breakaway(close, open_)
    angle = np.array([setup.closed + p * (setup.opened - setup.closed) for p in paired])
    spacing = float(np.median(np.diff(np.sort(angle)))) if len(angle) > 1 else 0.0
    return {
        "positions": paired,
        "motor_angle_rad": angle.tolist(),
        "close_nm": close,
        "open_nm": open_,
        "friction_nm": friction.tolist(),
        "offset_nm": offset.tolist(),
        "friction_range_nm": [float(friction.min()), float(friction.max())],
        "offset_range_nm": [float(offset.min()), float(offset.max())],
        "offset_period": dominant_period(angle, offset, 2 * abs(spacing)) if spacing else None,
        "friction_period": dominant_period(angle, friction, 2 * abs(spacing)) if spacing else None,
    }


def analyze(trials: List[Trial], setup: Setup) -> Dict[str, Any]:
    """Fit the model: breakaway statistics, and tau_c / b / J from the torque steps."""
    fit: Dict[str, Any] = {"breakaway": {}, "steps": {}}
    for direction in ("close", "open"):
        values = [
            t.result["breakaway_torque"]
            for t in trials
            if t.kind == "breakaway" and t.direction == direction and t.result.get("broke_away")
        ]
        if values:
            fit["breakaway"][direction] = {
                "mean_nm": float(np.mean(values)),
                "std_nm": float(np.std(values)),
                "min_nm": float(np.min(values)),
                "max_nm": float(np.max(values)),
                "n": len(values),
            }
        # Only runs that crossed the stroke: one that stalled part-way (below the breakaway it met
        # there) has no steady speed, and its stub of motion bends the line.
        usable = [
            t
            for t in trials
            if t.kind == "step"
            and t.direction == direction
            and t.result.get("steady_speed_rad_s")
            and not t.result.get("speed_limited")
            and t.result.get("reached_end")
        ]
        if len(usable) >= 2:
            line = fit_speed_line(
                [t.torque for t in usable], [t.result["steady_speed_rad_s"] for t in usable], setup.kd
            )
            taus = [t.result["time_constant_s"] for t in usable if t.result.get("time_constant_s")]
            if taus:
                line["time_constant_s"] = float(np.median(taus))
                line["inertia_kg_m2"] = float(np.median(taus) * line["total_damping"])
            line["n"] = len(usable)
            fit["steps"][direction] = line
    sweep = analyze_sweep(trials, setup)
    if sweep is not None:
        fit["sweep"] = sweep
    return fit


def _limiter_config(gripper_type: GripperType, arm_type: ArmType) -> Dict[str, Any]:
    """The gripper config's limiter block, which holds the force map's stroke parameters."""
    from i2rt.robots.utils import _load_gripper_config

    arm = ArmType.YAM if arm_type == ArmType.NO_ARM else arm_type
    return dict(_load_gripper_config(gripper_type.value, arm).limiter_params or {})


def _save(out: Path, meta: Dict[str, Any], trials: List[Trial], fit: Dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    (out / "fit.json").write_text(json.dumps(fit, indent=2))
    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "kind", "direction", "t_s", "s", "s_dot_per_s", "eff_closing_nm", "cmd_closing_nm"])
        for i, tr in enumerate(trials):
            for row in tr.samples:
                w.writerow([i, tr.kind, tr.direction, *(f"{v:.6g}" for v in row)])
    with open(out / "trials.csv", "w", newline="") as f:
        keys = sorted({k for tr in trials for k in tr.result})
        w = csv.writer(f)
        w.writerow(["trial", "kind", "direction", "start_s", "repeat", "torque_nm", *keys])
        for i, tr in enumerate(trials):
            w.writerow([i, tr.kind, tr.direction, tr.start_s, tr.repeat, tr.torque, *(tr.result.get(k) for k in keys)])


def main(
    arm: ArmName = "no_arm",
    gripper: str = "linear_4310",
    channel: str = "can0",
    out: Path = Path("gripper_characterization"),
    breakaway_positions: Tuple[float, ...] = (0.3, 0.5, 0.7),
    repeats: int = 3,
    ramp_rate: float = 0.2,
    ramp_ceiling: float = 0.6,
    dwell: float = 1.0,
    motion_threshold: float = 0.01,
    step_torques: Tuple[float, ...] = (0.2, 0.25, 0.3, 0.4, 0.5),
    step_timeout: float = 6.0,
    experiments: Tuple[Experiment, ...] = ("breakaway", "steps"),
    sweep_points: int = 21,
    sweep_range: Tuple[float, float] = (0.15, 0.85),
    sweep_repeats: int = 1,
    sweep_approach: float = 0.03,
    jaw_positions: Tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    yes: bool = False,
) -> None:
    """Characterize the gripper's dynamics under torque control on real hardware.

    Args:
        arm: arm variant, or no_arm for a gripper alone on the bench. An arm is held at its starting pose.
        gripper: gripper variant.
        channel: CAN channel.
        out: directory the run's folder is written into.
        breakaway_positions: normalized start positions (0 closed, 1 open) for the breakaway ramps.
        repeats: repetitions of every breakaway ramp and torque step.
        ramp_rate: breakaway torque ramp rate (Nm/s). Slower resolves the threshold better.
        ramp_ceiling: largest torque a breakaway ramp reaches (Nm) before giving up.
        dwell: rest at zero torque before each ramp (s); static friction can grow with dwell.
        motion_threshold: normalized travel that counts as breakaway.
        step_torques: constant torques (Nm) for the step runs, each in both directions.
        step_timeout: longest a step run may take (s).
        experiments: which to run, in order: breakaway, steps, sweep, jaw_travel.
        sweep_points: positions in the breakaway sweep, evenly spaced over sweep_range.
        sweep_range: normalized range the sweep covers; must stay clear of the wall zones.
        sweep_repeats: repetitions of every sweep position (in both directions).
        sweep_approach: how far beyond each sweep position the jaws start (normalized), on the far side.
        jaw_positions: normalized positions the jaws are held at for the gap measurement.
        yes: start without the confirmation prompt.
    """
    arm_type = ArmType.from_string_name(arm)
    gripper_type = GripperType.from_string_name(gripper)
    sweep_positions = tuple(float(v) for v in np.linspace(*sweep_range, sweep_points)) if sweep_points > 1 else ()
    for v in (*breakaway_positions, *(p + d for p in sweep_positions for d in (-sweep_approach, sweep_approach))):
        if not 0.1 <= v <= 0.9:
            raise SystemExit(f"breakaway positions must stay clear of the wall zones (0.1..0.9), got {v:.3f}")
    for v in jaw_positions:
        if not 0.0 <= v <= 1.0:
            raise SystemExit(f"jaw positions must lie in 0..1, got {v}")
    if max([ramp_ceiling, *step_torques]) > 1.5:
        raise SystemExit("torques above 1.5 Nm exceed every gripper's force_control max_torque")
    logging.getLogger().addFilter(_ForceControlSwitchFilter())

    run_dir = out / f"{gripper}_{datetime.now():%Y%m%d_%H%M%S}"
    trials: List[Trial] = []
    jaw_travel: Optional[Dict[str, Any]] = None
    setup: Optional[Setup] = None
    meta: Dict[str, Any] = {}
    try:
        with hardware_session(
            lambda: get_yam_robot(channel=channel, arm_type=arm_type, gripper_type=gripper_type, sim=False)
        ) as robot:
            info = robot.get_robot_info()
            params = info["gripper_force_control"]
            if info["gripper_index"] is None or params is None:
                raise SystemExit(f"{gripper!r} has no force-controllable gripper motor")
            closed, opened = (float(v) for v in info["gripper_limits"])
            kd = params.kd if params.kd is not None else float(info["kd"][info["gripper_index"]])
            setup = Setup(
                gi=info["gripper_index"],
                n=robot.num_dofs(),
                closed=closed,
                opened=opened,
                kd=kd,
                max_speed=params.max_speed,
                hold=robot.get_joint_pos().copy(),
            )
            meta = {
                "gripper": gripper,
                "arm": arm,
                "date": datetime.now().isoformat(),
                "limits_raw_rad": [closed, opened],
                "stroke_rad": setup.stroke,
                "kd_force_mode": kd,
                "force_control": asdict(params),
                "breakaway_positions": list(breakaway_positions),
                "repeats": repeats,
                "ramp_rate_nm_s": ramp_rate,
                "dwell_s": dwell,
                "motion_threshold": motion_threshold,
                "step_torques_nm": list(step_torques),
                "experiments": list(experiments),
                "sweep_positions": list(sweep_positions),
                "sweep_repeats": sweep_repeats,
                "sweep_approach": sweep_approach,
                "torque_convention": "motor torque, positive closes",
            }
            print(f"[char] calibrated stroke {setup.stroke:.3f} rad, force-mode kd {kd}")
            if not yes and set(experiments) - {"jaw_travel"}:
                input("[char] Keep the jaws FREE (nothing between them) for the whole run. Press Enter to start ... ")

            plan: List[Trial] = []
            for experiment in experiments:
                if experiment == "breakaway":
                    plan += [
                        Trial("breakaway", d, p, r, torque=ramp_ceiling)
                        for r in range(repeats)
                        for p in breakaway_positions
                        for d in ("close", "open")
                    ]
                elif experiment == "steps":
                    plan += [
                        Trial("step", d, 0.92 if d == "close" else 0.08, r, torque=tq)
                        for r in range(repeats)
                        for tq in step_torques
                        for d in ("close", "open")
                    ]
                elif experiment == "sweep":
                    plan += [
                        Trial("sweep", d, p, r, torque=ramp_ceiling)
                        for r in range(sweep_repeats)
                        for p in sweep_positions
                        for d in ("close", "open")
                    ]
            for i, trial in enumerate(plan, 1):
                if trial.kind in ("breakaway", "sweep"):
                    approach = sweep_approach if trial.kind == "sweep" else 0.0
                    run_breakaway(robot, setup, trial, ramp_rate, dwell, motion_threshold, approach)
                    print(
                        f"[char] {i}/{len(plan)} {trial.kind} {trial.direction:5s} from {trial.start_s:.2f}: "
                        f"{trial.result['breakaway_torque']} Nm"
                    )
                else:
                    run_step(robot, setup, trial, step_timeout)
                    r = trial.result
                    print(
                        f"[char] {i}/{len(plan)} step {trial.direction:5s} {trial.torque:.2f} Nm: "
                        f"{r['steady_speed_stroke_s']} stroke/s"
                        + (" (speed-limited)" if r["speed_limited"] else "")
                        + ("" if r["reached_end"] else " (did not reach the end)")
                    )
                trials.append(trial)
            if "jaw_travel" in experiments:
                print("[char] jaw travel: measure the gap between the jaw faces at each position (calipers)")
                points = run_jaw_travel(robot, setup, jaw_positions)
                limiter = gripper_type.get_gripper_limiter_params(
                    ArmType.YAM if arm_type == ArmType.NO_ARM else arm_type
                )
                jaw_travel = analyze_jaw_travel(
                    points, setup.stroke, limiter[3], _limiter_config(gripper_type, arm_type)
                )
            _move_to(robot, setup, 0.5)
    except KeyboardInterrupt:
        print("[char] interrupted; saving the trials that finished")
    finally:
        if (trials or jaw_travel) and setup is not None:
            fit = analyze(trials, setup)
            if jaw_travel is not None:
                fit["jaw_travel"] = jaw_travel
            _save(run_dir, meta, trials, fit)
            print(f"[char] wrote {run_dir}")
            print(json.dumps(fit, indent=2))


if __name__ == "__main__":
    tyro.cli(main)
