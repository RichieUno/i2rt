"""The gripper characterization's analysis, checked on synthetic runs of a gripper with known parameters.

The hardware script cannot run here, but its fits can: a run simulated from

    J * domega/dt = tau - tau_c * sign(omega) - (b + kd) * omega,   held still while |tau| <= tau_s

must give back the J, b, tau_c and breakaway torque tau_s it was generated with.
"""

import sys
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "gripper_force_control"))
import characterize_gripper as cg

J, B, TAU_C, TAU_S, KD = 4e-4, 0.03, 0.12, 0.18, 0.05
DT = 0.002  # the script's sampling period
STROKE = 6.6  # rad


def _simulate(
    tau_of_t: Callable[[float], float], seconds: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Motor angle and speed under a torque profile, with breakaway at TAU_S and moving friction TAU_C."""
    sub = 20
    h = DT / sub
    q = omega = 0.0
    t_log, q_log, w_log, tau_log = [], [], [], []
    for k in range(int(seconds / DT)):
        t = k * DT
        tau = tau_of_t(t)
        for _ in range(sub):
            if omega == 0.0 and abs(tau) <= TAU_S:
                continue  # stuck
            direction = np.sign(omega) if omega != 0.0 else np.sign(tau)
            new = omega + (tau - TAU_C * direction - (B + KD) * omega) / J * h
            omega = 0.0 if omega != 0.0 and np.sign(new) != np.sign(omega) else new
            q += omega * h
        t_log.append(t)
        q_log.append(q)
        w_log.append(omega)
        tau_log.append(tau)
    return np.array(t_log), np.array(q_log), np.array(w_log), np.array(tau_log)


def test_breakaway_is_found_where_the_motion_starts() -> None:
    rate = 0.2  # Nm/s, the script's default
    _, q, _, tau = _simulate(lambda t: rate * t, 2.0)
    s = 0.5 - q / STROKE  # closing lowers s
    idx = cg.breakaway_index(s, threshold=0.01)
    assert idx is not None
    # Within two samples' worth of ramp of the true breakaway torque, not where 1 % of stroke is reached.
    assert tau[idx] == pytest.approx(TAU_S, abs=2 * rate * DT + 1e-3)


def test_no_breakaway_below_the_threshold() -> None:
    _, q, _, _ = _simulate(lambda t: 0.9 * TAU_S, 1.0)
    assert cg.breakaway_index(0.5 - q / STROKE, threshold=0.01) is None


def test_steps_recover_friction_damping_and_inertia() -> None:
    torques = [0.2, 0.25, 0.3, 0.4, 0.5]
    speeds, taus = [], []
    for tq in torques:
        t, q, w, _ = _simulate(lambda t, tq=tq: tq, 3.0)
        omega = cg.steady_speed(t, q, (0.3 * STROKE, 0.7 * STROKE))
        assert omega is not None
        speeds.append(omega)
        taus.append(cg.rise_time_constant(t, w, omega))
    fit = cg.fit_speed_line(torques, speeds, KD)
    assert fit["tau_c_kinetic"] == pytest.approx(TAU_C, rel=0.02)
    assert fit["b_viscous"] == pytest.approx(B, rel=0.03)
    # The rise is ~2.5 samples here (T = J / (b + kd) = 5 ms against 2 ms sampling): the curve fit must
    # still recover the inertia.
    inertia = np.median(taus) * fit["total_damping"]
    assert inertia == pytest.approx(J, rel=0.1)


def test_the_speed_line_needs_two_torques() -> None:
    with pytest.raises(ValueError):
        cg.fit_speed_line([0.3], [5.0], KD)


# ---- the experiment loops, against a gripper simulated in real time ----------------------

import threading
import time
from types import SimpleNamespace
from typing import Any, Dict, List

CLOSED, OPENED = 2.0, 2.0 - STROKE  # raw limits; closing is +q here


class _FakeGripperRobot:
    """The robot surface the script uses, over the same physics as ``_simulate``, stepped in a thread."""

    def __init__(self) -> None:
        self.motor_chain = SimpleNamespace(running=True)
        self.s, self.omega, self.tau = 0.5, 0.0, 0.0  # omega: closing speed in motor rad/s
        self.position_target: Any = 0.5
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        # Advance by the wall time that actually passed, not a fixed step per iteration: the loop
        # also sleeps and is scheduled late under load, and a sim running slower than the wall clock
        # reads as extra damping in a fit done against wall time.
        h = 1e-4
        last = time.perf_counter()
        while self.motor_chain.running:
            now = time.perf_counter()
            steps, last = int((now - last) / h), last + int((now - last) / h) * h
            with self._lock:
                for _ in range(steps):
                    if self.position_target is not None:  # position mode: an ideal servo
                        self.s, self.omega = float(self.position_target), 0.0
                        continue
                    tau = self.tau
                    if self.omega == 0.0 and abs(tau) <= TAU_S:
                        continue
                    direction = np.sign(self.omega) if self.omega != 0.0 else np.sign(tau)
                    new = self.omega + (tau - TAU_C * direction - (B + KD) * self.omega) / J * h
                    self.omega = 0.0 if self.omega != 0.0 and np.sign(new) != np.sign(self.omega) else new
                    self.s -= self.omega * h / STROKE
            time.sleep(5e-4)

    def get_joint_pos(self) -> np.ndarray:
        with self._lock:
            return np.array([self.s])

    def get_observations(self) -> Dict[str, np.ndarray]:
        with self._lock:
            # Motor-frame feedback: closing is +q (CLOSED > OPENED), so eff is the closing torque itself.
            return {
                "gripper_pos": np.array([self.s]),
                "gripper_vel": np.array([-self.omega / STROKE]),
                "gripper_eff": np.array([self.tau]),
            }

    def command_joint_state(self, state: Dict[str, Any]) -> None:
        with self._lock:
            if "gripper_torque" in state:
                self.position_target = None
                self.tau = float(np.asarray(state["gripper_torque"]).reshape(-1)[0])
            else:
                self.position_target = float(state["pos"][0])
                self.tau = 0.0


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Any:
    robot = _FakeGripperRobot()
    setup = cg.Setup(gi=0, n=1, closed=CLOSED, opened=OPENED, kd=KD, max_speed=2.0, hold=np.array([0.5]))
    fast_move = cg._move_to.__wrapped__ if hasattr(cg._move_to, "__wrapped__") else cg._move_to
    monkeypatch.setattr(cg, "_move_to", lambda r, su, target: fast_move(r, su, target, duration=0.1, settle=0.05))
    yield robot, setup
    robot.motor_chain.running = False


def test_breakaway_trials_measure_the_static_threshold_both_ways(rig: Any) -> None:
    robot, setup = rig
    trials = []
    for direction in ("close", "open"):
        trial = cg.Trial("breakaway", direction, 0.5, 0, torque=0.6)
        cg.run_breakaway(robot, setup, trial, ramp_rate=0.2, dwell=0.1, threshold=0.01)
        trials.append(trial)
        assert trial.result["broke_away"]
        # Real time and two threads add jitter, so allow a few ramp steps more than the offline check.
        assert trial.result["breakaway_torque"] == pytest.approx(TAU_S, abs=0.01)
    fit = cg.analyze(trials, setup)
    assert set(fit["breakaway"]) == {"close", "open"}


def test_step_trials_run_end_to_end_and_fit_the_line(rig: Any) -> None:
    robot, setup = rig
    trials = []
    for tq in (0.25, 0.35, 0.5):
        for direction, start in (("close", 0.92), ("open", 0.08)):
            trial = cg.Trial("step", direction, start, 0, torque=tq)
            cg.run_step(robot, setup, trial, timeout=6.0)
            assert trial.result["reached_end"], (direction, tq)
            assert not trial.result["speed_limited"]
            trials.append(trial)
    fit = cg.analyze(trials, setup)
    for direction in ("close", "open"):
        line = fit["steps"][direction]
        assert line["tau_c_kinetic"] == pytest.approx(TAU_C, abs=0.02)
        assert line["b_viscous"] == pytest.approx(B, abs=0.01)


def test_a_dead_chain_aborts_a_trial(rig: Any) -> None:
    robot, setup = rig
    robot.motor_chain.running = False
    with pytest.raises(RuntimeError, match="motor chain stopped"):
        cg.run_step(robot, setup, cg.Trial("step", "close", 0.92, 0, torque=0.3), timeout=1.0)


def test_a_run_is_saved_as_csv_and_json(rig: Any, tmp_path: Path) -> None:
    import csv
    import json

    robot, setup = rig
    trial = cg.Trial("step", "close", 0.92, 0, torque=0.4)
    cg.run_step(robot, setup, trial, timeout=6.0)
    cg._save(tmp_path, {"gripper": "linear_4310"}, [trial], cg.analyze([trial], setup))
    assert json.loads((tmp_path / "meta.json").read_text())["gripper"] == "linear_4310"
    assert "steps" in json.loads((tmp_path / "fit.json").read_text())
    with open(tmp_path / "samples.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(trial.samples) and rows[0]["kind"] == "step"
    with open(tmp_path / "trials.csv") as f:
        assert float(next(csv.DictReader(f))["steady_speed_rad_s"]) > 0


def test_a_stalled_step_is_left_out_of_the_speed_line() -> None:
    """A run that stopped part-way has no steady speed; its stub of motion must not bend the fit."""
    setup = cg.Setup(gi=0, n=1, closed=CLOSED, opened=OPENED, kd=KD, max_speed=2.0, hold=np.array([0.5]))
    trials = []
    for tq, reached in ((0.2, False), (0.3, True), (0.4, True), (0.5, True)):
        speed = 0.04 if not reached else (tq - TAU_C) / (B + KD)
        trial = cg.Trial("step", "open", 0.08, 0, torque=tq)
        trial.result = {
            "steady_speed_rad_s": speed,
            "speed_limited": False,
            "reached_end": reached,
            "time_constant_s": J / (B + KD),
        }
        trials.append(trial)
    line = cg.analyze(trials, setup)["steps"]["open"]
    assert line["n"] == 3
    assert line["tau_c_kinetic"] == pytest.approx(TAU_C, rel=1e-6)


# ---- the breakaway sweep and the jaw travel -------------------------------------------------


def test_decomposition_separates_friction_from_an_offset() -> None:
    friction = np.array([0.15, 0.14, 0.16])
    offset = np.array([0.04, -0.05, -0.02])  # + pushes the jaws open: closing needs more
    got_friction, got_offset = cg.decompose_breakaway(friction + offset, friction - offset)
    assert np.allclose(got_friction, friction) and np.allclose(got_offset, offset)


def test_a_cogging_like_offset_shows_its_period() -> None:
    angle = np.linspace(1.2, 6.5, 21)  # 21 points over a 5.3 rad stroke, as the default sweep
    offset = 0.03 * np.sin(2 * np.pi * angle / 1.6) + 0.002 * np.cos(7 * angle)
    found = cg.dominant_period(angle, offset, shortest=2 * float(np.median(np.diff(angle))))
    assert found is not None
    assert found["period_rad"] == pytest.approx(1.6, rel=0.05)
    assert found["amplitude_nm"] == pytest.approx(0.03, rel=0.1)
    assert found["variance_explained"] > 0.9


def _linear_map(ratio: float) -> Any:
    return lambda gripper_force, current_angle: gripper_force * ratio


def test_jaw_travel_measures_the_real_ratio_and_flags_the_config() -> None:
    true_ratio = 0.096 / 5.27  # m per motor rad: 96 mm over the calibrated 5.27 rad
    config = {"force_torque_map": "linear", "gripper_stroke": 0.096, "motor_stroke": 6.57}
    pts = [cg.JawPoint(s, s, 6.5 - s * 5.27, 1000 * true_ratio * s * 5.27) for s in (0.05, 0.25, 0.5, 0.75, 0.95)]
    result = cg.analyze_jaw_travel(pts, 5.27, _linear_map(0.096 / 6.57), config)
    assert result["measured_nm_per_n"] == pytest.approx(true_ratio)
    assert result["travel_over_calibrated_stroke_mm"] == pytest.approx(96.0)
    # The config asks for 6.57/5.27 too little torque per newton: real forces come out ~20 % low.
    assert result["force_map_error"] == pytest.approx(5.27 / 6.57 - 1.0)
    assert result["suggested"] == {"gripper_stroke": 0.096, "motor_stroke": 5.27}
    assert all(seg["map_error"] == pytest.approx(5.27 / 6.57 - 1.0) for seg in result["segments"])
    assert result["linearity_rms_mm"] == pytest.approx(0.0, abs=1e-9)


def test_jaw_travel_prompts_until_it_gets_a_gap(rig: Any) -> None:
    robot, setup = rig
    answers = iter(["", "abc", "-1", "4.8", "47.5"])
    asked = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return next(answers)

    points = cg.run_jaw_travel(robot, setup, (0.05, 0.5), ask=ask)
    assert [p.gap_mm for p in points] == [4.8, 47.5]
    assert len(asked) == 5, "an empty, a non-numeric and a negative answer must each be asked again"
    assert points[1].motor_rad == pytest.approx(CLOSED + points[1].s * (OPENED - CLOSED))


def test_sweep_trials_approach_from_the_far_side_and_decompose(rig: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    robot, setup = rig
    targets: List[float] = []
    move = cg._move_to
    monkeypatch.setattr(cg, "_move_to", lambda r, su, target: (targets.append(round(target, 3)), move(r, su, target)))
    trials = []
    for p in (0.3, 0.5, 0.7):
        for d in ("close", "open"):
            trial = cg.Trial("sweep", d, p, 0, torque=0.6)
            cg.run_breakaway(robot, setup, trial, ramp_rate=0.2, dwell=0.1, threshold=0.01, approach=0.03)
            trials.append(trial)
    # Closing lowers s, so a closing test comes down from above its start, an opening one up from below.
    assert targets[:4] == [0.33, 0.3, 0.27, 0.3]
    sweep = cg.analyze(trials, setup)["sweep"]
    assert sweep["positions"] == [0.3, 0.5, 0.7]
    # The simulated gripper has one breakaway both ways: all friction, no offset.
    assert np.allclose(sweep["friction_nm"], TAU_S, atol=0.01)
    assert np.allclose(sweep["offset_nm"], 0.0, atol=0.01)


def test_a_twitch_before_breakaway_is_reported_as_first_movement_only(
    rig: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Jaws that jump ~0.1 % of the stroke, hold, and only then slide: breakaway is where they slide."""
    t = np.arange(0.0, 2.0, 0.004)
    cmd = np.minimum(0.2 * t, 0.6)
    s = np.full_like(t, 0.5)
    s[cmd >= 0.10] -= 0.001  # the twitch, at 0.10 Nm
    sliding = cmd >= 0.17
    s[sliding] -= 0.02 * (t[sliding] - t[sliding][0]) / 0.1  # the slide, from 0.17 Nm
    arr = np.column_stack([t, s, np.zeros_like(t), np.zeros_like(t), cmd])
    first = cg.breakaway_index(arr[:, 1], 0.01, cg.FIRST_MOVEMENT_TOL)
    sustained = cg.breakaway_index(arr[:, 1], 0.01, cg.SUSTAINED_TOL)
    assert cmd[first] == pytest.approx(0.10, abs=0.002)
    # The sustained reading lags the slide's start by the samples it takes to pass SUSTAINED_TOL (a few
    # mNm of ramp here) -- small against the 0.02-0.10 Nm twitch it separates out.
    assert cmd[sustained] == pytest.approx(0.17, abs=0.005)


def test_jaw_travel_checks_the_force_control_map_against_the_calipers() -> None:
    """Force control's map is over the calibrated stroke; with the nominal jaw travel it is right here."""
    from i2rt.robots.utils import ArmType as _Arm
    from i2rt.robots.utils import GripperType as _Gripper

    stroke = 5.27
    true_ratio = 0.096 / stroke
    config = {"force_torque_map": "linear", "gripper_stroke": 0.096, "motor_stroke": 6.57}
    pts = [cg.JawPoint(s, s, 6.5 - s * stroke, 1000 * true_ratio * s * stroke) for s in (0.05, 0.5, 0.95)]
    fc_map = _Gripper.LINEAR_4310.get_force_control_map(_Arm.YAM, calibrated_stroke=stroke)
    result = cg.analyze_jaw_travel(pts, stroke, _linear_map(0.096 / 6.57), config, force_control_map=fc_map)
    assert result["force_control_map_error"] == pytest.approx(0.0, abs=1e-9)
    assert result["force_map_error"] == pytest.approx(stroke / 6.57 - 1.0), "the config map is still reported"
    assert all(seg["force_control_error"] == pytest.approx(0.0, abs=1e-9) for seg in result["segments"])
