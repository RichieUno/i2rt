"""Compare a characterization run with the model fitted to it: where does the simple model hold, and where not?

The model is the one ``characterize_gripper.py`` fits, with its parameters taken from the run's
``fit.json``, separately for each direction:

    J * domega/dt = tau - tau_c * sign(omega) - (b + kd) * omega      while moving
    the jaws stay put while |tau| <= tau_s                             from rest (tau_s: mean breakaway)

Every trial is simulated with the torque that was actually commanded (from ``samples.csv``),
starting where the trial started, and compared with what the gripper did:

* **breakaway** -- the model has one threshold per direction; the data shows whether it depends on
  where the jaws are, which the model cannot express;
* **torque steps** -- steady speed against torque (the model's line), the rise time constant
  against torque (the model says constant), and the whole position trajectory.

If the run includes a **breakaway sweep**, it is plotted against the model's single thresholds,
with its split into friction and position-dependent offset; a **jaw travel** measurement is
reported as the force map's error. Either can be the whole run.

Writes ``comparison/`` inside the run folder: ``trials.csv`` (measured against predicted, per
trial), ``summary.json``, and -- if matplotlib is available -- plots. matplotlib is not a project
dependency; to get the plots without adding it:

    uv run --with matplotlib python examples/gripper_force_control/compare_gripper_model.py <run_dir>
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tyro


@dataclass(frozen=True)
class DirectionModel:
    """The fitted model for one direction, in motor units (Nm, rad/s, kg m^2)."""

    tau_s: float  # breakaway
    tau_c: float  # moving friction
    damping: float  # b + kd
    inertia: float


@dataclass
class TrialData:
    index: int
    kind: str
    direction: str
    start_s: float
    torque: float
    t: np.ndarray
    s: np.ndarray
    cmd: np.ndarray  # closing positive
    result: Dict[str, str]


def load_run(run_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any], List[TrialData]]:
    meta = json.loads((run_dir / "meta.json").read_text())
    fit = json.loads((run_dir / "fit.json").read_text())
    per_trial: Dict[int, List[Dict[str, str]]] = {}
    with open(run_dir / "samples.csv") as f:
        for row in csv.DictReader(f):
            per_trial.setdefault(int(row["trial"]), []).append(row)
    with open(run_dir / "trials.csv") as f:
        rows = {int(r["trial"]): r for r in csv.DictReader(f)}
    trials = []
    for i, samples in sorted(per_trial.items()):
        r = rows[i]
        trials.append(
            TrialData(
                index=i,
                kind=r["kind"],
                direction=r["direction"],
                start_s=float(r["start_s"]),
                torque=float(r["torque_nm"]),
                t=np.array([float(x["t_s"]) for x in samples]),
                s=np.array([float(x["s"]) for x in samples]),
                cmd=np.array([float(x["cmd_closing_nm"]) for x in samples]),
                result=r,
            )
        )
    return meta, fit, trials


def models_from_fit(fit: Dict[str, Any]) -> Dict[str, DirectionModel]:
    models = {}
    for direction in ("close", "open"):
        steps, brk = fit["steps"].get(direction), fit["breakaway"].get(direction)
        if not steps or not brk or "inertia_kg_m2" not in steps:
            continue
        models[direction] = DirectionModel(
            tau_s=brk["mean_nm"],
            tau_c=steps["tau_c_kinetic"],
            damping=steps["total_damping"],
            inertia=steps["inertia_kg_m2"],
        )
    return models


def simulate(model: DirectionModel, t: np.ndarray, cmd_along: np.ndarray, substeps: int = 40) -> np.ndarray:
    """Travel (rad, along the commanded direction) at each sample time, from rest at the first sample.

    ``cmd_along`` is the commanded torque projected onto the trial's direction, held between
    samples as the robot holds a command between updates.
    """
    q = omega = 0.0
    out = np.zeros(len(t))
    for k in range(1, len(t)):
        h = (t[k] - t[k - 1]) / substeps
        tau = cmd_along[k - 1]
        for _ in range(substeps):
            if omega == 0.0 and abs(tau) <= model.tau_s:
                continue  # static friction holds
            direction = np.sign(omega) if omega != 0.0 else np.sign(tau)
            new = omega + (tau - model.tau_c * direction - model.damping * omega) / model.inertia * h
            omega = 0.0 if omega != 0.0 and np.sign(new) != np.sign(omega) else new  # friction stops it
            q += omega * h
        out[k] = q
    return out


def _along(trial: TrialData, stroke: float) -> Tuple[np.ndarray, np.ndarray]:
    """Measured travel (rad) and commanded torque, both positive along the trial's direction."""
    sign = 1.0 if trial.direction == "close" else -1.0
    travel = -sign * (trial.s - trial.s[0]) * stroke  # closing lowers s
    return travel, sign * trial.cmd


def compare(meta: Dict[str, Any], fit: Dict[str, Any], trials: List[TrialData]) -> Dict[str, Any]:
    stroke = float(meta["stroke_rad"])
    models = models_from_fit(fit)
    rows: List[Dict[str, Any]] = []
    for trial in trials:
        model = models.get(trial.direction)
        if model is None:
            continue
        travel, cmd = _along(trial, stroke)
        predicted = simulate(model, trial.t, cmd)
        row: Dict[str, Any] = {
            "trial": trial.index,
            "kind": trial.kind,
            "direction": trial.direction,
            "start_s": trial.start_s,
            "torque_nm": trial.torque,
        }
        if trial.kind == "breakaway":
            measured = float(trial.result["breakaway_torque"]) if trial.result["breakaway_torque"] else None
            row.update(
                breakaway_measured_nm=measured,
                breakaway_model_nm=model.tau_s,
                breakaway_error_nm=None if measured is None else measured - model.tau_s,
            )
        else:
            moved = travel[-1] > 0.05 * stroke
            row.update(
                reached_end=trial.result.get("reached_end") == "True",
                speed_measured_rad_s=_float(trial.result.get("steady_speed_rad_s")),
                speed_model_rad_s=max(0.0, (trial.torque - model.tau_c) / model.damping)
                if trial.torque > model.tau_s
                else 0.0,
                time_constant_measured_s=_float(trial.result.get("time_constant_s")),
                time_constant_model_s=model.inertia / model.damping,
                travel_rms_error_rad=float(np.sqrt(np.mean((travel - predicted) ** 2))) if moved else None,
                travel_final_measured_rad=float(travel[-1]),
                travel_final_model_rad=float(predicted[-1]),
            )
        rows.append(row)
    return {"models": {d: m.__dict__ for d, m in models.items()}, "rows": rows, "summary": _summarize(rows, models)}


def _float(value: Optional[str]) -> Optional[float]:
    return float(value) if value not in (None, "", "None") else None


def _summarize(rows: List[Dict[str, Any]], models: Dict[str, DirectionModel]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for direction, model in models.items():
        brk = [
            r
            for r in rows
            if r["kind"] == "breakaway" and r["direction"] == direction and r["breakaway_measured_nm"] is not None
        ]
        by_pos: Dict[float, List[float]] = {}
        for r in brk:
            by_pos.setdefault(r["start_s"], []).append(r["breakaway_measured_nm"])
        spread_within = max((np.std(v) for v in by_pos.values()), default=0.0)
        spread_between = float(np.std([np.mean(v) for v in by_pos.values()])) if by_pos else 0.0
        steps = [r for r in rows if r["kind"] == "step" and r["direction"] == direction]
        speed_err = [
            r["speed_measured_rad_s"] - r["speed_model_rad_s"]
            for r in steps
            if r["speed_measured_rad_s"] is not None and r["reached_end"]
        ]
        tc = sorted(
            (r["torque_nm"], r["time_constant_measured_s"])
            for r in steps
            if r["time_constant_measured_s"] and r["reached_end"]
        )
        out[direction] = {
            "breakaway_by_position_nm": {f"{p:g}": float(np.mean(v)) for p, v in sorted(by_pos.items())},
            "breakaway_model_nm": model.tau_s,
            "breakaway_spread_within_a_position_nm": float(spread_within),
            "breakaway_spread_between_positions_nm": spread_between,
            "steady_speed_rms_error_rad_s": float(np.sqrt(np.mean(np.square(speed_err)))) if speed_err else None,
            "time_constant_by_torque_s": {
                f"{t:g}": float(np.mean([v for tt, v in tc if tt == t])) for t in sorted({t for t, _ in tc})
            },
            "time_constant_model_s": model.inertia / model.damping,
            "steps_not_reaching_the_end": [r["trial"] for r in steps if not r["reached_end"]],
        }
    return out


def _extras(fit: Dict[str, Any]) -> Dict[str, Any]:
    """The sweep and jaw-travel results worth reading at a glance."""
    out: Dict[str, Any] = {}
    if "sweep" in fit:
        sw = fit["sweep"]
        out["sweep"] = {k: sw[k] for k in ("friction_range_nm", "offset_range_nm", "offset_period", "friction_period")}
    if "jaw_travel" in fit:
        jt = fit["jaw_travel"]
        keys = (
            "force_control_nm_per_n",
            "force_control_map_error",
            "mm_per_rad",
            "travel_over_calibrated_stroke_mm",
            "linearity_rms_mm",
            "measured_nm_per_n",
            "config_nm_per_n",
            "force_map_error",
            "suggested",
        )
        out["jaw_travel"] = {k: jt[k] for k in keys if k in jt}
    return out


def _write(out_dir: Path, result: Dict[str, Any], fit: Dict[str, Any]) -> None:
    out_dir.mkdir(exist_ok=True)
    summary = {"models": result["models"], **result["summary"], **_extras(fit)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    keys = sorted({k for r in result["rows"] for k in r})
    with open(out_dir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(result["rows"])


def _plot_sweep(plt: Any, out_dir: Path, fit: Dict[str, Any], models: Dict[str, DirectionModel]) -> None:
    sweep = fit["sweep"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(sweep["positions"], sweep["close_nm"], "o-", color="#1f77b4", label="close: measured")
    ax1.plot(sweep["positions"], sweep["open_nm"], "o-", color="#d62728", label="open: measured")
    for d, color in (("close", "#1f77b4"), ("open", "#d62728")):
        if d in models:
            ax1.axhline(models[d].tau_s, color=color, ls="--", label=f"{d}: model (one value)")
    ax1.set(xlabel="start position (0 closed, 1 open)", ylabel="breakaway torque (Nm)", title="Breakaway sweep")
    ax1.legend(fontsize=8)
    ax2.plot(sweep["motor_angle_rad"], sweep["friction_nm"], "o-", color="#2ca02c", label="friction (mean)")
    ax2.plot(sweep["motor_angle_rad"], sweep["offset_nm"], "o-", color="#9467bd", label="offset (+ pushes open)")
    ax2.axhline(0.0, color="gray", lw=0.8)
    ax2.set(xlabel="motor angle (rad)", ylabel="torque (Nm)", title="Friction and offset")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "breakaway_sweep.png", dpi=150)


def _plot(
    out_dir: Path, meta: Dict[str, Any], fit: Dict[str, Any], trials: List[TrialData], result: Dict[str, Any]
) -> bool:
    try:
        import matplotlib  # pyright: ignore[reportMissingImports]  -- optional, see the module docstring

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    models = {d: DirectionModel(**m) for d, m in result["models"].items()}
    if "sweep" in fit:
        _plot_sweep(plt, out_dir, fit, models)
    if not models:  # a sweep or jaw-travel run alone has no steps to fit a model from
        plt.close("all")
        return True
    rows = result["rows"]
    colors = {"close": "#1f77b4", "open": "#d62728"}

    fig, ax = plt.subplots(figsize=(6, 4))
    for d, m in models.items():
        pts = [
            (r["torque_nm"], r["speed_measured_rad_s"])
            for r in rows
            if r["kind"] == "step" and r["direction"] == d and r["speed_measured_rad_s"] is not None
        ]
        ax.plot(*zip(*pts, strict=True), "o", color=colors[d], alpha=0.7, label=f"{d}: measured")
        tq = np.linspace(min(p[0] for p in pts), max(p[0] for p in pts), 50)
        ax.plot(tq, np.clip((tq - m.tau_c) / m.damping, 0, None), "-", color=colors[d], label=f"{d}: model")
    ax.set(xlabel="motor torque (Nm)", ylabel="steady speed (rad/s)", title="Steady speed vs torque")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "steady_speed.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for d, m in models.items():
        pts = [
            (r["start_s"], r["breakaway_measured_nm"])
            for r in rows
            if r["kind"] == "breakaway" and r["direction"] == d and r["breakaway_measured_nm"] is not None
        ]
        offset = -0.01 if d == "close" else 0.01
        ax.plot(
            [p + offset for p, _ in pts], [v for _, v in pts], "o", color=colors[d], alpha=0.7, label=f"{d}: measured"
        )
        ax.axhline(m.tau_s, color=colors[d], ls="--", label=f"{d}: model (one value)")
    ax.set(xlabel="start position (0 closed, 1 open)", ylabel="breakaway torque (Nm)", title="Breakaway vs position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "breakaway.png", dpi=150)

    stroke = float(meta["stroke_rad"])
    steps = [t for t in trials if t.kind == "step" and t.direction in models]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, d in zip(axes, ("close", "open"), strict=True):
        for tr in [t for t in steps if t.direction == d and t.result.get("repeat") == "0"]:
            travel, cmd = _along(tr, stroke)
            (line,) = ax.plot(tr.t, travel, "-", lw=1.5, label=f"{tr.torque:g} Nm")
            ax.plot(tr.t, simulate(models[d], tr.t, cmd), "--", color=line.get_color(), lw=1)
        ax.set(title=f"{d}: measured (solid) vs model (dashed)", xlabel="time (s)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("travel (rad)")
    fig.tight_layout()
    fig.savefig(out_dir / "step_trajectories.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for d, m in models.items():
        pts = [
            (r["torque_nm"], r["time_constant_measured_s"] * 1e3)
            for r in rows
            if r["kind"] == "step" and r["direction"] == d and r["time_constant_measured_s"] and r["reached_end"]
        ]
        ax.plot(*zip(*pts, strict=True), "o", color=colors[d], alpha=0.7, label=f"{d}: measured")
        ax.axhline(m.inertia / m.damping * 1e3, color=colors[d], ls="--", label=f"{d}: model (constant)")
    ax.set(xlabel="motor torque (Nm)", ylabel="rise time constant (ms)", title="Rise time constant vs torque")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_constant.png", dpi=150)
    plt.close("all")
    return True


def main(run_dir: tyro.conf.Positional[Path]) -> None:
    """Compare a characterize_gripper.py run with the model in its fit.json.

    Args:
        run_dir: the run's folder, holding meta.json, fit.json, trials.csv and samples.csv.
    """
    meta, fit, trials = load_run(run_dir)
    result = compare(meta, fit, trials)
    out_dir = run_dir / "comparison"
    _write(out_dir, result, fit)
    plotted = _plot(out_dir, meta, fit, trials, result)
    print(json.dumps({**result["summary"], **_extras(fit)}, indent=2))
    print(f"[compare] wrote {out_dir}" + ("" if plotted else " (no plots: run with `uv run --with matplotlib`)"))


if __name__ == "__main__":
    tyro.cli(main)
