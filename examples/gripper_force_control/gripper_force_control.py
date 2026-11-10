"""Hardware test for gripper force control: drive the gripper motor by force from a browser GUI.

Opens a viser page with the same safety gate as ``control_with_viser``: nothing is commanded until
you tick **Alignment Confirmed** and click **Enable**. Then the gripper runs in one of two modes:

* **Position** -- the ordinary position-controlled gripper (slider 0 = closed, 1 = open).
* **Force** -- the gripper motor is torque-controlled to the force on the slider (N, positive closes,
  negative opens), under the clamps, the virtual wall at the stops and the speed limit of the
  gripper config's ``force_control`` block. The slider is bounded by ``--max-force``, which is
  itself capped at the config's ``max_force``.

With an arm attached, the arm is held at the pose it had when you clicked Enable, by its PD gains
plus gravity comp, while the gripper is tested. The default ``--arm no_arm`` runs a gripper alone on
the bench.

A 3-D view mirrors the robot from its measured joint positions, gripper included, so you can watch
the jaws while you test. A gripper on its own gets a close-up camera.

A live plot shows the commanded force against the force estimated from the motor's measured torque
through the same map. The estimate includes friction and damping while the jaws move, so compare the
two at a stall on an object -- and against a force gauge before trusting either.

Hardware only: ``SimRobot`` has no gripper dynamics to apply a force to.

Usage:
    python examples/gripper_force_control/gripper_force_control.py --channel can0
    python examples/gripper_force_control/gripper_force_control.py --gripper crank_4310 --max-force 15
    python examples/gripper_force_control/gripper_force_control.py --arm yam --gripper linear_4310
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
from collections import deque
from typing import Any, Deque, Dict, List

import mujoco
import numpy as np
import tyro
from hardware_session import chain_alive, hardware_session

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import ArmName, ArmType, GripperType

_HISTORY_S = 10.0  # seconds of force history in the plot
_FORCE_RAMP = 40.0  # N/s: how fast the commanded force may grow toward a new setting


class _RobotView:
    """The robot's MuJoCo model drawn in the viser scene, posed from the robot's joint positions.

    Self-contained on purpose, so the example changes no library code. The mirroring follows
    ``ViserControlInterface._mirror_robot``: the gripper's normalized 0..1 is scaled onto its
    finger slide joint, and the second finger is set from the model's joint equality. Only joint
    positions are read, so it shows the real robot exactly as it reports itself.
    """

    def __init__(self, server: Any, robot: MotorChainRobot, close_up: bool) -> None:
        self._robot = robot
        self._model = mujoco.MjModel.from_xml_path(robot.xml_path)
        self._data = mujoco.MjData(self._model)
        self._geoms: List[int] = []
        self._handles: Dict[int, Any] = {}
        for geom_id in range(self._model.ngeom):
            if self._model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = self._model.geom_dataid[geom_id]
            v0, nv = self._model.mesh_vertadr[mesh], self._model.mesh_vertnum[mesh]
            f0, nf = self._model.mesh_faceadr[mesh], self._model.mesh_facenum[mesh]
            self._geoms.append(geom_id)
            self._handles[geom_id] = server.scene.add_mesh_simple(
                f"/robot/geom_{geom_id}",
                self._model.mesh_vert[v0 : v0 + nv].copy(),
                self._model.mesh_face[f0 : f0 + nf].copy(),
                color=tuple(int(c * 255) for c in self._model.geom_rgba[geom_id][:3]),
            )
        self.mirror()

        # Frame the gripper up close, or the whole arm from its base (the origin) to the grasp site.
        site = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
        tool = self._data.site_xpos[site].copy() if site != -1 else np.zeros(3)
        look_at = tool if close_up else 0.5 * tool
        offset = np.array([0.18, 0.18, 0.12]) if close_up else np.array([0.6, 0.6, 0.4])

        @server.on_client_connect
        def _(client: Any) -> None:
            client.camera.position = tuple(float(v) for v in look_at + offset)
            client.camera.look_at = tuple(float(v) for v in look_at)

    def mirror(self) -> None:
        """Pose the model from the robot's current joint positions and refresh every mesh."""
        q = self._robot.get_joint_pos()
        model, data = self._model, self._data
        n = min(len(q), model.nq)
        data.qpos[:n] = q[:n]
        for j in range(model.njnt):  # normalized slide joints (the gripper) onto their range in metres
            adr = model.jnt_qposadr[j]
            if adr < n and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                lo, hi = model.jnt_range[j]
                data.qpos[adr] = lo + data.qpos[adr] * (hi - lo)
        for i in range(model.neq):  # the second finger follows the first
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                a, b = model.jnt_qposadr[model.eq_obj1id[i]], model.jnt_qposadr[model.eq_obj2id[i]]
                data.qpos[b] = np.polyval(model.eq_data[i, :5][::-1], data.qpos[a])
        mujoco.mj_forward(model, data)
        wxyz = np.empty(4)
        for geom_id in self._geoms:
            mujoco.mju_mat2Quat(wxyz, data.geom_xmat[geom_id])
            handle = self._handles[geom_id]
            handle.position = data.geom_xpos[geom_id].copy()
            handle.wxyz = wxyz.copy()


def sanitize_force(raw: Any, last_good: float, limit: float) -> "tuple[float, bool]":
    """The slider's force as a usable number: ``(force, accepted)``.

    A partly typed entry ("-", "", "1e") reaches the slider as NaN or garbage. That is not a request
    to change the force, so the last good value stands; anything numeric is clamped to +-limit,
    since a typed number can bypass the slider's own range.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return last_good, False
    if not np.isfinite(value):
        return last_good, False
    return float(np.clip(value, -limit, limit)), True


def ramp_force(commanded: float, target: float, max_step: float) -> float:
    """Move the commanded force toward ``target``: instantly when backing off, at most ``max_step`` when pushing harder.

    Typing "15" passes through "1" on the way, and a fat-fingered "50" for "5" would otherwise land at
    once; ramping every increase bounds what one keystroke can do. Reducing the force, reversing it
    (which first drops it to zero) and Zero force all act immediately, since releasing is the safe way.
    """
    if commanded != 0.0 and np.sign(target) != np.sign(commanded):
        commanded = 0.0
    if abs(target) <= abs(commanded):
        return target
    return commanded + float(np.clip(target - commanded, -max_step, max_step))


def main(
    arm: ArmName = "no_arm",
    gripper: str = "linear_4310",
    channel: str = "can0",
    max_force: float = 20.0,
    dt: float = 0.02,
    port: int = 8080,
) -> None:
    """Browser GUI to command the gripper by force on real hardware.

    Args:
        arm: arm variant, or no_arm for a gripper alone on the bench.
        gripper: gripper variant; must have a force_control block in its config.
        channel: CAN channel.
        max_force: largest force (N) the slider offers, either direction. Capped at the config's max_force.
        dt: loop timestep (s).
        port: viser server port.
    """
    import viser  # optional dependency, as in control_with_viser
    from viser import uplot

    arm_type = ArmType.from_string_name(arm)
    gripper_type = GripperType.from_string_name(gripper)
    params = gripper_type.get_force_control_params(ArmType.YAM if arm_type == ArmType.NO_ARM else arm_type)
    if params is None:
        raise SystemExit(f"{gripper!r} has no force_control config, so it cannot be force-controlled")
    if not max_force > 0:
        raise SystemExit(f"--max-force must be positive, got {max_force}")
    slider_max = min(max_force, params.max_force)
    if slider_max < max_force:
        print(f"[force] --max-force {max_force} N capped at the config's max_force, {params.max_force} N")

    # On every exit -- Ctrl+C, kill, an error -- the session lets go of the gripper and closes the robot.
    try:
        with hardware_session(
            lambda: get_yam_robot(channel=channel, arm_type=arm_type, gripper_type=gripper_type, sim=False)
        ) as robot:
            gi = robot.get_robot_info()["gripper_index"]
            if gi is None:
                raise SystemExit(f"{gripper!r} gives this robot no gripper motor")
            n = robot.num_dofs()
            _run(viser, uplot, robot, gi, n, slider_max, dt, port, close_up=arm_type == ArmType.NO_ARM)
    except KeyboardInterrupt:
        pass
    print("[force] stopped")


def _run(
    viser: Any,
    uplot: Any,
    robot: MotorChainRobot,
    gi: int,
    n: int,
    slider_max: float,
    dt: float,
    port: int,
    close_up: bool,
) -> None:
    server = viser.ViserServer(port=port)
    print(f"[force] open http://localhost:{port} -- DISABLED until you confirm and enable")

    state: Dict[str, Any] = {"enabled": False, "hold": None}
    view = _RobotView(server, robot, close_up)

    with server.gui.add_folder("Safety"):
        align_cb = server.gui.add_checkbox("Alignment Confirmed", initial_value=False)
        enable_btn = server.gui.add_button("Enable", disabled=True)
        status_md = server.gui.add_markdown("**Status:** DISABLED (read-only)")

    with server.gui.add_folder("Gripper"):
        mode_dd = server.gui.add_dropdown(
            "Mode", options=["Position", "Force"], initial_value="Position", disabled=True
        )
        pos_slider = server.gui.add_slider("Position (0 closed, 1 open)", 0.0, 1.0, 0.01, 1.0, disabled=True)
        force_slider = server.gui.add_slider(
            "Force (N, + closes)",
            -slider_max,
            slider_max,
            0.5,
            0.0,
            disabled=True,
            marks=((-slider_max, f"{-slider_max:g}"), (0.0, "0"), (slider_max, f"{slider_max:g}")),
        )
        zero_btn = server.gui.add_button("Zero force", disabled=True)
        readout_md = server.gui.add_markdown("")

    history: Dict[str, Deque[float]] = {k: deque() for k in ("t", "cmd", "est")}
    plot = server.gui.add_uplot(
        data=(np.zeros(1), np.zeros(1), np.zeros(1)),
        series=(
            uplot.Series(label="t (s)"),
            uplot.Series(label="commanded (N)", stroke="#1f77b4", width=2),
            uplot.Series(label="estimated (N)", stroke="#d62728", width=2),
        ),
        title="Grip force",
        aspect=1.6,
    )

    @align_cb.on_update
    def _(_: object) -> None:
        enable_btn.disabled = not align_cb.value

    @enable_btn.on_click
    def _(_: object) -> None:
        q = robot.get_joint_pos().copy()
        state["hold"] = q  # the arm stays here; the gripper starts where it is, so nothing moves
        pos_slider.value = float(np.clip(q[gi], 0.0, 1.0))
        state["enabled"] = True
        align_cb.disabled = enable_btn.disabled = True
        mode_dd.disabled = pos_slider.disabled = False
        status_md.content = "**Status:** ENABLED"
        print("[force] ENABLED")

    @mode_dd.on_update
    def _(_: object) -> None:
        force = mode_dd.value == "Force"
        force_slider.disabled = zero_btn.disabled = not force
        pos_slider.disabled = force
        if force:
            force_slider.value = 0.0  # always enter at zero force, never at a stale setting
        else:
            # Back to position control where the jaws are now, so leaving force mode does not move them.
            pos_slider.value = float(np.clip(robot.get_joint_pos()[gi], 0.0, 1.0))

    @zero_btn.on_click
    def _(_: object) -> None:
        force_slider.value = 0.0

    t0 = time.time()
    last_t = t0
    target_force = 0.0  # last accepted slider value
    commanded = 0.0  # what is actually sent: target_force, ramped
    try:
        while True:
            if not chain_alive(robot):
                print("[force] the motor chain stopped (see the first error above); exiting")
                break
            view.mirror()
            q = robot.get_joint_pos()
            obs = robot.get_observations()
            est = robot.get_gripper_force_estimate()
            force_mode = state["enabled"] and mode_dd.value == "Force"
            now_t = time.time()
            if force_mode:
                target_force, accepted = sanitize_force(force_slider.value, target_force, slider_max)
                if not accepted:
                    force_slider.value = target_force  # show the value that is actually in force
                commanded = ramp_force(commanded, target_force, _FORCE_RAMP * (now_t - last_t))
            else:
                target_force = commanded = 0.0  # Force mode always restarts from zero
            last_t = now_t

            if state["enabled"]:
                cmd = state["hold"].copy()
                cmd[gi] = float(pos_slider.value)
                command: Dict[str, Any] = {"pos": cmd, "vel": np.zeros(n)}
                if force_mode:
                    command["gripper_force"] = commanded
                try:
                    robot.command_joint_state(command)
                except ValueError as e:
                    # A refused command leaves the robot on the last valid one, so keep the session
                    # going rather than tearing it down over one bad value.
                    print(f"[force] command refused, robot keeps its last valid command: {e}")
            else:
                pos_slider.value = float(np.clip(q[gi], 0.0, 1.0))

            readout_md.content = (
                f"**Mode:** {'FORCE' if force_mode else 'position'}  \n"
                + (f"**Force:** {commanded:+.1f} N (setting {target_force:+.1f} N)  \n" if force_mode else "")
                + f"**Position:** {q[gi]:.3f}  \n"
                f"**Velocity:** {float(obs['gripper_vel'][0]):+.3f} /s  \n"
                f"**Motor torque:** {float(obs['gripper_eff'][0]):+.3f} Nm  \n"
                f"**Estimated force:** {est or 0.0:+.1f} N"
            )

            now = time.time() - t0
            history["t"].append(now)
            history["cmd"].append(commanded)
            history["est"].append(est if est is not None else 0.0)
            while history["t"] and history["t"][0] < now - _HISTORY_S:
                for v in history.values():
                    v.popleft()
            plot.data = tuple(np.array(history[k]) for k in ("t", "cmd", "est"))

            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    tyro.cli(main)
