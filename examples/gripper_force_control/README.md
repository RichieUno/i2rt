# Gripper Force Control (hardware test)

Browser GUI that torque-controls the gripper motor to a commanded grip force, for testing
`MotorChainRobot`'s gripper force control on real hardware.

```bash
# A gripper alone on the bench (CAN id 0x07)
python examples/gripper_force_control/gripper_force_control.py --channel can0

# Another gripper, with a lower force ceiling
python examples/gripper_force_control/gripper_force_control.py --gripper crank_4310 --max-force 15

# On an arm: the arm is held at its pose from the moment you click Enable
python examples/gripper_force_control/gripper_force_control.py --arm yam --gripper linear_4310
```

Open `http://localhost:8080`, tick **Alignment Confirmed**, click **Enable**.

The page has a 3D view of the robot, posed live from its measured joint positions, gripper
included. A gripper on its own gets a close-up camera; with an arm the camera frames the whole arm.

## Modes

- **Position**: the normal position-controlled gripper (0 = closed, 1 = open).
- **Force**: the gripper motor runs at `kp = 0` with a feedforward torque mapped from the slider's
  force (N). **Positive closes, negative opens.** Entering force mode always starts at 0 N, and
  **Zero force** returns there. A partly typed or invalid entry is ignored (the slider snaps back
  to the last valid value), typed values are clamped to `--max-force`, and increases ramp at
  40 N/s, while reductions, reversals and **Zero force** act at once. Going back to Position re-targets the jaws where they are, so
  nothing jumps. The robot logs a warning on that switch, and that warning is the guard working.

## Safety envelope

The robot applies these on every control cycle, from the gripper config's `force_control` block
(`i2rt/robots/config/<gripper>.yml`):

| Guard | Default | Effect |
|---|---|---|
| `max_force` | 50 N | Commands are clamped to this. `--max-force` (default 20 N) bounds the slider further. |
| `max_torque` | 1.5 Nm | Motor-torque clamp after the force map. |
| `wall_margin` | 0.05 | Within 5 % of stroke of a stop, torque toward it tapers to zero at the stop. Torque away from a stop always passes. |
| `max_speed` | 2.0 /s | Torque along the motion fades out between 1x and 2x this normalized speed. |

The wall means a thin object, one closing the jaws to within `wall_margin` of the closed stop,
gets proportionally less than the commanded force.

## Reading the plot

*Commanded* is the slider value. *Estimated* is the motor's measured torque mapped back through
the same force map. While the jaws move, the estimate also includes friction and damping, so
compare the two at a stall on an object. Both rest on the gripper config's force map, so neither
is an independent measurement of force.

## Known limitations

Measured on one linear_4310 (on a yam_ultra_2), with jaw travel taken by calipers:

- **The config's force map was about 19 % off.** It assumes 96 mm of jaw travel over 6.57 rad of
  motor travel; that gripper calibrates to about 5.25 rad, and its jaws travel 94.5 mm. Force
  control therefore maps force over the stroke each gripper calibrates to (linear grippers;
  the crank keeps its map), which leaves about 1.5 % from the nominal jaw travel. **Position mode's
  force limiter still uses the config's map**, so there "50 N" is still about 40 N.
- **Friction is not compensated.** Starting the jaws took 0.13–0.29 Nm at the motor (about 9–20 N
  commanded), depending on position and on the jaws' previous motion. Below that they don't move,
  and at a standstill the delivered force can sit that much above or below the command.
- **The `kd` damping sets the speed.** It is about 80 % of the total damping; at 0.5 Nm the jaws
  close at about one stroke per second.
- **Crank gripper:** near fully open the crank's force-to-torque factor approaches zero, so it barely
  starts closing from there. Close it a little in Position mode first.

## Characterizing the gripper's dynamics

`characterize_gripper.py` measures how the gripper actually responds to torque, and fits a model.
**Run it with the jaws free; nothing may be between them.** The gripper moves on its own between
trials.

```bash
python examples/gripper_force_control/characterize_gripper.py --channel can0
python examples/gripper_force_control/characterize_gripper.py --channel can0 --arm yam_ultra_2
```

The default run takes about 3 minutes and does two experiments:

1. **Breakaway.** From rest at 30 %, 50 % and 70 % of the stroke, in both directions, 3 times
   each, the motor torque ramps up from 0 at 0.2 Nm/s until the jaws move. The torque at that
   moment is the static friction. Its spread across positions, directions and repeats is what makes
   the starting threshold vary.
2. **Torque steps.** A constant torque (0.2 to 0.5 Nm by default) drives the jaws from one end of
   the free stroke to the other, in both directions, 3 times each. The fit uses the line
   `speed = (torque - moving friction) / (viscous damping + kd)`; the rise time to that speed gives
   the inertia.

Two more experiments run on request, with `--experiments` choosing which run, in order:

```bash
python examples/gripper_force_control/characterize_gripper.py --channel can0 --experiments sweep jaw_travel
```

3. **Breakaway sweep** (`sweep`, about 3.5 minutes). The breakaway ramp at 21 positions from 15 %
   to 85 % of the stroke, in both directions. Each position is approached from the far side, so the
   mechanism's slack is already taken up in the test direction. Each position's two thresholds
   split into **friction** (their average) and a position-dependent **offset** (half their
   difference; positive means something pushes the jaws open). The fit also reports whether the
   offset repeats with motor angle, which would point to motor cogging. `--sweep-points`,
   `--sweep-range` and `--sweep-repeats` set its density.
4. **Jaw travel** (`jaw_travel`, interactive; have calipers ready). The jaws are held at 5, 25, 50,
   75 and 95 % of the stroke, and you type in the gap you measure between the jaw faces, in mm. The
   measured mm per motor radian is the real force-to-torque ratio. The fit compares it with the
   gripper config's force map (`force_map_error` > 0 means real forces run high, < 0 low) and
   suggests corrected `gripper_stroke` / `motor_stroke` values. Nothing may be between the jaws
   for the sweep, but this step needs the calipers between them.

Compare a run with its fitted model, with plots:

```bash
uv run --with matplotlib python examples/gripper_force_control/compare_gripper_model.py gripper_characterization/<run>
```

Results land in `gripper_characterization/<gripper>_<time>/`:

| File | Contents |
|---|---|
| `samples.csv` | every sample of every trial: time, position, velocity, measured and commanded torque |
| `trials.csv` | one row per trial: breakaway torque, or steady speed, rise time, speed-limited flag |
| `fit.json` | breakaway statistics per direction; moving friction, viscous damping and inertia per direction; the sweep's friction and offset; the jaw travel's force-map check |
| `meta.json` | the setup: limits, stroke, force-mode `kd`, and every argument |

Things to know:

- **A step below the breakaway torque does nothing.** It's reported as "did not reach the end" and
  left out of the fit. If that happens, pass higher `--step-torques`.
- **Trials that hit the speed limit are flagged and left out of the fit**, since the limit cuts
  the torque that was actually applied.
- **The inertia depends on sampling.** It comes from a rise of a few milliseconds. `rise_samples` in
  `trials.csv` says how many samples it spanned; below about 3, treat the inertia as an order of
  magnitude only.
- Ctrl+C stops the run, keeps the trials that finished and releases the gripper.
