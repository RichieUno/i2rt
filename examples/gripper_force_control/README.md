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
