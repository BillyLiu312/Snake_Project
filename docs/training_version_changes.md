# Snake Velocity Training Version Changes

This document records the main training-version changes from the git history and the observed evaluation results.

## Submission Target From README

The assignment asks for:

1. The velocity-tracking task code under `source/snake_project/snake_project/tasks/manager_based/velocity_tracking`.
2. The best evaluated policy `.pt` file.
3. The MuJoCo `sim2sim_eval.py` output directory containing 25-command velocity-tracking MAE results.

The key score used here is the MuJoCo Virtual Chassis tracking MAE from `sim2sim_eval.py`.

## Version 1: `2026-06-01_10-08-55_env8192`

Representative checkpoint:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-01_10-08-55_env8192/model_3000.pt
```

Exported JIT policy:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-01_10-08-55_env8192/exported/policy.pt
```

Relevant git commits:

```text
457ac4e Improve snake velocity tracking training config
aa20745 Increase PhysX patch buffer for high parallelism
```

Main code changes:

- Increased PPO training horizon from `5000` to `20000` iterations and checkpoint interval from `200` to `500`.
- Added critic privileged Virtual Chassis observations:
  - `virtual_chassis_lin_vel`
  - `virtual_chassis_ang_vel_z`
- Added additional gait and sim2sim-oriented reward terms:
  - `joint_centering_l2`
  - `joint_velocity_l2`
  - `body_wave_smoothness`
- Reworked command range from the earlier broad range to `lin_vel_x=(-0.15, 0.15)` and `lin_vel_y=(-0.15, 0.15)`.
- Enabled command curriculum with EMA tracking reward and min environment-count gating.
- Enabled/reset-tuned domain randomization:
  - reset joint/pose/root velocity randomization
  - material friction randomization
  - link mass randomization
  - yaw actuator gain randomization
- Reduced domain randomization intensity compared with the earlier attempted broader settings:
  - friction to `static=(0.7, 1.3)`, `dynamic=(0.6, 1.2)`
  - mass scaling to `(0.95, 1.05)`
  - actuator stiffness to `(0.95, 1.05)`
- Set play-mode `planar_zero_threshold=0.0` so manual low-speed commands are not zeroed.
- Increased PhysX `gpu_max_rigid_patch_count` from `10 * 2**15` to `2**20` for high-parallelism training.

MuJoCo sim2sim evaluation:

```text
Output directory: sim2sim/eval_output
Mean planar_mae: 0.3336
Mean vx_mae:     0.1401
Mean vy_mae:     0.1414
Mean wz_mae:     0.2134
Best planar_mae: 0.2641 @ vx=+0.00, vy=+0.00
Worst planar_mae: 0.4773 @ vx=+0.20, vy=+0.10
```

Conclusion:

Version 1 is currently the best evaluated submission candidate. It has weaker Isaac-side final reward than Version 2, but transfers better to MuJoCo.

## Version 2: `2026-06-02_07-55-24_evalgrid_norm_v1`

Representative checkpoint:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-02_07-55-24_evalgrid_norm_v1/model_15500.pt
```

Exported JIT policy:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-02_07-55-24_evalgrid_norm_v1/exported/policy.pt
```

Relevant git commit:

```text
f6141fc Tune velocity tracking training
```

Main code changes:

- Aligned training command sampling more directly with the README evaluation grid:
  - `lin_vel_x=(-0.20, 0.20)`
  - `lin_vel_y=(-0.10, 0.10)`
- Tightened Virtual Chassis tracking rewards:
  - `track_lin_vel_xy_exp` weight `5.0 -> 6.0`
  - linear velocity reward `std 0.4 -> 0.28`
  - linear tracking penalty `linear_coef 0.5 -> 0.8`
  - `track_ang_vel_z_exp` weight `1.0 -> 1.5`
  - yaw-rate reward `std 0.25 -> 0.18`
- Reduced gait-shaping reward pressure:
  - `joint_amplitude` weight `0.2 -> 0.12`
  - `phase_propagation` weight `0.4 -> 0.25`
  - `motion_coordination` penalty `-0.35 -> -0.25`
- Split curriculum limits by axis:
  - `max_curriculum_x=0.25`
  - `max_curriculum_y=0.15`
  - `min_curriculum_x=0.20`
  - `min_curriculum_y=0.10`
- Lowered curriculum threshold ratio from `0.72` to `0.70`.
- Enabled actor and critic observation normalization.
- Stabilized PPO after Version 1 showed value loss instability:
  - `value_loss_coef 0.01 -> 0.02`
  - `entropy_coef 0.01 -> 0.008`
  - `learning_rate 1.0e-3 -> 5.0e-4`
  - `desired_kl 0.01 -> 0.008`
- Added `sim2sim/render_mujoco_video.py` to render MuJoCo rollout videos offscreen when Isaac video rendering is unavailable.

Training observations:

- Isaac Lab training reached stable full-length episodes.
- `invalid_state` termination was almost zero near the end.
- Value function loss stayed finite and low, unlike the earlier unstable curve.
- Mean reward plateaued around `138-140`.

MuJoCo sim2sim evaluation:

```text
Output directory: sim2sim/eval_output_evalgrid_norm_v1_model15500
Mean planar_mae: 0.5473
Mean vx_mae:     0.1627
Mean vy_mae:     0.2496
Mean wz_mae:     0.3809
Best planar_mae: 0.4291 @ vx=-0.10, vy=+0.05
Worst planar_mae: 0.7309 @ vx=+0.00, vy=+0.10
```

Conclusion:

Version 2 improved Isaac-side stability and reward, but MuJoCo transfer became significantly worse. The stronger tracking reward and observation normalization likely encouraged policy behavior that fits the Isaac training dynamics better than the MuJoCo transfer setting. Do not use Version 2 as the current best submission policy.

## Current Recommendation

Use Version 1 as the best evaluated policy unless a later version beats its MuJoCo MAE:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-01_10-08-55_env8192/exported/policy.pt
sim2sim/eval_output
```

When making a new version, compare against:

```text
Version 1 mean planar_mae: 0.3336
Version 1 worst planar_mae: 0.4773
```

The next experimental direction should be conservative:

- Keep Version 1's observation-normalization setting unless a controlled ablation proves otherwise.
- Avoid over-tightening tracking rewards before checking MuJoCo transfer.
- Test one change at a time and run `sim2sim_eval.py` before accepting the new version.
