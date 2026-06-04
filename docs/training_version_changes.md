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

## Version 3A: `2026-06-03_13-06-52_2026-06-03_v3_stable_transfer`

Representative checkpoint:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-03_13-06-52_2026-06-03_v3_stable_transfer/model_0.pt
```

Relevant git commit:

```text
c19ec14 Tune stable transfer training v3
```

Main code changes:

- Kept actor and critic observation normalization disabled, following the Version 1 transfer direction.
- Relaxed the strong Version 2 tracking rewards back toward the gentler Version 1 shape:
  - `track_lin_vel_xy_exp` weight `6.0 -> 5.0`, `std 0.28 -> 0.4`, `linear_coef 0.8 -> 0.5`.
  - `track_ang_vel_z_exp` weight `1.5 -> 1.0`, `std 0.18 -> 0.25`.
- Restored stronger gait-shaping pressure:
  - `joint_amplitude` weight `0.12 -> 0.2`.
  - `phase_propagation` weight `0.25 -> 0.4`.
  - `motion_coordination` penalty `-0.25 -> -0.35`.
- Added a small Virtual Chassis yaw-rate penalty:
  - `virtual_chassis_yaw_rate_l1` weight `-0.05`.
- Started command curriculum from the easier Version 1-like range:
  - initial `lin_vel_x=(-0.10, 0.10)`.
  - initial `lin_vel_y=(-0.05, 0.05)`.
  - target `max_curriculum_x=0.20`, `max_curriculum_y=0.10`.

Training observations:

- The run progressed to `398/20000` iterations with full-length episodes and near-zero invalid-state termination.
- Value function loss became numerically unstable after roughly iteration `382`, rose to `inf`, and the run crashed with:

```text
RuntimeError: normal expects all elements of std >= 0.0
```

Conclusion:

This run is not an evaluation candidate. The failure happened before `model_500.pt`, so no MuJoCo evaluation or visualization was generated. The likely issue is not the V3A reward/curriculum hypothesis alone, but the PPO stability settings in this commit: learning rate was increased to `1.0e-3` while entropy pressure remained `0.01` and observation normalization was disabled. The follow-up V3A stabilization keeps the reward/curriculum hypothesis, but restores the more conservative PPO settings from the stable Version 2 training curve:

```text
value_loss_coef=0.02
entropy_coef=0.008
learning_rate=5.0e-4
desired_kl=0.008
```

## Version 3A Stabilization Attempt: `2026-06-03_13-58-05_2026-06-03_v3a_stable_ppo`

Representative checkpoint:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-03_13-58-05_2026-06-03_v3a_stable_ppo/model_0.pt
```

Main code changes relative to the first Version 3A attempt:

- Kept the Version 3A reward and command-curriculum hypothesis unchanged.
- Restored the more conservative Version 2 PPO settings:
  - `value_loss_coef=0.02`
  - `entropy_coef=0.008`
  - `learning_rate=5.0e-4`
  - `desired_kl=0.008`

Training observations:

- The run reached `394/20000` iterations with full-length episodes and near-zero invalid-state terminations.
- Isaac-side reward and velocity errors looked healthy before the failure:
  - final logged mean reward around `122.37`
  - final logged `error_vel_xy` around `0.1144`
  - final logged `error_vel_yaw` around `0.0606`
- The value function loss still exploded from low finite values to very large values, then `inf`, and the run crashed with:

```text
RuntimeError: normal expects all elements of std >= 0.0
```

Conclusion:

This attempt is also not an evaluation candidate and produced no usable checkpoint beyond `model_0.pt`. The repeated failure at roughly the same training stage suggests that disabling critic observation normalization while using privileged Virtual Chassis critic terms makes the value function numerically fragile. The next stabilization step keeps actor observation normalization disabled for policy-transfer consistency, but enables critic observation normalization and further reduces PPO update pressure:

```text
actor_obs_normalization=False
critic_obs_normalization=True
entropy_coef=0.004
learning_rate=3.0e-4
desired_kl=0.006
max_grad_norm=0.5
```

## Version 3A Critic-Normalized Stabilization: `2026-06-03_14-57-48_2026-06-03_v3a_critic_norm_stable`

Representative checkpoint:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-03_14-57-48_2026-06-03_v3a_critic_norm_stable/model_3000.pt
```

Exported JIT policy:

```text
logs/rsl_rl/snake_velocity_flat_tracking/2026-06-03_14-57-48_2026-06-03_v3a_critic_norm_stable/exported/policy.pt
```

Main code changes relative to the failed Version 3A stabilization attempt:

- Kept actor observation normalization disabled to preserve the Version 1 transfer-oriented policy input style.
- Enabled critic observation normalization to stabilize privileged critic inputs, especially Virtual Chassis velocity terms.
- Further reduced PPO update pressure:
  - `entropy_coef=0.004`
  - `learning_rate=3.0e-4`
  - `desired_kl=0.006`
  - `max_grad_norm=0.5`

Training observations at the first stability gate:

- The run passed the repeated failure region around `390-400` iterations and produced `model_500.pt`.
- At roughly iteration `541/20000`:
  - `Mean value_function loss` remained finite around `0.0019`.
  - max value loss over the last 100 logged updates was about `0.0020`.
  - `Mean reward` was around `122.56`.
  - `Mean episode length` was `1000.00`.
  - `Metrics/base_velocity/error_vel_xy` was around `0.1197`.
  - `Metrics/base_velocity/error_vel_yaw` was around `0.0605`.
  - `Episode_Termination/invalid_state` was `0.0000`.
- No `RuntimeError`, `inf`, or `nan` appeared in the value-loss tail.

Interim conclusion:

This is the first Version 3A variant that survives the previous numerical failure point. The run later produced `model_1000.pt`; at roughly iteration `1116/20000`, value loss was still finite around `0.0016`, the max value loss over the last 200 logged updates was about `0.0019`, `Mean reward` was around `122.85`, and `Episode_Termination/invalid_state` remained `0.0000`. It then produced `model_2000.pt`; at roughly iteration `2026/20000`, value loss was around `0.0014`, `Mean reward` around `123.38`, `Metrics/base_velocity/error_vel_xy` around `0.1079`, `Metrics/base_velocity/error_vel_yaw` around `0.0678`, and `Episode_Termination/invalid_state` remained `0.0000`.

MuJoCo sim2sim evaluation at `model_3000.pt`:

```text
Output directory: sim2sim/eval_output_v3a_critic_norm_model3000
Mean planar_mae: 0.5297
Mean vx_mae:     0.1818
Mean vy_mae:     0.1686
Mean wz_mae:     0.4158
Best planar_mae: 0.2089 @ vx=-0.10, vy=+0.00
Worst planar_mae: 0.8585 @ vx=-0.20, vy=-0.10
Video:           sim2sim/videos/snake_v3a_critic_norm_model3000_vx0.2_vy0.0.mp4
```

Eval-grid breakdown:

```text
Mean planar_mae by vx:
  vx=-0.20: 0.8457
  vx=-0.10: 0.5659
  vx=+0.00: 0.3228
  vx=+0.10: 0.3845
  vx=+0.20: 0.5297

Mean planar_mae by vy:
  vy=-0.10: 0.5109
  vy=-0.05: 0.5376
  vy=+0.00: 0.4612
  vy=+0.05: 0.5703
  vy=+0.10: 0.5686
```

Final conclusion:

Version 3A fixed the numerical instability but did not beat Version 1 in MuJoCo transfer. The training command curriculum stayed at `lin_vel_x=(-0.10, 0.10)` and `lin_vel_y=(-0.05, 0.05)` through `model_3000.pt`, while the README/evaluation grid tests `lin_vel_x=(-0.20, 0.20)` and `lin_vel_y=(-0.10, 0.10)`. The high error at grid boundaries is therefore a coverage failure, not a reason to keep training the same setup indefinitely. The next Version 3B should keep the stable critic-normalized PPO setup but train directly on the full eval-grid command range and prevent the curriculum from shrinking away from that range.

## Version 3B Eval-Grid Fixed Training: `2026-06-03_20-37-59_2026-06-03_v3b_eval_grid_fixed`

Main code changes relative to Version 3A critic-normalized stabilization:

- Kept the stable actor/critic setup from Version 3A:
  - `actor_obs_normalization=False`
  - `critic_obs_normalization=True`
  - `learning_rate=3.0e-4`
  - `desired_kl=0.006`
  - `max_grad_norm=0.5`
- Trained directly on the README/evaluation grid command range from the first iteration:
  - `lin_vel_x=(-0.20, 0.20)`
  - `lin_vel_y=(-0.10, 0.10)`
- Set curriculum min/max to the same eval-grid range so the command range cannot shrink away from the evaluation domain:
  - `min_curriculum_x=max_curriculum_x=0.20`
  - `min_curriculum_y=max_curriculum_y=0.10`
- Slightly increased entropy pressure from `0.004` to `0.006` to preserve exploration over the wider command range.

Initial training observation:

- The run started successfully in tmux session `snake_v3b_eval_grid_fixed`.
- The first logged iterations confirmed the intended command range:

```text
Curriculum/command/lin_vel_x_min: -0.2000
Curriculum/command/lin_vel_x_max:  0.2000
Curriculum/command/lin_vel_y_min: -0.1000
Curriculum/command/lin_vel_y_max:  0.1000
```
- The run produced `model_500.pt` and stayed numerically stable through the first gate:
  - at roughly iteration `506/20000`, `Mean value_function loss` was around `0.0159`;
  - no `RuntimeError`, `inf`, or `nan` appeared in the loss tail;
  - `Mean reward` was around `106.41`;
  - `Metrics/base_velocity/error_vel_xy` was around `0.2353`;
  - `Metrics/base_velocity/error_vel_yaw` was around `0.1068`;
  - `Episode_Termination/invalid_state` was `0.0000`;
  - command range stayed fixed at the full eval grid.
- The run then produced `model_1000.pt` and remained stable:
  - at roughly iteration `1091/20000`, `Mean value_function loss` was around `0.0156`;
  - max value loss over the last 200 logged updates was about `0.0192`;
  - `Mean action noise std` was around `0.47`;
  - `Mean reward` was around `110.22`;
  - `Metrics/base_velocity/error_vel_xy` was around `0.2247`;
  - `Metrics/base_velocity/error_vel_yaw` was around `0.0853`;
  - `Episode_Termination/invalid_state` was around `0.0001`;
  - command range stayed fixed at the full eval grid.
- The run produced `model_2000.pt`. Near iteration `1996/20000`, before the checkpoint landed:
  - `Mean value_function loss` was around `0.0124`;
  - `Mean action noise std` was around `0.49`;
  - `Mean reward` was around `110.49`;
  - `Metrics/base_velocity/error_vel_xy` was around `0.2338`;
  - `Metrics/base_velocity/error_vel_yaw` was around `0.0941`;
  - `Episode_Termination/invalid_state` was `0.0000`;
  - command range stayed fixed at the full eval grid.

Planned verification:

Evaluate `model_3000.pt` with `sim2sim_eval.py`, generate a MuJoCo video, and compare against Version 1 (`mean planar_mae=0.3336`) and Version 3A (`mean planar_mae=0.5297`).

MuJoCo sim2sim evaluation at `model_3000.pt`:

```text
Representative checkpoint: logs/rsl_rl/snake_velocity_flat_tracking/2026-06-03_20-37-59_2026-06-03_v3b_eval_grid_fixed/model_3000.pt
Exported policy:          logs/rsl_rl/snake_velocity_flat_tracking/2026-06-03_20-37-59_2026-06-03_v3b_eval_grid_fixed/exported/policy.pt
Output directory:         sim2sim/eval_output_v3b_eval_grid_fixed_model3000
Mean planar_mae:          0.2810
Mean vx_mae:              0.1159
Mean vy_mae:              0.1580
Mean wz_mae:              0.1508
Best planar_mae:          0.2063 @ vx=+0.00, vy=-0.05
Worst planar_mae:         0.3669 @ vx=+0.20, vy=+0.10
Video:                    sim2sim/videos/snake_v3b_eval_grid_fixed_model3000_vx0.2_vy0.0.mp4
```

Eval-grid breakdown:

```text
Mean planar_mae by vx:
  vx=-0.20: 0.3216
  vx=-0.10: 0.2477
  vx=+0.00: 0.2227
  vx=+0.10: 0.2715
  vx=+0.20: 0.3414

Mean planar_mae by vy:
  vy=-0.10: 0.2814
  vy=-0.05: 0.2701
  vy=+0.00: 0.2733
  vy=+0.05: 0.2820
  vy=+0.10: 0.2982
```

Final conclusion:

Version 3B is the current best evaluated submission candidate. It improves over Version 1 (`mean planar_mae 0.3336 -> 0.2810`, worst planar_mae `0.4773 -> 0.3669`) and fixes Version 3A's boundary-command failure (`mean planar_mae 0.5297 -> 0.2810`). The direct eval-grid command range was the decisive change; the stable critic-normalized PPO setup remained necessary to avoid the Version 3A numerical failures.
