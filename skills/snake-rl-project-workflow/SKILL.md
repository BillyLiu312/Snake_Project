---
name: snake-rl-project-workflow
description: Manage the Snake_Project Isaac Lab reinforcement-learning workflow. Use when working on the Snake_Project repo for GitHub fork/dev-branch setup, local-to-server synchronization, Isaac Lab/rsl_rl training commands, high-parallelism tuning, TensorBoard loss/log inspection, checkpoint play/export, MuJoCo sim2sim evaluation, or troubleshooting errors such as PhysX patch buffer overflow and TensorBoard pkg_resources failures.
---

# Snake RL Project Workflow

## Overview

Use this skill to continue development and training for `Snake_Project`, a snake robot velocity-tracking assignment built on Isaac Lab and rsl_rl. Treat the local machine as the development source and the remote server as the training receiver unless the user says otherwise.

## Repo Workflow

Prefer this branch layout:

```bash
origin   https://github.com/BillyLiu312/Snake_Project.git
upstream https://github.com/Zomnk/Snake_Project.git
```

Keep `upstream` pull-only when possible:

```bash
git remote set-url --push upstream DISABLED
```

Use `dev` for custom development:

```bash
git switch dev || git switch -c dev origin/dev
git status --short --branch
```

When committing local changes:

```bash
git add <changed-files>
git commit -m "<concise message>"
git push
```

On the server, sync only from the user's fork:

```bash
cd /path/to/Snake_Project
git fetch origin
git switch dev || git switch -c dev origin/dev
git pull
python -m pip install -e source/snake_project
```

If the server is strictly a receiver and local server changes should be discarded:

```bash
git fetch origin
git reset --hard origin/dev
```

Warn before using `reset --hard` unless the user clearly wants strong synchronization.

## Project Map

The core task code lives under:

```text
source/snake_project/snake_project/tasks/manager_based/velocity_tracking/
```

Key files:

- `velocity_env_cfg.py`: central Isaac Lab environment config: commands, observations, actions, events, rewards, terminations, curriculum, sim settings.
- `config/Snake_7DOF/flat_env_cfg.py`: flat training/play variants.
- `config/Snake_7DOF/agents/rsl_rl_ppo_cfg.py`: PPO runner and algorithm settings.
- `mdp/virtual_chassis.py`: Virtual Chassis frame and velocity computation.
- `mdp/commands.py`: velocity command sampling and visualization.
- `mdp/observations.py`: actor/critic observation terms.
- `mdp/rewards.py`: tracking and gait-shaping reward terms.
- `mdp/events.py`: reset and randomization helpers.
- `mdp/terminations.py`: invalid state and timeout termination.

Assignment outputs usually come from:

```text
logs/rsl_rl/snake_velocity_flat_tracking/<run>/
sim2sim/eval_output/
```

## Training

Standard server training command:

```bash
python scripts/rsl_rl/train.py \
  --task Snake-VelocityTracking-Flat-v0 \
  --num_envs 4096 \
  --headless \
  --logger tensorboard \
  --run_name <run-name>
```

For speed tuning, increase `--num_envs` gradually and compare throughput:

```bash
--num_envs 4096
--num_envs 8192
--num_envs 12288
--num_envs 16384
```

Do not judge by GPU utilization alone. Compare iteration time, environment steps per second, reward progress, and stability. If GPU memory is low and GPU utilization is below saturation, trying `8192` is usually reasonable.

If a run reports:

```text
PhysX error: Patch buffer overflow detected
```

increase `self.sim.physx.gpu_max_rigid_patch_count` in `velocity_env_cfg.py`. A practical value for high parallelism is:

```python
self.sim.physx.gpu_max_rigid_patch_count = 2**20
```

Increase further, for example `2**21`, only if needed.

## TensorBoard And Logs

Start training with:

```bash
--logger tensorboard
```

Open TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/snake_velocity_flat_tracking --host 0.0.0.0 --port 6006
```

If TensorBoard fails with:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

use a setuptools version that still provides `pkg_resources`:

```bash
python -m pip install --force-reinstall setuptools==80.9.0
python -c "import pkg_resources; print('pkg_resources ok')"
```

Check whether TensorBoard event files exist:

```bash
find logs/rsl_rl/snake_velocity_flat_tracking -name "events.out.tfevents.*"
```

Inspect available scalar tags:

```bash
tensorboard --inspect --logdir logs/rsl_rl/snake_velocity_flat_tracking
```

For terminal-only monitoring, run with `tee`:

```bash
python scripts/rsl_rl/train.py ... 2>&1 | tee train_<run-name>.log
tail -f train_<run-name>.log
```

Interpret common metrics:

- `Mean reward` rising usually means learning is improving.
- `Value function loss` exploding or oscillating hard suggests critic instability.
- `Surrogate loss` should fluctuate but not blow up.
- `Mean action noise std` usually decreases over training; collapsing too early may indicate premature convergence.
- `Mean episode length` near the maximum suggests fewer invalid-state terminations.

## Checkpoint Play And Export

To run a checkpoint:

```bash
python scripts/rsl_rl/play.py \
  --task Snake-VelocityTracking-Flat-Play-v0 \
  --checkpoint logs/rsl_rl/snake_velocity_flat_tracking/<run>/model_<iter>.pt \
  --headless \
  --video \
  --cmd_vx 0.2 \
  --cmd_vy 0.0
```

`play.py` exports a TorchScript policy under:

```text
logs/rsl_rl/snake_velocity_flat_tracking/<run>/exported/policy.pt
```

Use the exported `policy.pt` for MuJoCo sim2sim:

```bash
python sim2sim/sim2sim_mujoco.py \
  --policy logs/rsl_rl/snake_velocity_flat_tracking/<run>/exported/policy.pt \
  --cmd_vx 0.2 \
  --cmd_vy 0.0
```

Run the assignment's 25-command evaluation:

```bash
python sim2sim/sim2sim_eval.py \
  --policy logs/rsl_rl/snake_velocity_flat_tracking/<run>/exported/policy.pt
```

## Version Change Records

After each meaningful training/evaluation cycle, update the project version-change record before deciding which policy is the current submission candidate:

```text
docs/training_version_changes.md
```

Use git history as the source of truth for code changes:

```bash
git log --oneline --decorate --max-count 12
git show --stat <commit>
git show <commit> -- <changed-task-files>
```

For each version/run, record:

- run name and representative checkpoint;
- exported JIT policy path;
- git commit hash(es) that define the version;
- concrete changes to commands, observations, rewards, curriculum, domain randomization, PPO settings, and helper scripts;
- TensorBoard training behavior, especially reward, episode length, value loss, velocity errors, action noise, and invalid-state termination;
- MuJoCo `sim2sim_eval.py` output directory and MAE summary;
- whether the version improves or worsens the current best submission candidate.

Do not judge a version only by Isaac Lab reward. Prefer the policy with the best MuJoCo Virtual Chassis MAE, since that is the assignment evaluation target. If a version has high Isaac-side reward but worse MuJoCo MAE, document that as a failed transfer experiment rather than treating it as the new best.

## Code Editing Guidance

For assignment changes, prefer editing the allowed task area:

```text
source/snake_project/snake_project/tasks/manager_based/velocity_tracking/
```

Keep actor observations restricted to robot-local/proprioceptive information plus commands. Critic observations may include privileged terms such as Virtual Chassis velocity when useful. Keep reward changes interpretable: direct velocity tracking first, then small gait-shaping and smoothness terms.

After edits, validate locally when Isaac Lab is unavailable:

```bash
python3 -m compileall -q source/snake_project/snake_project/tasks/manager_based/velocity_tracking scripts sim2sim
git diff --check
```

If Isaac Lab is installed, additionally run:

```bash
python scripts/list_envs.py --headless
```
