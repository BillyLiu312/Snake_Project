#!/usr/bin/env python3
"""Render a MuJoCo rollout video for the exported snake policy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from sim2sim_eval import EvalCfg, EvalRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Render MuJoCo video for a fixed velocity command.")
    parser.add_argument("--policy", type=str, required=True, help="Path to TorchScript policy.pt")
    parser.add_argument(
        "--mjcf",
        type=str,
        default="source/snake_project/snake_project/data/Snake/14DOF-DW.xml",
        help="Path to MJCF model XML",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--cmd_vx", type=float, default=0.2)
    parser.add_argument("--cmd_vy", type=float, default=0.0)
    parser.add_argument("--cmd_wz", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="sim2sim/videos/snake_vx0.2_vy0.0.mp4")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def configure_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.15], dtype=np.float64)
    camera.distance = 2.4
    camera.azimuth = 135.0
    camera.elevation = -35.0
    return camera


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = EvalCfg(
        device=args.device,
        seed=args.seed,
        cmd_vx=args.cmd_vx,
        cmd_vy=args.cmd_vy,
        cmd_wz=args.cmd_wz,
    )
    runner = EvalRunner(args.mjcf, args.policy, cfg)
    runner.reset()

    renderer = mujoco.Renderer(runner.model, height=args.height, width=args.width)
    camera = configure_camera(runner.model)
    frames = []

    sim_steps = int(args.seconds / runner.mj_dt)
    frame_interval = max(1, int(round(1.0 / (args.fps * runner.mj_dt))))
    sim_t = 0.0
    for step in range(sim_steps):
        if step % runner.decimation == 0:
            if sim_t >= 0.5:
                obs = runner._build_obs()
                action = runner._policy(obs)
                runner.last_actions[:] = action
                runner._apply_position_targets(action)

        if step % frame_interval == 0:
            # Keep the camera centered on the virtual chassis.
            origin_w, _, _, _ = runner._log_virtual_state_for_camera()
            camera.lookat[:] = np.array([origin_w[0], origin_w[1], 0.15], dtype=np.float64)
            renderer.update_scene(runner.data, camera=camera)
            frames.append(renderer.render())

        mujoco.mj_step(runner.model, runner.data)
        sim_t += runner.mj_dt

    imageio.mimsave(output_path, frames, fps=args.fps, macro_block_size=8)
    renderer.close()
    print(f"[Video] Saved: {output_path}")
    print(f"[Video] Frames: {len(frames)}  FPS: {args.fps}  Duration: {len(frames) / args.fps:.2f}s")


def _log_virtual_state_for_camera(self):
    origin_w, axes_w, lin_vel_vc, ang_vel_z_vc = self._compute_virtual_chassis_state_for_video()
    return origin_w, axes_w, lin_vel_vc, ang_vel_z_vc


def _compute_virtual_chassis_state_for_video(self):
    from sim2sim_eval import compute_virtual_chassis_state

    if not hasattr(self, "_video_prev_axes_w"):
        self._video_prev_axes_w = np.zeros((3, 3), dtype=np.float64)
        self._video_has_prev = False
    origin_w, axes_w, lin_vel_vc, ang_vel_z_vc = compute_virtual_chassis_state(
        self.data,
        self.vc_body_ids,
        self._video_prev_axes_w,
        self._video_has_prev,
    )
    self._video_prev_axes_w = axes_w
    self._video_has_prev = True
    return origin_w, axes_w, lin_vel_vc, ang_vel_z_vc


EvalRunner._log_virtual_state_for_camera = _log_virtual_state_for_camera
EvalRunner._compute_virtual_chassis_state_for_video = _compute_virtual_chassis_state_for_video


if __name__ == "__main__":
    main()
