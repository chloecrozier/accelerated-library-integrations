#!/usr/bin/env python3
"""Compare JAX vs Warp PPO training on WalkerRun."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PLAYGROUND_DIR = Path(__file__).resolve().parent / "mujoco_playground"
LOGDIR_PATTERN = re.compile(r"Logs are being stored in: (.+)")
OUTPUT_FILENAME = "terminal_output.txt"
REPLAY_OUTPUT_FILENAME = "replay_output.txt"
ROLLOUT_VIDEO_NAME = "rollout0.mp4"
REWARD_PATTERN = re.compile(r"^(\d+): reward=([\d.]+)")
TRAIN_TIME_PATTERN = re.compile(r"Time to train: ([\d.]+)")
JIT_TIME_PATTERN = re.compile(r"Time to JIT compile: ([\d.]+)")
REPLAY_LOGDIR_PATTERN = re.compile(r"Logs are being stored in: (.+)")

VERIFY_COMMANDS = [
    ["uv", "--no-config", "run", "python", "-c", "import mujoco_playground; print('Success')"],
    [
        "uv",
        "--no-config",
        "run",
        "python",
        "-c",
        "from mujoco_playground import dm_control_suite; "
        "dm_control_suite.load('WalkerRun'); print('Success')",
    ],
]

# Shared hyperparameters for a fair JAX vs Warp comparison on WalkerRun.
# Other PPO settings (batch_size, learning_rate, unroll_length, etc.) come from
# the tuned defaults in mujoco_playground/config/dm_control_suite_params.py.
TRAINING_BASE: dict[str, object] = {
    "env_name": "WalkerRun",
    "num_timesteps": 150_000_000,
    "seed": 1,
    "num_videos": 0,
    "use_tb": True,
}

TRAINING_SWEEP = [
    {**TRAINING_BASE, "impl": "jax", "suffix": "jax"},
    {**TRAINING_BASE, "impl": "warp", "suffix": "warp"},
]


@dataclass
class RunResult:
    impl: str
    returncode: int
    log_dir: Path | None = None
    log_path: Path | None = None
    final_step: int | None = None
    final_reward: float | None = None
    train_time_s: float | None = None
    jit_time_s: float | None = None
    replay_time_s: float | None = None
    video_path: Path | None = None


def build_train_command(config: dict[str, object]) -> list[str]:
    cmd = ["uv", "--no-config", "run", "train-jax-ppo"]
    for key, value in config.items():
        cmd.append(f"--{key}={value}")
    return cmd


def build_replay_command(
    config: dict[str, object],
    *,
    checkpoint_dir: Path,
) -> list[str]:
    """Render a rollout video from a finished training checkpoint."""
    replay_config = {
        key: value
        for key, value in config.items()
        if key not in {"num_timesteps", "num_videos", "use_tb", "suffix"}
    }
    replay_config.update(
        {
            "play_only": True,
            "run_evals": False,
            "num_videos": 1,
            "load_checkpoint_path": checkpoint_dir,
            "suffix": f"{config['suffix']}-replay",
            "use_tb": False,
        }
    )
    return build_train_command(replay_config)


def copy_rollout_video(*, replay_log_dir: Path, train_log_dir: Path) -> Path | None:
    src = replay_log_dir / ROLLOUT_VIDEO_NAME
    if not src.is_file():
        return None

    dst = train_log_dir / ROLLOUT_VIDEO_NAME
    shutil.copy2(src, dst)
    return dst


def parse_run_output(text: str) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {}

    reward_matches = REWARD_PATTERN.findall(text)
    if reward_matches:
        final_step, final_reward = reward_matches[-1]
        metrics["final_step"] = int(final_step)
        metrics["final_reward"] = float(final_reward)

    if match := TRAIN_TIME_PATTERN.search(text):
        metrics["train_time_s"] = float(match.group(1))
    if match := JIT_TIME_PATTERN.search(text):
        metrics["jit_time_s"] = float(match.group(1))

    return metrics


def run_command(
    cmd: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    terminal_log_path: Path | None = None,
    terminal_filename: str = OUTPUT_FILENAME,
) -> tuple[int, Path | None, str]:
    """Run a command, echo output live, and tee terminal output when known."""
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
    )

    log_file = None
    log_path = terminal_log_path
    detected_log_dir: Path | None = None
    buffered: list[str] = []
    captured: list[str] = []

    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured.append(line)

        if log_file is None:
            buffered.append(line)
            match = REPLAY_LOGDIR_PATTERN.search(line)
            if match:
                detected_log_dir = Path(match.group(1).strip())
                if log_path is None:
                    log_path = detected_log_dir / terminal_filename
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("w", encoding="utf-8")
                log_file.writelines(buffered)
                log_file.flush()
            elif log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("w", encoding="utf-8")
                log_file.writelines(buffered)
                log_file.flush()
        else:
            log_file.write(line)
            log_file.flush()

    returncode = process.wait()
    if log_file is not None:
        log_file.close()
        print(f"Saved terminal output to: {log_path}", flush=True)
    elif returncode == 0 and terminal_log_path is None:
        print(
            "Warning: command succeeded but no log directory was detected in output.",
            flush=True,
        )

    return returncode, detected_log_dir, "".join(captured)


def run_replay(
    config: dict[str, object],
    *,
    train_log_dir: Path,
    env: dict[str, str],
) -> tuple[int, float | None, Path | None]:
    checkpoint_dir = train_log_dir / "checkpoints"
    if not checkpoint_dir.is_dir():
        print(f"No checkpoints found at {checkpoint_dir}", file=sys.stderr)
        return 1, None, None

    cmd = build_replay_command(config, checkpoint_dir=checkpoint_dir)
    replay_start = time.monotonic()
    returncode, replay_log_dir, _ = run_command(
        cmd,
        env=env,
        cwd=PLAYGROUND_DIR,
        terminal_log_path=train_log_dir / REPLAY_OUTPUT_FILENAME,
        terminal_filename=REPLAY_OUTPUT_FILENAME,
    )
    replay_time_s = time.monotonic() - replay_start

    video_path = None
    if returncode == 0 and replay_log_dir is not None:
        video_path = copy_rollout_video(
            replay_log_dir=replay_log_dir,
            train_log_dir=train_log_dir,
        )
        if video_path is not None:
            print(f"Saved rollout video to: {video_path}", flush=True)
        else:
            print(
                f"Replay finished but {ROLLOUT_VIDEO_NAME} was not found in "
                f"{replay_log_dir}",
                file=sys.stderr,
            )
            returncode = 1

    return returncode, replay_time_s, video_path


def print_summary(results: list[RunResult]) -> None:
    print("\n=== WalkerRun JAX vs Warp summary ===", flush=True)
    header = (
        f"{'impl':<6} {'status':<8} {'final_step':>12} {'final_reward':>14} "
        f"{'train_s':>10} {'jit_s':>10} {'replay_s':>10}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for result in results:
        status = "ok" if result.returncode == 0 else "failed"
        final_step = str(result.final_step) if result.final_step is not None else "-"
        final_reward = (
            f"{result.final_reward:.3f}" if result.final_reward is not None else "-"
        )
        train_time = (
            f"{result.train_time_s:.1f}" if result.train_time_s is not None else "-"
        )
        jit_time = f"{result.jit_time_s:.1f}" if result.jit_time_s is not None else "-"
        replay_time = (
            f"{result.replay_time_s:.1f}" if result.replay_time_s is not None else "-"
        )
        print(
            f"{result.impl:<6} {status:<8} {final_step:>12} {final_reward:>14} "
            f"{train_time:>10} {jit_time:>10} {replay_time:>10}",
            flush=True,
        )

    successful = [r for r in results if r.returncode == 0]
    if len(successful) == 2:
        jax_run, warp_run = successful
        if (
            jax_run.final_reward is not None
            and warp_run.final_reward is not None
            and warp_run.final_reward > jax_run.final_reward
        ):
            print(
                f"\nWarp final reward is higher by "
                f"{warp_run.final_reward - jax_run.final_reward:.3f}.",
                flush=True,
            )
        if (
            jax_run.train_time_s is not None
            and warp_run.train_time_s is not None
            and warp_run.train_time_s < jax_run.train_time_s
        ):
            speedup = jax_run.train_time_s / warp_run.train_time_s
            print(
                f"Warp trained faster by {speedup:.2f}x "
                f"({jax_run.train_time_s:.1f}s vs {warp_run.train_time_s:.1f}s).",
                flush=True,
            )

    log_dirs = [r.log_dir for r in results if r.log_dir is not None]
    if log_dirs:
        print("\nRun artifacts:", flush=True)
        for result in results:
            if result.log_dir is not None:
                print(f"  {result.impl}: {result.log_dir}", flush=True)
                if result.video_path is not None:
                    print(f"    video: {result.video_path}", flush=True)
        print(
            "\nTensorBoard (install once: uv pip install tensorboard):",
            flush=True,
        )
        print(
            f"  cd {PLAYGROUND_DIR} && tensorboard --logdir logs",
            flush=True,
        )
        print(
            "  Compare eval/episode_reward for JAX vs Warp runs in the Scalars tab.",
            flush=True,
        )


def main() -> int:
    if not PLAYGROUND_DIR.is_dir():
        print(f"Expected playground directory at {PLAYGROUND_DIR}", file=sys.stderr)
        return 1

    env = {**os.environ, "MUJOCO_GL": "egl"}
    results: list[RunResult] = []

    for cmd in VERIFY_COMMANDS:
        returncode, _, _ = run_command(cmd, env=env, cwd=PLAYGROUND_DIR)
        if returncode != 0:
            print(f"Verification failed: {' '.join(cmd)}", file=sys.stderr)
            return returncode

    for config in TRAINING_SWEEP:
        impl = str(config["impl"])
        cmd = build_train_command(config)
        returncode, _, output = run_command(cmd, env=env, cwd=PLAYGROUND_DIR)
        metrics = parse_run_output(output)
        train_log_dir = None
        match = REPLAY_LOGDIR_PATTERN.search(output)
        if match:
            train_log_dir = Path(match.group(1).strip())
        log_path = train_log_dir / OUTPUT_FILENAME if train_log_dir else None

        replay_returncode = 0
        replay_time_s = None
        video_path = None
        if returncode == 0 and train_log_dir is not None:
            print(f"\nRendering rollout video for {impl} (not included in train_s)...", flush=True)
            replay_returncode, replay_time_s, video_path = run_replay(
                config,
                train_log_dir=train_log_dir,
                env=env,
            )

        run_returncode = returncode if returncode != 0 else replay_returncode
        results.append(
            RunResult(
                impl=impl,
                returncode=run_returncode,
                log_dir=train_log_dir,
                log_path=log_path,
                final_step=metrics.get("final_step"),  # type: ignore[arg-type]
                final_reward=metrics.get("final_reward"),  # type: ignore[arg-type]
                train_time_s=metrics.get("train_time_s"),  # type: ignore[arg-type]
                jit_time_s=metrics.get("jit_time_s"),  # type: ignore[arg-type]
                replay_time_s=replay_time_s,
                video_path=video_path,
            )
        )
        if returncode != 0:
            print(f"Training run failed: {' '.join(cmd)}", file=sys.stderr)
            print_summary(results)
            return returncode
        if replay_returncode != 0:
            print(f"Replay run failed for {impl}", file=sys.stderr)
            print_summary(results)
            return replay_returncode

    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
