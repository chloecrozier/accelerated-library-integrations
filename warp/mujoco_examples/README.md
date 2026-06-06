# Mujoco Examples

Examples for benchmarking GPU-accelerated robot learning with [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground). Playground supports two physics backends: the default **JAX** implementation of MuJoCo MJX, and **Warp** via [`mujoco_warp`](https://github.com/google-deepmind/mujoco_warp). The benchmark script trains a PPO agent on the `WalkerRun` locomotion task with both backends so you can compare sample throughput, wall-clock training time, and learned policy quality on your GPU.

**JAX vs Warp.** JAX/MJX is often fastest on small, low-contact environments (for example `CartpoleBalance`) where physics is cheap and the whole RL loop fuses cleanly through XLA. Warp tends to pull ahead as simulation cost grows with more complex scenes and more contact points.

## Requirements
- CUDA-enabled NVIDIA GPU
- Python 3.10+
- Tensorboard installed
- Linux

## Mujoco Playground
[Mujoco Playground](https://github.com/google-deepmind/mujoco_playground/tree/main#) is a suite of GPU-accelerated environments for robot learning built with [MuJoCo MJX](https://github.com/google-deepmind/mujoco/tree/main/mjx), an implementation of the MuJoCo physics engine in JAX. Mujoco Playground (installed from source at [`/mujoco_playground`](/warp/mujoco_examples/mujoco_playground/)) supports both JAX and NVIDIA Warp based backends. While JAX is performant for small environments, the Warp backend leveraging [`mujoco_warp`](https://github.com/google-deepmind/mujoco_warp) enables efficient scaling to larger and more complex scenes. 


### Build Mujoco Playground
To build `mujoco_playground` from source:
1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. From `warp/mujoco_examples`: `cd mujoco_playground`
3. Create a virtual environment: `uv venv --python 3.12`
4. Activate the venv: `source .venv/bin/activate`
5. Install CUDA 12 jax: `uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple`
    - Verify GPU backend: `python -c "import jax; print(jax.default_backend())"` should print gpu. `unset LD_LIBRARY_PATH` may need to be run before running this command.
6. Install playground from source: `uv --no-config sync --all-extras`
7. Verify installation: `uv --no-config run python -c "import mujoco_playground; print('Success')"`
    - Note: Menagerie assets will be downloaded automatically the first time you load a locomotion or manipulation environment. You can trigger this with:  uv --no-config run python -c "from mujoco_playground import locomotion; locomotion.load('G1JoystickFlatTerrain')"   



## Run benchmark

[`mujoco_jax_warp.py`](mujoco_jax_warp.py) automates a training comparison between the JAX and Warp backends. It:

1. Verifies the Playground installation and loads the `WalkerRun` environment.
2. Runs two full PPO training jobs on `WalkerRun`—one with `--impl jax` and one with `--impl warp`—using the same hyperparameters (150M timesteps, tuned defaults from `mujoco_playground/config/dm_control_suite_params.py`).
3. Renders a rollout video from each checkpoint and prints a side-by-side summary of final reward, JIT compile time, training time, and replay time.
4. Streams each run's terminal output and saves a copy to `terminal_output.txt` inside that run's log directory.

Complete the [Build Mujoco Playground](#build-mujoco-playground) steps first, then from `warp/mujoco_examples`:

```bash
python3 mujoco_jax_warp.py
```

Each run invokes `train-jax-ppo` inside `mujoco_playground/` with `MUJOCO_GL=egl` for headless rendering. Logs, checkpoints, and rollout videos are written under `mujoco_playground/logs/<env_name>-<timestamp>/`. You can also track the training runs with Tensorboard.

To run a single training job manually (for example, to try a different environment or backend):

```bash
cd mujoco_playground
source .venv/bin/activate
MUJOCO_GL=egl uv --no-config run train-jax-ppo \
  --env_name=WalkerRun \
  --impl=warp \
  --num_timesteps=150000000 \
  --seed=1
```


### Key `train-jax-ppo` flags

Flags are defined in [`mujoco_playground/learning/train_jax_ppo.py`](mujoco_playground/learning/train_jax_ppo.py). The benchmark uses a small subset; the most relevant ones are:

| Flag  | Description |
| --- | --- |
| `--env_name` | Environment to train on. Any name from Playground's registry (e.g. `WalkerRun`, `CartpoleBalance`, `G1JoystickFlatTerrain`). |
| `--impl` | Physics backend: `jax` (MuJoCo MJX) or `warp` (MuJoCo Warp). |
| `--num_timesteps`  | Total environment steps to collect during training. |
| `--episode_length` | Maximum steps per rollout episode. |
| `--num_envs` | Number of parallel training environments. Higher values increase GPU utilization but require more memory. |
| `--num_evals`  | Number of evaluation checkpoints spread across training. |
| `--num_eval_envs` | Parallel environments used during evaluation rollouts. |
| `--seed` | Random seed for reproducibility. |
| `--logdir`  | Root directory for experiment logs. |
| `--warp_kernel_cache_dir` | Directory for caching compiled Warp kernels across runs. |
| `--use_wandb`  | Log metrics to Weights & Biases. |
| `--use_tb` | Log metrics to TensorBoard. |

During training, progress lines report `eval/episode_reward` at each evaluation step. When a run finishes, the script also prints JIT compile time and total training time.