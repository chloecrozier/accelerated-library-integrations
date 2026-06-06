# NVIDIA Warp

## Purpose & Prerequisites

[NVIDIA Warp](https://github.com/NVIDIA/warp) is a Python framework for writing high-performance simulation and robotics code that runs on NVIDIA GPUs (and CPUs). You write kernels in a Python-like syntax; Warp compiles them to CUDA or CPU backends and manages arrays, launches, and device placement.

This integration repo shows how to install Warp, run starter kernels and bundled examples, and benchmark GPU robot learning examples with the MuJoCo Warp backend in [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

### Prerequisites
Warp runs on CPU or GPU. 

Hardware:
- CPU: x86-64 and ARMv8 on Windows and Linux, Apple Silicon (ARMv8) on macOS
- GPU (optional): CUDA-enabled NVIDIA GPU + drivers installed
    - CUDA 12.x requires driver 525 or newer
    - CUDA 13.x requires driver 580 or newer

Software:
- Python 3.10+
    - This set of examples was developed on Python 3.10.12
- Numpy

For other setups (e.g. latest nightly build, Docker, Omniverse), refer to the full [installation guide](https://nvidia.github.io/warp/stable/user_guide/installation.html). Some features may require [additional dependencies](https://nvidia.github.io/warp/stable/user_guide/installation.html) that do not support the latest version of Python.

> All examples developed in this repo were run on an NVIDIA L4 Brev instance

## Installation & Basic Functionality

Install Warp, confirm the bundled FEM example runs on your device, then try the local scripts below.

### 1. Install Warp and dependencies
```
cd accelerated-library-integrations/warp
pip install -r requirements.txt     # install warp and a set of examples
```

### 2. Verify installation
Check that warp examples were successfully installed via `warp-land[examples]`:
```$
accelerated-library-integrations/warp $  python3 -m warp.examples.fem.example_diffusion
```
You should see something like the following in your terminal:
```
Warp 1.14.0 initialized:
   CUDA Toolkit 12.9, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA L4" (22 GiB, sm_89, mempool enabled)
   Kernel cache:
     /home/ubuntu/.cache/warp/1.14.0

     ...

Module warp._src.fem.space.basis_space.dyn.fill_node_positions_93a2f4f4 e693b8f load on device 'cuda:0' took 0.80 ms  (cached)
```

#### Examples

| File | Expected behavior | 
| --- | --- |
| [`examples/gravity.py`](/warp/examples/gravity.py) |   Terminal output similar to the example above. Try toggling the cpu / cuda flag in [`gravity.py`](/warp/examples/gravity.py) with `wp.set_device("cpu")` and `wp.set_device("cuda")`. |
| [`examples/example_mesh.py`](/warp/examples/example_mesh.py) | PBD particle simulation with mesh collision. Run `python3 examples/example_mesh.py` (default: `bunny` mesh). Swap meshes with `--object` flag. USD assets live in [`examples/assets/`](/warp/examples/assets/). Writes rendered simulation to `example_mesh.usd` and prints a success message when done. |
| [`mujoco_examples/`](/warp/mujoco_examples/) | PPO training on `WalkerRun` comparing `--impl jax` vs `--impl warp`. See [mujoco_examples/README.md](/warp/mujoco_examples/README.md). |

## Relevant Use Case

**RL training — JAX vs MuJoCo Warp**

Robot policy training is often simulation-bound: PPO and similar algorithms need millions of environment steps, and wall-clock time can be dominated by physics rollouts rather than the policy network.

[`mujoco_examples/mujoco_jax_warp.py`](/warp/mujoco_examples/mujoco_jax_warp.py) runs the same PPO job on `WalkerRun` with `--impl jax` (MuJoCo MJX) and `--impl warp` (MuJoCo Warp). Each run logs `eval/episode_reward`, JIT compile time, and total training time, then renders a rollout video from the final checkpoint.

JAX/MJX is a strong default for small classic-control tasks where per-step physics is light. Warp is built for GPU-native batch simulation and usually scales better when rollouts are expensive—locomotion with contacts, manipulation, vision rendering, or thousands of parallel environments. `WalkerRun` is a contact-rich locomotion benchmark that reflects the workloads where MuJoCo Warp is meant to shine.

```bash
cd warp/mujoco_examples
python3 mujoco_jax_warp.py
```

For a single backend comparison after [building Playground](/warp/mujoco_examples/README.md#build-mujoco-playground):

```bash
cd mujoco_playground && source .venv/bin/activate
MUJOCO_GL=egl uv --no-config run train-jax-ppo \
  --env_name=WalkerRun --impl=warp --num_timesteps=150000000 --seed=1
```

Use this workflow when scaling RL experiments to thousands of parallel worlds—locomotion, manipulation, or custom  environments where GPU batching matters more than single-env latency.

## Helpful Links
- [Official Documentation](https://nvidia.github.io/warp/stable/)
- [GitHub Repository](https://github.com/nvidia/warp)
- [NVIDIA Developer Page](https://developer.nvidia.com/warp-python)

## Contributor

[Megan Tian](https://github.com/Megan-Tian)