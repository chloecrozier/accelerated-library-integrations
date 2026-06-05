# [Library Name]

## Purpose & Prerequisites

<!-- Document the library's core purpose and what's needed to use it. -->


### Prerequisites
Warp can be be run on both CPU and GPU. 

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

## Installation & Basic Functionality
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

#### Examples in [`/examples`](/warp/examples/)

| File | Expected behavior | 
| --- | --- |
| [`gravity.py`](/warp/examples/gravity.py) |   Terminal output similar to the example above. Try toggling the cpu / cuda flag in [`gravity.py`](/warp/examples/gravity.py) with `wp.set_device("cpu")` and `wp.set_device("cuda")`. |
| [`example_mesh.py`](/warp/examples/example_mesh.py) |  This example simulates particles falling on an object mesh. You can change the object by        |


<!-- Verify installation and demonstrate basic functionality through scripts, playbooks, or demos in a scripts/ or examples/ subdirectory. -->

## Relevant Use Case

<!-- Identify a workflow within an industry (vertical) or segment (horizontal) where this library is highly relevant. Explain how it integrates into the end-to-end workflow. -->

## Helpful Links

- [Official Documentation](https://nvidia.github.io/warp/stable/)
- [GitHub Repository](https://github.com/nvidia/warp)
- [NVIDIA Developer Page](https://developer.nvidia.com/warp-python)

## Contributor

[Megan Tian](https://github.com/Megan-Tian)