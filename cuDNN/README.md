# cuDNN

NVIDIA cuDNN is the GPU-accelerated library of deep learning primitives that
frameworks such as PyTorch, TensorFlow, and JAX rely on for high-performance
neural network execution. Most developers do not call cuDNN directly; they use
a framework, and the framework routes operations such as convolution, pooling,
normalization, activation, attention, and recurrent layers into cuDNN.

The short SA version:

> cuDNN is the hidden acceleration layer that turns ordinary deep learning code
> into optimized NVIDIA GPU work.

This project demonstrates that idea with a small convolution benchmark and a
ResNet-18 inference benchmark. The same workflow can be run on a Brev L4
instance and on DGX Spark, then compared side by side.

## Purpose & Prerequisites

cuDNN is useful when a team has deep learning models that need faster training
or inference without rewriting model code in CUDA. It provides tuned kernels,
algorithm selection, graph execution, and fusion paths for core neural network
operations.

Required for this project:

- NVIDIA GPU visible to the system
- NVIDIA driver, CUDA, and cuDNN versions that match the official cuDNN support
  matrix
- Python environment with CUDA-enabled PyTorch and torchvision
- `matplotlib` and `pandas` for charting benchmark results

This will not run with GPU acceleration on local macOS because Apple Silicon
GPUs do not support CUDA. Use Brev, DGX Spark, or another CUDA-capable NVIDIA
GPU machine.

## Installation & Basic Functionality

Create a virtual environment on the target GPU system:

```bash
cd cuDNN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-torch-cu128.txt
python -m pip install -r requirements.txt
```

The `requirements-torch-cu128.txt` file installs the CUDA 12.8 PyTorch wheels.
If the target machine needs a different CUDA wheel, use the official PyTorch
install selector and replace that one install command.

Verify the install:

```bash
python3 examples/install_verification.py
```

Expected result: the script prints the GPU name, CUDA availability, cuDNN
version, runs a small convolution on the GPU, and ends with a `PASS` message.

Run the benchmark on each platform:

```bash
python3 examples/benchmark_cudnn.py --platform brev_l4 --output results/brev_l4.csv
python3 examples/benchmark_cudnn.py --platform dgx_spark --output results/dgx_spark.csv
```

The benchmark compares:

- CPU baseline
- GPU with the cuDNN backend disabled
- GPU with cuDNN enabled
- GPU with cuDNN enabled and `torch.backends.cudnn.benchmark = True`

The "cuDNN disabled" case is a framework fallback comparison, not a generic
CUDA product benchmark. It is included to show what PyTorch can do when eligible
ops do not route through cuDNN.

The notebook version is available at
[`examples/cudnn_resnet18_benchmark.ipynb`](./examples/cudnn_resnet18_benchmark.ipynb).

## Relevant Use Case

**First Bowl of Soup: Computer vision inference for manufacturing quality
inspection**

A manufacturing partner may run camera-based inspection on an assembly line.
The application captures images of parts, preprocesses them, sends batches to a
CNN classifier or defect detector, and writes predictions into a quality
dashboard or factory control system.

cuDNN fits into the workflow under the model framework:

1. Cameras capture images from the line.
2. A preprocessing service batches and normalizes images.
3. A PyTorch vision model, such as ResNet, runs inference on an NVIDIA GPU.
4. cuDNN accelerates the convolution, normalization, activation, and pooling
   operations used by the model.
5. The application sends pass/fail labels, confidence scores, and defect
   metadata to downstream dashboards or alerting systems.

This is relevant to an SA because the integration path is low friction. A
partner can keep a familiar PyTorch workflow while cuDNN automatically maps the
model's deep learning primitives to optimized NVIDIA GPU kernels.

This connects to AIPS:

- **Accelerate:** reduce deep learning inference latency with cuDNN-backed GPU
  execution.
- **Integrate:** use cuDNN through standard frameworks rather than rewriting an
  application from scratch.
- **Promote:** show a clear before/after benchmark and an Nsight Systems view of
  fused GPU work.
- **Sell:** position NVIDIA GPUs as the easiest way to speed up existing deep
  learning products.

## Files

- `examples/install_verification.py` - checks PyTorch, CUDA, cuDNN, and one GPU
  convolution.
- `examples/benchmark_cudnn.py` - runs the Conv2d and ResNet-18 benchmark and
  writes CSV results.
- `examples/cudnn_resnet18_benchmark.ipynb` - notebook flow for the final demo
  and charts.
- `examples/README.md` - setup and runbook for Brev or DGX Spark.
- `cpp/fused_conv_bias_relu_demo.cu` - optional direct cuDNN C++ demo of a fused
  convolution, bias, and ReLU call.
- `requirements-torch-cu128.txt` - CUDA 12.8 PyTorch install requirements.
- `requirements.txt` - charting and notebook dependencies.
- `results/` - CSV outputs from Brev L4 and DGX Spark.
- `screenshots/` - benchmark plots and optional Nsight Systems screenshots.

## Helpful Links

- [Official cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/)
- [cuDNN Backend Documentation](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/)
- [cuDNN Frontend GitHub Repository](https://github.com/NVIDIA/cudnn-frontend)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)

## Contributor

akrishnakuma
