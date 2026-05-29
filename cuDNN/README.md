# cuDNN

NVIDIA cuDNN is the GPU-accelerated library of deep learning primitives that
frameworks such as PyTorch, TensorFlow, and JAX rely on for high-performance
neural network execution. Most developers do not call cuDNN directly; they use
a framework, and the framework routes operations such as convolution, pooling,
normalization, activation, attention, and recurrent layers into cuDNN.

The short version:

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

**First Bowl of Soup: ChatGPT-style enterprise support assistant**

A partner building an internal support chatbot needs low-latency answers over
company docs, tickets, and product knowledge. cuDNN sits below the AI framework
and accelerates supported deep learning primitives, including attention paths
used by transformer models.

cuDNN fits into the workflow under the model framework:

1. A user asks a question in a chat UI.
2. A retrieval service finds relevant internal documents.
3. The prompt and context are tokenized and sent to a transformer model.
4. The framework runs inference on an NVIDIA GPU.
5. cuDNN accelerates supported operations such as scaled dot product attention,
   normalization, pointwise ops, and other neural network primitives.
6. The application streams the answer back to the user and logs feedback.

This is relevant for the integration path is low friction: partners
can keep a familiar PyTorch or framework-based workflow while cuDNN maps
supported model operations to optimized NVIDIA GPU kernels.

This connects to AIPS:

- **Accelerate:** reduce deep learning inference latency with cuDNN-backed GPU
  execution.
- **Integrate:** use cuDNN through standard frameworks rather than rewriting an
  application from scratch.
- **Promote:** show a clear before/after benchmark and an Nsight Systems view of
  cuDNN-backed GPU execution.
- **Sell:** position NVIDIA GPUs as the fastest path to lower-latency AI
  experiences without a full application rewrite.

## Files

- `examples/install_verification.py` - checks PyTorch, CUDA, cuDNN, and one GPU
  convolution.
- `examples/benchmark_cudnn.py` - runs the Conv2d and ResNet-18 benchmark and
  writes CSV results.
- `examples/profile_inference_for_nsight.py` - runs one labeled inference per
  mode for cleaner Nsight Systems screenshots.
- `examples/cudnn_resnet18_benchmark.ipynb` - notebook flow for the final demo
  and charts.
- `examples/fused_conv_bias_relu_demo.cu` - optional direct cuDNN C++ demo of a
  fused convolution, bias, and ReLU call.
- `examples/README.md` - setup and runbook for Brev or DGX Spark.
- `requirements-torch-cu128.txt` - CUDA 12.8 PyTorch install requirements.
- `requirements.txt` - charting and notebook dependencies.
- `results/` - CSV outputs from Brev L4 and DGX Spark.
- `screenshots/` - Brev L4 / DGX Spark benchmark proof screenshots, the
  presentation latency chart, and the optional Nsight Systems report.

## Helpful Links

- [Official cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/)
- [cuDNN Backend Documentation](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/)
- [cuDNN Attention Operation](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)
- [Accelerating Transformers with NVIDIA cuDNN 9](https://developer.nvidia.com/blog/accelerating-transformers-with-nvidia-cudnn-9/)
- [cuDNN Frontend GitHub Repository](https://github.com/NVIDIA/cudnn-frontend)
- [PyTorch Get Started](https://pytorch.org/get-started/locally/)
- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)

## Contributor

amrik05
