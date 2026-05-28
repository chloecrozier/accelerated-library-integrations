# Optional Direct cuDNN Demo

The main project demonstrates cuDNN the way most customers encounter it:
through PyTorch. This folder is an optional "under the hood" companion that
shows a direct cuDNN fused convolution, bias, and ReLU call.

Build on a machine with CUDA, cuDNN headers, and cuDNN libraries installed:

```bash
nvcc -std=c++17 fused_conv_bias_relu_demo.cu -lcudnn -o fused_conv_bias_relu_demo
./fused_conv_bias_relu_demo
```

If the target image does not include cuDNN development headers, skip this path
and use the Python benchmark plus an Nsight Systems profile instead.
