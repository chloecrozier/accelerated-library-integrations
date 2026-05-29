#include <cudnn.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t status = (call);                                             \
        if (status != cudaSuccess) {                                             \
            std::cerr << "CUDA error: " << cudaGetErrorString(status)            \
                      << " at line " << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

#define CHECK_CUDNN(call)                                                       \
    do {                                                                        \
        cudnnStatus_t status = (call);                                           \
        if (status != CUDNN_STATUS_SUCCESS) {                                    \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status)          \
                      << " at line " << __LINE__ << std::endl;                  \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (0)

int main() {
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));

    const int n = 1;
    const int c = 3;
    const int h = 32;
    const int w = 32;
    const int k = 16;
    const int r = 3;
    const int s = 3;

    cudnnTensorDescriptor_t x_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnTensorDescriptor_t z_desc;
    cudnnTensorDescriptor_t bias_desc;
    cudnnActivationDescriptor_t relu_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&z_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&relu_desc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, k, c, r, s));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        relu_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));

    int out_n;
    int out_c;
    int out_h;
    int out_w;
    CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, x_desc, w_desc, &out_n, &out_c, &out_h, &out_w));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        z_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_c, 1, 1));

    const size_t x_bytes = n * c * h * w * sizeof(float);
    const size_t w_bytes = k * c * r * s * sizeof(float);
    const size_t y_bytes = out_n * out_c * out_h * out_w * sizeof(float);
    const size_t bias_bytes = out_c * sizeof(float);

    float *x = nullptr;
    float *filter = nullptr;
    float *z = nullptr;
    float *bias = nullptr;
    float *y = nullptr;
    CHECK_CUDA(cudaMalloc(&x, x_bytes));
    CHECK_CUDA(cudaMalloc(&filter, w_bytes));
    CHECK_CUDA(cudaMalloc(&z, y_bytes));
    CHECK_CUDA(cudaMalloc(&bias, bias_bytes));
    CHECK_CUDA(cudaMalloc(&y, y_bytes));

    CHECK_CUDA(cudaMemset(x, 1, x_bytes));
    CHECK_CUDA(cudaMemset(filter, 1, w_bytes));
    CHECK_CUDA(cudaMemset(z, 0, y_bytes));
    CHECK_CUDA(cudaMemset(bias, 0, bias_bytes));
    CHECK_CUDA(cudaMemset(y, 0, y_bytes));

    const cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_bytes));

    void *workspace = nullptr;
    if (workspace_bytes > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_bytes));
    }

    const float alpha1 = 1.0f;
    const float alpha2 = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBiasActivationForward(
        handle,
        &alpha1,
        x_desc,
        x,
        w_desc,
        filter,
        conv_desc,
        algo,
        workspace,
        workspace_bytes,
        &alpha2,
        z_desc,
        z,
        bias_desc,
        bias,
        relu_desc,
        y_desc,
        y));

    CHECK_CUDA(cudaDeviceSynchronize());
    std::cout << "PASS: fused Conv + Bias + ReLU completed with output shape "
              << out_n << "x" << out_c << "x" << out_h << "x" << out_w
              << std::endl;

    if (workspace) {
        CHECK_CUDA(cudaFree(workspace));
    }
    CHECK_CUDA(cudaFree(x));
    CHECK_CUDA(cudaFree(filter));
    CHECK_CUDA(cudaFree(z));
    CHECK_CUDA(cudaFree(bias));
    CHECK_CUDA(cudaFree(y));

    CHECK_CUDNN(cudnnDestroyActivationDescriptor(relu_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(z_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(w_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc));
    CHECK_CUDNN(cudnnDestroy(handle));

    return 0;
}
