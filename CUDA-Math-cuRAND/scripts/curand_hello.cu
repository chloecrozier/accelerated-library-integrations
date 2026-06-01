// curand_hello.cu

#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>

int main() {
  const int n = 10;
  float *d_data, h_data[n];
  curandGenerator_t gen;

  cudaMalloc(&d_data, n * sizeof(float));
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateUniform(gen, d_data, n);
  cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);

  printf("cuRAND hello world:\n");
  for (int i = 0; i < n; i++) printf("%f\n", h_data[i]);
  curandDestroyGenerator(gen);
  cudaFree(d_data);
  return 0;
}
