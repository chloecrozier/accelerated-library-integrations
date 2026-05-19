// curand_cmd.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand.h>

int main(int argc, char *argv[]) {    
const int n = 10;    
float *d_data = NULL;    
float h_data[n];    
curandGenerator_t gen;    
unsigned long long seed = 1234ULL;    
if (argc > 1) {        
	seed = strtoull(argv[1], NULL, 10);    
}    
cudaMalloc((void **)&d_data, n * sizeof(float));    
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);    curandSetPseudoRandomGeneratorSeed(gen, seed);    curandGenerateUniform(gen, d_data, n);    
cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);    
printf("cuRAND hello world (seed=%llu):\n", seed);    
for (int i = 0; i < n; i++) {        
printf("%f\n", h_data[i]);    }    curandDestroyGenerator(gen);    
cudaFree(d_data);    return 0;
}
