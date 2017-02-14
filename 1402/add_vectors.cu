#include <iostream>
#include <stdio.h>

__global__ void vectorAddition(float *d_A, float *d_B, float *d_C) {
  int global_index = blockIdx.x * blockDim.x + threadIdx.x;
  d_C[global_index] = d_A[global_index] + d_B[global_index];
  return;
}

int main() {
  cudaEvent_t start,stop;
  float time = 0;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);

  int N = 1024;
  float *h_A = (float *) malloc(N * sizeof(float));
  float *h_B = (float *) malloc(N * sizeof(float));
  float *h_C = (float *) malloc(N * sizeof(float));

  for (int i = 0; i < N; ++i) {
    h_A[i] = i;
    h_B[i] = i - 1;
  }

  float *d_A;
  float *d_B;
  float *d_C;

  cudaMalloc((void **) &d_A, N * sizeof(float));
  cudaMalloc((void **) &d_B, N * sizeof(float));
  cudaMalloc((void **) &d_C, N * sizeof(float));

  cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

  vectorAddition << < N / 256, 256 >> > (d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

cudaEventRecord(stop,0);
cudaEventSynchronize(stop);

  for(int i = 0; i < N; ++i){
    std::cout<<h_C[i]<<std::endl;
  }

  free(h_A);
  free(h_B);
  free(h_C);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  cudaEventElapsedTime(&time, start, stop);
  printf("Elapsed time: %.2f ms\n", time);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}