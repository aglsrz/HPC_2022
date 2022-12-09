#include <iostream>
#include <fstream>
#include <string>
#include "Matrix.h"

#define MAX_THREADS 1024

__global__ void matrConvolution(const float *matrA, const float *matrB, float *matrC, int N, int M, int vertic_iter_count) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x; //new element id
	int conv_size = N - M + 1;
	int idx = cid; // element id for calculation
	if (idx >= conv_size)
		idx += M - 1;
	
	//multiply part of A with B
	float sum = 0;
	for (int i = 0; i < M; ++i)
		for (int j = 0; j < M; ++j)
			sum += matrA[idx+i*N+j] * matrB[i*M+j];
	if (cid < conv_size*vertic_iter_count)
		matrC[cid] = sum;
}

float *convoluteGPU(const float *plainA, const float *plainB, int sizeA, int sizeB, int vertic_iter_count){
	int conv_size = sizeA - sizeB + 1;
	float* plainC = NULL;
	plainC = new float[conv_size * vertic_iter_count];

	/*allocate device memory*/
	float* d_A, *d_B, *d_C;
	cudaMalloc((void **) &d_A, sizeA * sizeA * sizeof(float));
	cudaMemcpy(d_A, plainA, sizeA * sizeA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_B, sizeB * sizeB* sizeof(float));
	cudaMemcpy(d_B, plainB, sizeB * sizeB * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_C, conv_size * vertic_iter_count * sizeof(float));
	/*kernel call*/
	int blockSize = MAX_THREADS;
	int numBlocks = (conv_size*vertic_iter_count + blockSize - 1) / blockSize;

	matrConvolution<<<numBlocks, blockSize>>>(d_A, d_B, d_C, sizeA, sizeB, vertic_iter_count);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	cudaMemcpy(plainC, d_C, conv_size * vertic_iter_count  * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_A); cudaFree(d_B); cudaFree (d_C);

	/*Free bufs*/
	delete[] plainA;
	delete[] plainB;

	return plainC;
}