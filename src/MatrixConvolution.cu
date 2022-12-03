#include <iostream>
#include <fstream>
#include <string>
#include "Matrix.h"

__global__ void matrConvolution(const float *matrA, const float *matrB, float *matrC, int N, int M) {
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
	if (cid < conv_size*conv_size)
		matrC[cid] = sum;
}


int main(int argc, char** argv)
{
	string fname, res_fname;
	unsigned int fsize;
	unsigned int start_time, diff_time;
	int sizeA, sizeB;
	int conv_size; //size of result - matrC size

	/*bufs for arrays*/
	float* plainA = NULL;
	float* plainB = NULL;
	float* plainC = NULL;

	/* Get file name */
	std::cout << "Enter file name > " << std::endl;
	std::cin >> fname;
	std::cout << "File name is " << fname << std::endl;
	std::ifstream fin(fname);
	
	Matrix matrA, matrB;

	try {
		matrA.readFromFile(fin);
		matrB.readFromFile(fin);

		/*Gen result filename*/
		fin.seekg(0, std::ios::end);
		fsize = fin.tellg() / (1024 * 1024);
		res_fname = "data/res_" + std::to_string(fsize) + ".txt";
		std::cout << "Result file name is " << res_fname << std::endl;
		fin.close();
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
		return -1;
	}

	sizeA = matrA.getSize();
	sizeB = matrB.getSize();
	plainA = matrA.to_plain_array();
	plainB = matrB.to_plain_array();

	conv_size = sizeA - sizeB + 1;
	plainC = new float[conv_size * conv_size];
	
	/*allocate device memory*/
	float* d_A, *d_B, *d_C;
	cudaMalloc((void **) &d_A, sizeA * sizeA * sizeof(float));
	cudaMemcpy(d_A, plainA, sizeA * sizeA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_B, sizeB * sizeB* sizeof(float));
	cudaMemcpy(d_B, plainB, sizeB * sizeB * sizeof(float), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &d_C, conv_size * conv_size * sizeof(float));
	/*kernel call*/
	int blockSize = 1024;
	int numBlocks = (conv_size*conv_size + blockSize - 1) / blockSize;
	start_time = clock();
	matrConvolution<<<numBlocks, blockSize>>>(d_A, d_B, d_C, sizeA, sizeB);
	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();
	diff_time = (clock() - start_time) / CLOCKS_PER_SEC;

	cudaMemcpy(plainC, d_C, conv_size * conv_size  * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_A); cudaFree(d_B); cudaFree (d_C);

	/*Free bufs*/
	delete[] plainA;
	delete[] plainB;
	Matrix matrC(conv_size, plainC);
	delete[] plainC;

	/* Save results */
	matrC.saveToFile(res_fname, diff_time, fsize);
	std::cout << "Time: " << diff_time << std::endl;
	return 0;
}