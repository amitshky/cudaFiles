/**
 * Shared Memory:
 * 1. should be declared with __shared__
 * 2. shared memory resides in a block
 * 3. lifetime of a block
 * 4. thread synchronization is required: __syncthreads() // synchronizes all the threads in a block
 * 
 * shared memory can also be initialized dynamically
 * extern __shared__ int mem[];
 * Kernel<<<blocksPerGrid, threadsPerBlock, dynamicSharedMemSize>>>()
 */

#include <cstdio>
#include <cuda_runtime.h>

#include "utils/utils.cuh"
#include "utils/timer.h"

template<int ThreadsPerBlock, int BlocksPerGrid>
__global__ void DotProduct(const float* vecA, const float* vecB, float* tempResultVec, const int vecSize)
{
	// allocate a shared memory block
	// stores running sum of products 
	__shared__ float cache[ThreadsPerBlock];
	// thread id and cache id
	int tid     = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheId = threadIdx.x; 

	// each thread calcs product more than once
	// calc running sum of products 
	float temp = 0.0f;
	while (tid < vecSize)
	{
		temp += vecA[tid] * vecB[tid];
		//tid  += blockDim.x * gridDim.x; // shift by the total number of threads in a grid
		tid  += ThreadsPerBlock * BlocksPerGrid; // shift by the total number of allocated threads in a grid
	}
	cache[cacheId] = temp;
	__syncthreads(); // synchronize all threads in a block

	// reduction
	// number of threads per block should be power of 2
	// each block will add 2 of the values in the cache
	for (int i = blockDim.x / 2; i != 0; i /= 2)
	{
		if (cacheId < i)
			cache[cacheId] += cache[cacheId + i];
		__syncthreads(); // synchronize all threads in a block
	}

	// the sum will accumulate at the 0th index of the cache
	// each block will have one temp sum 
	if (cacheId == 0)
		tempResultVec[blockIdx.x] = cache[0];
	// the reset of the calc (summing up all the values in tempResultVec) can be performed in the CPU
}

int main()
{
	constexpr int vecSize         = (1 << 10) << 10;
	constexpr int threadsPerBlock = 256;
	constexpr int blocksPerGrid   = (vecSize + threadsPerBlock - 1) / threadsPerBlock;

	// allocate host memory
	float* vecA = new float[vecSize];
	float* vecB = new float[vecSize];
	float* vecC = new float[blocksPerGrid]; // result vector
	// initialize vectors
	for (int i = 0; i < vecSize; i++)
	{
		vecA[i] = 1.0f;
		vecB[i] = 2.0f;
	}

	// allocate device meory
	float* d_VecA = nullptr;
	float* d_VecB = nullptr;
	float* d_VecC = nullptr;  // result vector
	cudaCheckError(cudaMalloc(&d_VecA, vecSize * sizeof(float)));
	cudaCheckError(cudaMalloc(&d_VecB, vecSize * sizeof(float)));
	cudaCheckError(cudaMalloc(&d_VecC, blocksPerGrid * sizeof(float)));
	// copy from host to device
	cudaCheckError(cudaMemcpy(d_VecA, vecA, vecSize * sizeof(float), cudaMemcpyHostToDevice));
	cudaCheckError(cudaMemcpy(d_VecB, vecB, vecSize * sizeof(float), cudaMemcpyHostToDevice));

	printf("Vector size = %d\nKernel info:\n  Threads per block = %d\n  Blocks per grid   = %d\n", 
		vecSize, threadsPerBlock, blocksPerGrid);

	{
		Timer t("Dot product in GPU");
		DotProduct<threadsPerBlock, blocksPerGrid><<<blocksPerGrid, threadsPerBlock>>>(d_VecA, d_VecB, d_VecC, vecSize);
	}
	cudaCheckError(cudaGetLastError());
	// copy from device to host
	cudaCheckError(cudaMemcpy(vecC, d_VecC, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
	// finishing up dot product calc
	float result = 0.0f;
	for (int i = 0; i < blocksPerGrid; i++)
		result += vecC[i];
	
	printf("GPU Result = %f\n", result);

	// verifying result
	{
		Timer t("Dot product in CPU");
		result = 0.0f;
		for (int i = 0; i < vecSize; i++)
			result += vecA[i] * vecB[i];
	}
	
	printf("CPU Result = %f\n", result);

	// clean up
	delete[] vecA;
	delete[] vecB;
	delete[] vecC;

	cudaCheckError(cudaFree(d_VecA));
	cudaCheckError(cudaFree(d_VecB));
	cudaCheckError(cudaFree(d_VecC));

	vecA = vecB = vecC = d_VecA = d_VecB = d_VecC = nullptr;
}