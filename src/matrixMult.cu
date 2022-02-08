#include <cstdio>
#include <cstdint>

#include <curand.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/timer.h"
#include "utils/utils.cuh"


bool VerifyResult(const float* mat, const float* cublasMat, const int M, const int N);

// MxN = MxK * KxN
__global__ void MatrixMult(const float* matA, const float* matB, float* resultMat, const int M, const int N, const int K)
{
	// mapping thread id as 2D array indexes
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	// bounds check
	if (col >= N || row >= M)
		return;
	
	// mapping 2D array indexes as 1D array indexes
	// col maj (CuBLAS assumes col maj)
	int i = row + col * M;
	// matrix multiplication calc
	resultMat[i] = 0.0f;
	for (int j = 0; j < K; j++)
		resultMat[i] += matA[row + j * M] * matB[j + col * K];
}

int main()
{
	// thread properties
	constexpr int tx = 32;
	constexpr int ty = 32;
	// matrix properties
	// Matrix A = M x K
	// Matrix B - K x N
	// Result matrix = M x N
	constexpr int M = 1 << 10;
	constexpr int K = 1 << 8;
	constexpr int N = 1 << 9;
	constexpr int bytesA = M * K * sizeof(float);
	constexpr int bytesB = K * N * sizeof(float);
	constexpr int bytesC = M * N * sizeof(float);

	// allocate A and B matrices
	float* d_MatA = nullptr;
	float* d_MatB = nullptr;
	cudaCheckError(cudaMalloc(&d_MatA, bytesA));
	cudaCheckError(cudaMalloc(&d_MatB, bytesB));

	// device PRNG handler
	curandGenerator_t prng;
	{
		Timer t("Matrix initialization");
		curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_MT19937);
		// fill matrices with random values in device
		curandGenerateUniform(prng, d_MatA, M * K);
		curandGenerateUniform(prng, d_MatB, K * N);
	}

	// allocate memory for result matrix
	float* resultMat   = new float[M * N]; // host result matrix 
	float* d_ResultMat = nullptr; // device result matrix 
	cudaCheckError(cudaMalloc(&d_ResultMat, bytesC));

	// calc block size and grid size
	const dim3 threads(tx, ty); // block size
	const dim3 blocks((N + tx - 1) / tx, (M + ty - 1) / ty); // grid size
	// call cuda kernel
	{
		Timer t("MatrixMult");
		MatrixMult<<<blocks, threads>>>(d_MatA, d_MatB, d_ResultMat, M, N, K);
	}
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaMemcpy(resultMat, d_ResultMat, bytesC, cudaMemcpyDeviceToHost));

	float* cublasResultMat = new float[M * N]; // host result matrix for cuBLAS calc
	// cuBLAS handler
	cublasHandle_t handle;
	cublasCreate_v2(&handle);
	const float alpha = 1.0f;
	const float beta  = 0.0f;

	{
		Timer t("CuBLAS");
		// CuBLAS assumes matrices in column major order
		// so the returned matrix will be arranged in col maj order
		// Calculate: c = (alpha * a) * b + (beta * c)
		// MxN = MxK * KxN 
		// lda = leading dimension of A
		// ldb = leading dimension of B
		// ldc = leading dimension of C
		//          handle, operation,   operation,   M, N, K, alpha,  A,     lda,  B,  ldb, beta,  C,          ldc
		cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_MatA, M, d_MatB, K, &beta, d_ResultMat, M);
	}
	cudaCheckError(cudaGetLastError());
	cudaCheckError(cudaMemcpy(cublasResultMat, d_ResultMat, bytesC, cudaMemcpyDeviceToHost));

	{
		Timer t("Verification");
		printf("The results %s.\n", VerifyResult(resultMat, cublasResultMat, M, N) ? "match" : "don't match");
	}

	// clean up
	cudaCheckError(cudaFree(d_MatA));
	cudaCheckError(cudaFree(d_MatB));
	cudaCheckError(cudaFree(d_ResultMat));
	delete[] resultMat;
	delete[] cublasResultMat;
	cublasDestroy_v2(handle);

	d_MatA = d_MatB = d_ResultMat = resultMat = cublasResultMat = nullptr;
}

bool VerifyResult(const float* mat, const float* cublasMat, const int M, const int N)
{
	for (int i = 0; i < M * N; i++)
	{
		if ((mat[i] - cublasMat[i]) >= 0.001f) // checking error becuz we are comapring floats
			return false;
	}
	return true;
}